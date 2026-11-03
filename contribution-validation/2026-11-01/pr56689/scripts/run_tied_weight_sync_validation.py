"""Bounded A/B/A tied-weight validation on an explicitly assigned GPU list.

TIED_SYNC_GPUS lists the trainer GPU first, then TP*PP inference GPUs.
TIED_SYNC_TP/PP accept 1 or 2; TIED_SYNC_PACKED accepts 0 or 1.
Each cold control and the update arm use a fresh server process group.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time


from source_provenance import ROOT, SOURCE, verify_source, verify_model, runtime_provenance

ASSET = ROOT / "models/Qwen3-0.6B-c1899de"
BASE = "fe77d0906be0c38c5e2c575cd61b545a2fa687e4"
EXPECT_BUG = os.environ.get("TIED_SYNC_EXPECT_BUG", "0") == "1"
HARNESSES = (
    "run_tied_weight_sync_validation.py", "tied_weight_probe.py",
    "launch_tied_weight_sync_validation.py",
    "source_provenance.py", "validate_fs_rope.py", "model-revision-manifest.json",
)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def topology():
    gpus = [int(value) for value in os.environ["TIED_SYNC_GPUS"].split(",")]
    tp, pp = int(os.environ["TIED_SYNC_TP"]), int(os.environ["TIED_SYNC_PP"])
    assert tp in (1, 2) and pp in (1, 2)
    assert len(gpus) == 1 + tp * pp and len(set(gpus)) == len(gpus)
    assert all(0 <= gpu <= 7 for gpu in gpus)
    packed = os.environ["TIED_SYNC_PACKED"]
    assert packed in ("0", "1")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ",".join(map(str, gpus))
    return gpus, tp, pp, packed == "1"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def terminate_group(pid):
    try:
        if os.getpgid(pid) != pid:
            raise RuntimeError(f"Refusing reused/non-owned process leader {pid}")
    except ProcessLookupError:
        pass
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def modified_b(model):
    import torch

    with torch.no_grad():
        for layer in (0, 27):
            attn = model.model.layers[layer].self_attn
            for rank in (0, 1):
                attn.q_proj.weight[rank * (attn.q_proj.weight.shape[0] // 2), 0] = (
                    0.0625 + layer / 1024 + rank / 128
                )
                attn.o_proj.weight[0, rank * (attn.o_proj.weight.shape[1] // 2)] = (
                    0.125 + layer / 1024 + rank / 128
                )
        model.model.norm.weight.mul_(0.5)
        embedding = model.model.embed_tokens.weight
        embedding.mul_(0.75)
        for rank in (0, 1):
            embedding[rank * (embedding.shape[0] // 2), 0] = -0.1875 + rank / 64


def prepare(output):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert not torch.cuda.is_initialized()
    model = AutoModelForCausalLM.from_pretrained(
        str(ASSET), dtype=torch.bfloat16, local_files_only=True
    ).eval()
    assert model.config.tie_word_embeddings
    assert model.lm_head.weight is model.model.embed_tokens.weight
    modified_b(model)
    b_dir = output / "B-checkpoint"
    model.save_pretrained(b_dir, safe_serialization=True)
    weights = load_file(b_dir / "model.safetensors")
    weights["lm_head.weight"] = weights["model.embed_tokens.weight"].clone()
    with safe_open(ASSET / "model.safetensors", framework="pt", device="cpu") as original:
        assert set(weights) == set(original.keys())
    save_file(weights, b_dir / "model.safetensors", metadata={"format": "pt"})
    tokenizer = AutoTokenizer.from_pretrained(str(ASSET), local_files_only=True)
    texts = [
        "The laboratory recorded twelve red samples and seven blue samples. "
        "There were nineteen samples in total. The report was reviewed carefully.",
        "A distributed inference engine partitions attention projections across "
        "two devices. A weight update should reach both devices before serving resumes.",
    ]
    write(output / "prompts.json", [tokenizer.encode(text) for text in texts])
    write(output / "B-sentinels.json", {
        str(layer): {
            str(rank): {
                "q": float(model.model.layers[layer].self_attn.q_proj.weight[
                    rank * (model.model.layers[layer].self_attn.q_proj.weight.shape[0] // 2), 0]),
                "o": float(model.model.layers[layer].self_attn.o_proj.weight[
                    0, rank * (model.model.layers[layer].self_attn.o_proj.weight.shape[1] // 2)]),
            } for rank in (0, 1)
        } for layer in (0, 27)
    })
    write(output / "B-embedding-sentinels.json", {
        str(rank): float(model.model.embed_tokens.weight[
            rank * (model.model.embed_tokens.weight.shape[0] // 2), 0])
        for rank in (0, 1)
    })
    assert not torch.cuda.is_initialized()
    print("Prepared B on CPU with embedding*0.75 and explicit equal lm_head", flush=True)


def record_prepared(output, prior=None):
    import torch
    from safetensors import safe_open

    assert not torch.cuda.is_initialized()
    b_file = output / "B-checkpoint/model.safetensors"
    with safe_open(b_file, framework="pt", device="cpu") as b:
        with safe_open(ASSET / "model.safetensors", framework="pt", device="cpu") as a:
            assert set(a.keys()) == set(b.keys())
        embed, head = b.get_tensor("model.embed_tokens.weight"), b.get_tensor("lm_head.weight")
        assert embed.dtype == torch.bfloat16 and embed.shape == (151936, 1024)
        assert torch.equal(embed, head)
        sentinels = json.loads((output / "B-embedding-sentinels.json").read_text())
        for rank in (0, 1):
            expected = -0.1875 + rank / 64
            assert float(embed[rank * (embed.shape[0] // 2), 0]) == expected
            assert sentinels[str(rank)] == expected
        embedding_sha = hashlib.sha256(
            embed.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
    assert json.loads((output / "B-checkpoint/config.json").read_text())["tie_word_embeddings"]
    files = ["B-checkpoint/model.safetensors", "B-checkpoint/config.json", "prompts.json",
             "B-sentinels.json", "B-embedding-sentinels.json"]
    write(output / "prepared-evidence.json", {
        "reuse_from": str(prior) if prior else None,
        "files": {name: {"sha256": sha_file(output / name),
                         "bytes": (output / name).stat().st_size} for name in files},
        "original_weights_sha256": sha_file(ASSET / "model.safetensors"),
        "embedding_sha256": embedding_sha, "explicit_head_equals_embedding": True,
        "cuda_initialized": torch.cuda.is_initialized(),
    })
    assert not torch.cuda.is_initialized()


def http(url, path, payload=None, method="POST", timeout=120):
    import requests

    response = requests.request(method, url + path, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


@contextmanager
def server(output, arm, model):
    import requests

    gpus, tp, pp, _ = topology()
    port = free_port()
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model), "--tokenizer", str(ASSET), "--served-model-name", "m",
        "--host", "127.0.0.1", "--port", str(port),
        "--device-ids", ",".join(str(index) for index in range(1, 1 + tp * pp)),
        "--tensor-parallel-size", str(tp), "--pipeline-parallel-size", str(pp),
        "--distributed-executor-backend", "mp", "--dtype", "bfloat16",
        "--enforce-eager", "--max-model-len", "512", "--max-num-seqs", "1",
        "--max-num-batched-tokens", "512", "--kv-cache-memory-bytes", "134217728",
        "--gpu-memory-utilization", "0.1", "--enable-prefix-caching",
        "--attention-config", '{"backend":"FLASH_ATTN","flash_attn_version":2}',
        "--weight-transfer-config", '{"backend":"nccl"}',
        "--worker-extension-cls", "tied_weight_probe.TiedWeightProbe", "--seed", "0",
    ]
    env = os.environ.copy()
    env["VLLM_SERVER_DEV_MODE"] = "1"
    write(output / f"{arm}-server-command.json", {"command": command, "env": {
        key: value for key, value in env.items()
        if key.startswith("TIED_SYNC_") or key in (
            "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "VLLM_BATCH_INVARIANT",
            "VLLM_SERVER_DEV_MODE", "NO_PROXY", "no_proxy",
        )
    }})
    with (output / f"{arm}-server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        live = output / f"{arm}-live-server.json"
        write(live, {"pid": process.pid, "pgid": process.pid, "arm": arm, "port": port})
        print(f"SERVER {arm} pid={process.pid} physical_gpus={gpus[1:]}", flush=True)
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"{arm} server exited {process.returncode}; see log")
                try:
                    if requests.get(url + "/health", timeout=2).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError(f"{arm} server readiness deadline")
            yield url
        finally:
            terminate_group(process.pid)
            process.wait(timeout=15)
            live.unlink(missing_ok=True)
            write(output / f"{arm}-server-exit.json", {"returncode": process.returncode})


def observe(url, output, label):
    import requests

    gpus, tp, pp, _ = topology()
    census = http(url, "/collective_rpc", {"method": "tied_weight_census", "timeout": 60})
    rows = sorted(census["results"], key=lambda row: row["rank"])
    write(output / f"{label}-census.json", rows)
    assert [row["rank"] for row in rows] == list(range(tp * pp)), rows
    provenance = json.loads((output / "provenance.json").read_text())
    for row in rows:
        rank, stage = row["rank"], row["pp_rank"]
        assert row["tp_rank"] == rank % tp and stage == rank // tp
        visible = [int(device) for device in row["cuda_visible_devices"].split(",")]
        assert visible[row["cuda_device"]] == gpus[1 + rank], row
        expected_group = {"rank": rank + 1, "world_size": 1 + tp * pp}
        assert row["update_group"] == (
            expected_group if label in ("warm_B", "warm_A_restored") else None
        )
        start, end = stage * (28 // pp), (stage + 1) * (28 // pp)
        assert (row["start_layer"], row["end_layer"]) == (start, end)
        selected_layers = {start, end - 1}
        keys = {f"model.layers.{layer}.self_attn.{proj}.weight"
                for layer in selected_layers for proj in ("qkv_proj", "o_proj")}
        if stage == 0:
            keys.add("model.embed_tokens.weight")
        if stage == pp - 1:
            keys.update(("lm_head.weight", "model.norm.weight"))
        assert set(row["selected"]) == keys, row
        assert row["tie_word_embeddings"] is False
        assert row["word_embeddings_untied_by_checkpoint"] is True
        assert row["base_py_sha256"] == provenance["base_py_sha256"]
        assert row["probe_sha256"] == provenance["harness_sha256"]["tied_weight_probe.py"]
        if pp == 1:
            embed = row["selected"]["model.embed_tokens.weight"]
            head = row["selected"]["lm_head.weight"]
            assert row["input_output_same_parameter"] is True
            assert embed["data_ptr"] == head["data_ptr"]
            assert embed["sha256"] == head["sha256"]
        else:
            assert row["input_output_same_parameter"] is None
        if label == "cold_B":
            sentinels = json.loads((output / "B-sentinels.json").read_text())
            for layer in selected_layers & {0, 27}:
                expected = sentinels[str(layer)][str(row["tp_rank"])]
                for proj, field in (("qkv_proj", "q"), ("o_proj", "o")):
                    actual = row["selected"][f"model.layers.{layer}.self_attn.{proj}.weight"]["first"]
                    assert actual == expected[field], (rank, layer, proj, actual, expected)
            expected_embed = json.loads((output / "B-embedding-sentinels.json").read_text())
            for name in ("model.embed_tokens.weight", "lm_head.weight"):
                if name in row["selected"]:
                    assert row["selected"][name]["first"] == expected_embed[str(row["tp_rank"])]
    if label.startswith("cold_") and pp == 2:
        for rank in range(tp):
            assert rows[rank]["selected"]["model.embed_tokens.weight"]["sha256"] == rows[rank + tp]["selected"]["lm_head.weight"]["sha256"]
    prompts = json.loads((output / "prompts.json").read_text())
    responses = []
    for ids in prompts:
        response = http(url, "/v1/completions", {
            "model": "m", "prompt": ids, "max_tokens": 8, "temperature": 0,
            "ignore_eos": True, "logprobs": 5, "prompt_logprobs": 1, "echo": True,
            "return_token_ids": True, "seed": 0,
        })
        logprobs = response["choices"][0]["logprobs"]["token_logprobs"]
        assert len(logprobs) >= len(ids) and all(math.isfinite(x) for x in logprobs[1:])
        responses.append(response)
    requests.get(url + "/health", timeout=10).raise_for_status()
    result = {"census": rows, "responses": responses,
              "weight_info": http(url, "/weight_info", method="GET")}
    write(output / f"{label}.json", result)
    print(f"OBSERVED {label}: {len(rows)} ranks, {len(prompts)} prompts", flush=True)
    return result


def compare(actual, expected):
    hashes = all(
        a["selected"].keys() == b["selected"].keys() and all(
            a["selected"][key]["sha256"] == b["selected"][key]["sha256"]
            for key in a["selected"]
        ) for a, b in zip(actual["census"], expected["census"], strict=True)
    )
    max_prompt_delta, max_all_delta, tokens_equal = 0.0, 0.0, True
    for a, b in zip(actual["responses"], expected["responses"], strict=True):
        ca, cb = a["choices"][0], b["choices"][0]
        tokens_equal &= ca["token_ids"] == cb["token_ids"]
        count = a["usage"]["prompt_tokens"]
        assert count == b["usage"]["prompt_tokens"]
        la, lb = ca["logprobs"]["token_logprobs"], cb["logprobs"]["token_logprobs"]
        max_prompt_delta = max(max_prompt_delta, max(
            abs(x - y) for x, y in zip(la[1:count], lb[1:count], strict=True)
        ))
        max_all_delta = max(max_all_delta, max(
            abs(x - y) for x, y in zip(la[1:], lb[1:], strict=True)
        ))
    return {"parameter_hashes_match": hashes,
            "max_prompt_logprob_delta": max_prompt_delta,
            "max_all_token_logprob_delta": max_all_delta,
            "continuation_token_ids_match": tokens_equal,
            "all_captured_logprobs_exact": all(
                a["choices"][0]["logprobs"] == b["choices"][0]["logprobs"]
                for a, b in zip(actual["responses"], expected["responses"], strict=True)
            ),
            "prompt_token_ids_exact": all(
                a["choices"][0]["prompt_token_ids"] == b["choices"][0]["prompt_token_ids"]
                for a, b in zip(actual["responses"], expected["responses"], strict=True)
            )}


def run_arm(output, arm):
    import torch
    import vllm

    _, tp, pp, packed = topology()
    verify_source()
    assert Path(vllm.__file__).resolve().is_relative_to(SOURCE)
    if arm == "prepare":
        prepare(output)
        return
    if arm.startswith("cold_"):
        model = ASSET if arm == "cold_A" else output / "B-checkpoint"
        with server(output, arm, model) as url:
            observe(url, output, arm)
        assert not torch.cuda.is_initialized()
        return
    from transformers import AutoModelForCausalLM
    from vllm.distributed.weight_transfer import (
        HTTPVLLMWeightSyncClient, ModuleSource, WeightTransferTrainerFactory,
    )
    from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

    class VersionedClient(HTTPVLLMWeightSyncClient):
        version = "default"

        def finish_weight_update(self, weight_version=None):
            super().finish_weight_update(self.version)

    torch.cuda.set_device(0)
    model = AutoModelForCausalLM.from_pretrained(
        str(ASSET), dtype=torch.bfloat16, local_files_only=True
    ).eval().to("cuda:0")
    assert model.lm_head.weight is model.model.embed_tokens.weight
    originals = {name: param.detach().clone() for name, param in model.named_parameters()
                 if name in ("model.norm.weight", "model.embed_tokens.weight") or any(
                     name == f"model.layers.{layer}.self_attn.{proj}.weight"
                     for layer in (0, 27) for proj in ("q_proj", "o_proj"))}
    with server(output, arm, ASSET) as url:
        initial = observe(url, output, "warm_A_initial")
        world = http(url, "/get_world_size", method="GET")["world_size"]
        assert world == tp * pp, world
        source = ModuleSource(model)
        metadata = source.metadata
        write(output / "trainer-source-metadata.json", [
            {"name": item.name, "dtype": str(item.dtype), "shape": item.shape}
            for item in metadata
        ])
        names = {item.name for item in metadata}
        assert "model.embed_tokens.weight" in names
        assert ("lm_head.weight" not in names) if EXPECT_BUG else ("lm_head.weight" in names)
        client = VersionedClient(url, timeout=120)
        engine = WeightTransferTrainerFactory.trainer_init(
            init_info=NCCLTrainerInitInfo(
                rank=0, master_address="127.0.0.1", master_port=free_port(),
                world_size=1 + world, packed=packed,
            ), client=client, source=source,
        )
        try:
            for version in ("B", "A_restored"):
                if version == "B":
                    modified_b(model)
                else:
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if name in originals:
                                param.copy_(originals[name])
                http(url, "/pause?mode=wait&clear_cache=true")
                client.version = version
                started = time.monotonic()
                engine.send_weights()
                write(output / f"transfer-{version}.json", {
                    "seconds": time.monotonic() - started, "packed": packed,
                    "version": version,
                })
                assert http(url, "/weight_info", method="GET")["weight_version"] == version
                http(url, "/resume")
                actual = observe(url, output, "warm_" + version)
                stable = all(
                    row["pid"] == before["pid"] and all(
                        value["data_ptr"] == before["selected"][name]["data_ptr"]
                        for name, value in row["selected"].items()
                    ) for row, before in zip(actual["census"], initial["census"], strict=True)
                )
                write(output / f"storage-{version}.json", {"preserved": stable})
                assert stable
        finally:
            engine.shutdown()


def source_patch():
    return subprocess.check_output(["git", "-C", str(SOURCE), "diff", BASE, "--binary"])


def main():
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("prepare", "cold_A", "cold_B", "warm"))
    args = parser.parse_args()
    gpus, tp, pp, packed = topology()
    if args.arm:
        run_arm(args.output, args.arm)
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "source-manifest.json", verify_source(full=True))
    write(output / "model-verification.json", verify_model())
    write(output / "runtime-provenance.json", runtime_provenance())
    for name in HARNESSES:
        shutil.copy2(Path(__file__).with_name(name), output / name)
    head = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    subprocess.run(["git", "-C", str(SOURCE), "merge-base", "--is-ancestor", BASE, head], check=True)
    patch = source_patch()
    assert (not patch and head == BASE) if EXPECT_BUG else bool(patch)
    (output / "source.patch").write_bytes(patch)
    base_file = SOURCE / "vllm/distributed/weight_transfer/base.py"
    shutil.copy2(base_file, output / "tested-base.py")
    provenance = {
        "argv": sys.argv, "source": str(SOURCE), "base": BASE, "head": head,
        "native_wheel_commit": "fe77d0906be0c38c5e2c575cd61b545a2fa687e4",
        "python": sys.executable, "driver_pid": os.getpid(),
        "trainer_physical_gpu": gpus[0], "inference_physical_gpus": gpus[1:],
        "tensor_parallel_size": tp, "pipeline_parallel_size": pp, "packed": packed,
        "harness_sha256": {name: sha_file(output / name) for name in HARNESSES},
        "source_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "base_py_sha256": sha_file(base_file),
        "git_status": subprocess.check_output(
            ["git", "-C", str(SOURCE), "status", "--porcelain"], text=True
        ),
        "reuse_prepared": None,
        "expect_missing_tied_name_bug": EXPECT_BUG,
    }
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
        "PYTHONPATH": str(SOURCE) + os.pathsep + str(output),
        "VLLM_BATCH_INVARIANT": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "VLLM_NO_USAGE_STATS": "1", "PYTHONHASHSEED": "0",
        "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
    })
    env.pop("VLLM_TARGET_DEVICE", None)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    provenance["child_env"] = {
        key: value for key, value in env.items()
        if key.startswith(("NCCL_", "TIED_SYNC_")) or key in (
            "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "PATH", "NO_PROXY", "no_proxy",
            "VLLM_BATCH_INVARIANT", "VLLM_WORKER_MULTIPROC_METHOD", "PYTHONHASHSEED",
            "HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "VLLM_NO_USAGE_STATS",
        )
    }
    write(output / "provenance.json", provenance)
    arms = ["prepare", "cold_A", "cold_B", "warm"]
    for arm in arms:
        started, code, error = time.monotonic(), None, None
        command = [sys.executable, str(output / HARNESSES[0]), "--output", str(output), "--arm", arm]
        write(output / f"{arm}-driver-command.json", command)
        with (output / f"{arm}-driver.log").open("w") as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            print(f"ARM {arm} pid={proc.pid}", flush=True)
            try:
                code = proc.wait(timeout=1800)
            except BaseException as exc:
                error = repr(exc)
                raise
            finally:
                for live in output.glob("*-live-server.json"):
                    terminate_group(json.loads(live.read_text())["pid"])
                if proc.poll() is None:
                    terminate_group(proc.pid)
                    proc.wait(timeout=15)
                write(output / f"{arm}-exit.json", {
                    "returncode": proc.returncode, "seconds": time.monotonic() - started,
                    "error": error, "pid": proc.pid, "pgid": proc.pid,
                })
            if code:
                raise RuntimeError(f"{arm} failed with {code}; see logs")
        if arm == "prepare":
            record_prepared(output)
        if arm == "cold_B":
            negative = compare(json.loads((output / "cold_B.json").read_text()),
                               json.loads((output / "cold_A.json").read_text()))
            write(output / "negative-control-before-transfer.json", negative)
            assert not negative["parameter_hashes_match"], negative
            assert negative["max_prompt_logprob_delta"] > 0.02, negative
    load = lambda name: json.loads((output / f"{name}.json").read_text())
    checks = {
        "initial_A": compare(load("warm_A_initial"), load("cold_A")),
        "updated_B": compare(load("warm_B"), load("cold_B")),
        "restored_A": compare(load("warm_A_restored"), load("cold_A")),
        "negative_B_vs_A": compare(load("cold_B"), load("cold_A")),
    }
    write(output / "analysis.json", checks)
    assert hashlib.sha256(source_patch()).hexdigest() == provenance["source_patch_sha256"]
    assert sha_file(base_file) == provenance["base_py_sha256"]
    write(output / "source-after.json", {"patch_unchanged": True, "base_py_unchanged": True})
    for name in (("initial_A", "restored_A") if EXPECT_BUG else ("initial_A", "updated_B", "restored_A")):
        assert checks[name]["parameter_hashes_match"], (name, checks[name])
        assert checks[name]["max_prompt_logprob_delta"] == 0.0, (name, checks[name])
        assert checks[name]["max_all_token_logprob_delta"] == 0.0, (name, checks[name])
        assert checks[name]["continuation_token_ids_match"], (name, checks[name])
        assert checks[name]["all_captured_logprobs_exact"], (name, checks[name])
        assert checks[name]["prompt_token_ids_exact"], (name, checks[name])
    if EXPECT_BUG:
        assert tp == pp == 2 and packed
        assert not checks["updated_B"]["parameter_hashes_match"]
        assert checks["updated_B"]["max_prompt_logprob_delta"] > 0.02
        cold_b, cold_a, warm_b = load("cold_B"), load("cold_A"), load("warm_B")
        changed = []
        for actual, target, original in zip(warm_b["census"], cold_b["census"], cold_a["census"], strict=True):
            for name, value in actual["selected"].items():
                if value["sha256"] != target["selected"][name]["sha256"]:
                    changed.append({"rank": actual["rank"], "name": name})
                    assert value["sha256"] == original["selected"][name]["sha256"]
        assert changed == [{"rank": rank, "name": "lm_head.weight"} for rank in (2, 3)], changed
        write(output / "expected-baseline-failure.json", {"status": "bug_reproduced", "changed_vs_B": changed, "comparison": checks["updated_B"]})
    assert not checks["negative_B_vs_A"]["parameter_hashes_match"]
    assert checks["negative_B_vs_A"]["max_prompt_logprob_delta"] > 0.02
    verify_source()
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
