"""Six fresh GPU LLM processes; raw evidence is kept even when assertions fail."""

import argparse
from collections.abc import Mapping
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback

REPO = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
HEAD = "ba29c63bbaa130128268b9b65c0a0e4cfd85a38f"
TREE = "852ce8296e1c0399bb970a383a0c1fb299443203"
READ_METRIC = "vllm:kv_offload_tiering_read_bytes"
WRITE_METRIC = "vllm:kv_offload_tiering_write_bytes"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def file_hash(path, algorithm="sha256", git_blob=False):
    digest = hashlib.new(algorithm)
    if git_blob:
        digest.update(f"blob {Path(path).stat().st_size}\0".encode())
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def verify_source(source, full=False):
    source = Path(source).resolve()
    assert git(source, "rev-parse", "HEAD") == HEAD, "Wrong source commit"
    assert git(source, "rev-parse", "HEAD^{tree}") == TREE, "Wrong source tree"
    assert not git(source, "status", "--porcelain", "--untracked-files=no"), (
        "Tracked source modifications are not allowed"
    )
    result = {"path": str(source), "head": HEAD, "tree": TREE}
    if full:
        tracked = (
            subprocess.check_output(["git", "-C", str(source), "ls-files", "-z"])
            .decode()
            .split("\0")
        )
        result["tracked_files"] = {
            name: (
                {"symlink": os.readlink(source / name)}
                if (source / name).is_symlink()
                else {"sha256": file_hash(source / name)}
            )
            for name in tracked
            if name
        }
    return result


def verify_model(model, manifest_path):
    expected = json.loads(Path(manifest_path).read_text())
    assert (expected["repo"], expected["revision"]) == (REPO, REVISION)
    result = {
        "path": str(Path(model).resolve()),
        "repo": REPO,
        "revision": REVISION,
        "expected_manifest_sha256": file_hash(manifest_path),
        "files": {},
    }
    for name in expected["required_runtime_files"]:
        path = Path(model) / name
        info = expected["files"][name]
        assert path.is_file() and path.stat().st_size == info["size"], name
        sha256 = file_hash(path)
        actual = sha256 if info["lfs_sha256"] else file_hash(path, "sha1", True)
        assert actual == (info["lfs_sha256"] or info["git_blob_sha1"]), name
        result["files"][name] = {
            "size": path.stat().st_size,
            "sha256": sha256,
            "resolved_path": str(path.resolve()),
        }
    extra = [
        str(p.relative_to(model))
        for p in Path(model).rglob("*")
        if p.is_file()
        and not any(x.startswith(".") for x in p.relative_to(model).parts)
        and p.suffix in {".json", ".safetensors", ".bin", ".txt", ".py"}
        and str(p.relative_to(model)) not in expected["files"]
    ]
    assert not extra, f"Unexpected runtime model files: {extra}"
    return result


def cache_manifest(root):
    return {
        str(p.relative_to(root)): {
            "size": p.stat().st_size,
            "sha256": file_hash(p),
            "mtime_ns": p.stat().st_mtime_ns,
        }
        for p in sorted(Path(root).rglob("*"))
        if p.is_file() and (p.suffix == ".bin" or p.name == "config.json")
    }


def runtime_provenance(source):
    import torch
    import vllm

    source = Path(source).resolve()
    assert Path(vllm.__file__).resolve() == source / "vllm/__init__.py"
    modules = {}
    native_paths = set()
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if not path or not name.startswith(("vllm", "flash_attn")):
            continue
        path = Path(path).resolve()
        if not path.is_file():
            continue
        if name.startswith("vllm") and path.suffix == ".py":
            assert path.is_relative_to(source / "vllm"), (name, str(path))
        modules[name] = {"path": str(path), "sha256": file_hash(path)}
        if ".so" in path.name:
            native_paths.add(str(path))
    maps = Path("/proc/self/maps")
    if maps.exists():
        for line in maps.read_text().splitlines():
            path = line.split()[-1]
            if (
                path.startswith("/")
                and ".so" in path
                and any(item in path for item in ("vllm", "flash_attn"))
                and Path(path).is_file()
            ):
                native_paths.add(str(Path(path).resolve()))
    packages = {}
    for name in (
        "vllm",
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "flash-attn",
        "triton",
        "numpy",
        "msgspec",
    ):
        try:
            distribution = importlib.metadata.distribution(name)
            packages[name] = {
                "version": distribution.version,
                "location": str(distribution.locate_file("")),
            }
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "pid": os.getpid(),
        "python": sys.version,
        "vllm_version": vllm.__version__,
        "executable": sys.executable,
        "platform": platform.platform(),
        "source": verify_source(source),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "gpu": str(torch.cuda.get_device_properties(0)),
        "modules": modules,
        "native_libraries": {p: file_hash(p) for p in sorted(native_paths)},
    }


def worker_provenance(worker, source, factor):
    result = runtime_provenance(source)
    rope = worker.model_config.hf_text_config.rope_parameters
    assert rope["rope_type"] == "linear" and rope["factor"] == factor, rope
    result["effective_model_config"] = {
        "rope_parameters": rope,
        "dtype": str(worker.model_config.dtype),
        "max_model_len": worker.model_config.max_model_len,
    }
    return result


def prompt(tokenizer):
    facts = "\n".join(
        f"Record {i}: laboratory team {i % 7} catalogued {100 + i} mineral samples."
        for i in range(64)
    )
    encoded = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "Answer questions using only the supplied records.",
            },
            {
                "role": "user",
                "content": facts
                + "\nWhat did laboratory team 0 do? Summarize in two sentences.",
            },
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    assert isinstance(ids, list) and all(isinstance(t, int) for t in ids)
    return ids


def fs_counter(metrics, name):
    from prometheus_client import REGISTRY

    rows = [
        m for m in metrics if m["name"] == name and m["labels"].get("tier") == "1:fs"
    ]
    collectors = [m for m in REGISTRY.collect() if m.name == name]
    assert len(collectors) == 1, f"Missing/ambiguous registered counter: {name}"
    assert all("value" in row for row in rows), rows
    # OffloadPromMetrics creates bounded Counter labels lazily on first event.
    # No sample under a registered collector is an explicit no-increment case,
    # not an injected zero sample. Preserve that distinction in raw evidence.
    return {
        "value": sum(row["value"] for row in rows) if rows else None,
        "sample_present": bool(rows),
        "collector_registered": True,
        "rows": rows,
        "absence_meaning": "registered lazy counter has no increment sample"
        if not rows
        else None,
    }


def run_child(args):
    result_path = Path(args.result)
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "running",
        "factor": args.factor,
        "timing": {},
        "files_before": cache_manifest(cache_root),
    }
    write_json(result_path, state)
    llm = None
    shutdown_done = False
    try:
        started = time.monotonic()
        state["source"] = verify_source(args.source)
        state["model_snapshot"] = verify_model(args.model, args.model_manifest)
        state["timing"]["verify_seconds"] = time.monotonic() - started
        import torch
        import vllm
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.config import KVTransferConfig

        assert Path(vllm.__file__).resolve() == Path(args.source) / "vllm/__init__.py"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == args.gpu
        assert os.environ["VLLM_BATCH_INVARIANT"] == "1"
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        ids = prompt(tokenizer)
        transfer = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "TieringOffloadingSpec",
                "cpu_bytes_to_use": 512 * 1024**2,
                "block_size": 16,
                "secondary_tiers": [
                    {
                        "type": "fs",
                        "root_dir": str(cache_root),
                        "n_read_threads": 2,
                        "n_write_threads": 2,
                    }
                ],
            },
        }
        kwargs = {
            "model": args.model,
            "dtype": "bfloat16",
            "seed": 0,
            "enforce_eager": True,
            "enable_prefix_caching": True,
            "max_model_len": 4096,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 4096,
            "gpu_memory_utilization": 0.10,
            "kv_cache_memory_bytes": 512 * 1024**2,
            "block_size": 16,
            "disable_log_stats": False,
            "attention_config": {"backend": "FLASH_ATTN", "flash_attn_version": 2},
            "hf_overrides": {
                "rope_parameters": {
                    "rope_type": "linear",
                    "factor": args.factor,
                    "rope_theta": 1000000.0,
                }
            },
            "kv_transfer_config": KVTransferConfig(**transfer),
        }
        state.update(
            {
                "launch_kwargs": kwargs,
                "prompt_token_ids": ids,
                "prompt_token_count": len(ids),
            }
        )
        write_json(result_path, state)
        started = time.monotonic()
        llm = LLM(**kwargs)
        state["timing"]["startup_seconds"] = time.monotonic() - started
        state["metrics_before"] = [dataclasses.asdict(x) for x in llm.get_metrics()]
        state["read_before"] = fs_counter(state["metrics_before"], READ_METRIC)
        started = time.monotonic()
        output = llm.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(
                temperature=0.0, max_tokens=48, logprobs=20, seed=0, ignore_eos=True
            ),
            use_tqdm=False,
        )[0]
        state["timing"]["generation_seconds"] = time.monotonic() - started
        choice = output.outputs[0]
        state.update(
            {
                "text": choice.text,
                "token_ids": list(choice.token_ids),
                "logprobs": [
                    {str(k): dataclasses.asdict(v) for k, v in row.items()}
                    for row in choice.logprobs
                ],
                "cumulative_logprob": choice.cumulative_logprob,
                "num_cached_tokens": output.num_cached_tokens,
            }
        )
        write_json(result_path, state)
        started = time.monotonic()
        # Public synchronous idle barrier: no new requests; pending connector
        # cascades keep EngineCore.has_work true until the real FS I/O finishes.
        llm.sleep(level=0, mode="wait")
        # Exact-version internal read/drain API. The pause utility reply follows
        # prior output messages; draining those records final scheduler metrics.
        core = llm.llm_engine.engine_core
        assert type(core).__name__ == "SyncMPClient", type(core).__name__
        batches = 0
        while not core.outputs_queue.empty():
            assert llm.llm_engine.step() == []
            batches += 1
        state["tail_metric_batches_drained"] = batches
        state["timing"]["drain_seconds"] = time.monotonic() - started
        state["metrics_after"] = [dataclasses.asdict(x) for x in llm.get_metrics()]
        state["read_after"] = fs_counter(state["metrics_after"], READ_METRIC)
        state["write_after"] = fs_counter(state["metrics_after"], WRITE_METRIC)
        assert state["read_before"]["value"] in (None, 0), (
            "Fresh engine already read FS"
        )
        state["fs_read_bytes"] = state["read_after"]["value"]
        state["no_fs_read_events_observed"] = state["fs_read_bytes"] in (None, 0)
        state["effective_model_config"] = {
            "hf_config": llm.llm_engine.model_config.hf_config.to_dict(),
            "dtype": str(llm.llm_engine.model_config.dtype),
            "max_model_len": llm.llm_engine.model_config.max_model_len,
        }
        state["frontend_provenance"] = runtime_provenance(args.source)
        state["worker_provenance"] = llm.collective_rpc(
            worker_provenance, args=(args.source, args.factor), timeout=120
        )
        write_json(result_path, state)
        started = time.monotonic()
        core.shutdown(timeout=120)
        shutdown_done = True
        state["timing"]["shutdown_seconds"] = time.monotonic() - started
        state["files_after"] = cache_manifest(cache_root)
        state["run_configs"] = {
            str(p.relative_to(cache_root)): json.loads(p.read_text())
            for p in sorted(cache_root.rglob("config.json"))
        }
        assert len(state["token_ids"]) == len(state["logprobs"]) == 48
        assert all(
            len(row) >= 20 and all(math.isfinite(x["logprob"]) for x in row.values())
            for row in state["logprobs"]
        )
        assert state["worker_provenance"], "No real worker provenance"
        assert any(p.endswith(".bin") for p in state["files_after"]), "No FS KV files"
        assert any(
            (x["value"] or 0) > 0 for x in (state["read_after"], state["write_after"])
        ), "No recorded successful FS read or write events"
        assert all(c.get("model_config_hash") for c in state["run_configs"].values())
        state["status"] = "complete"
    except BaseException:
        state["status"] = "failed"
        state["failure"] = traceback.format_exc()
        raise
    finally:
        if llm is not None and not shutdown_done:
            try:
                llm.llm_engine.engine_core.shutdown(timeout=120)
            except BaseException:
                state["shutdown_failure"] = traceback.format_exc()
        state["files_at_exit"] = cache_manifest(cache_root)
        write_json(result_path, state)


def compare(control, observed):
    rows_equal = control["logprobs"] == observed["logprobs"]
    return {
        "same_prompt_ids": control["prompt_token_ids"] == observed["prompt_token_ids"],
        "same_token_ids": control["token_ids"] == observed["token_ids"],
        "same_all_captured_logprobs": rows_equal,
        "same_cumulative_logprob": control["cumulative_logprob"]
        == observed["cumulative_logprob"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", required=True, help="Local pinned HF snapshot")
    parser.add_argument("--output", required=True, help="Must not exist")
    parser.add_argument("--gpu", default="0", help="One physical GPU index or UUID")
    parser.add_argument(
        "--model-manifest",
        default=str(Path(__file__).with_name("model-revision-manifest.json")),
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--factor", type=float, choices=(1.0, 2.0), help=argparse.SUPPRESS
    )
    parser.add_argument("--cache-root", help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.source = str(Path(args.source).resolve())
    args.model = str(Path(args.model).resolve())
    args.model_manifest = str(Path(args.model_manifest).resolve())
    if args.child:
        run_child(args)
        return
    output = Path(args.output).resolve()
    assert not output.is_relative_to(Path(args.source)), (
        "Evidence must be outside source"
    )
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "status": "running",
        "source_head": HEAD,
        "source_tree": TREE,
        "driver_sha256": file_hash(__file__),
        "runs": {},
        "comparisons": {},
    }
    write_json(output / "summary.json", summary)
    try:
        write_json(
            output / "source-manifest.json", verify_source(args.source, full=True)
        )
        write_json(
            output / "model-verification.json",
            verify_model(args.model, args.model_manifest),
        )
        write_json(
            output / "model-expected-manifest.json",
            json.loads(Path(args.model_manifest).read_text()),
        )
        hardware_command = [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
        hardware = subprocess.run(hardware_command, capture_output=True, text=True)
        write_json(
            output / "gpu-inventory.json",
            {
                "command": hardware_command,
                "returncode": hardware.returncode,
                "stdout": hardware.stdout,
                "stderr": hardware.stderr,
                "selected_cuda_visible_devices": args.gpu,
            },
        )
        assert hardware.returncode == 0, "GPU inventory unavailable"
        env = os.environ.copy()
        for name in list(env):
            if name.startswith("VLLM_"):
                env.pop(name)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": args.gpu,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": args.source,
                "VLLM_BATCH_INVARIANT": "1",
                "VLLM_KV_CACHE_LAYOUT": "LBNHC",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
                # The worker provenance callback is a local callable sent over
                # vLLM's trusted, same-host engine channel via collective_rpc.
                "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
                "VLLM_NO_USAGE_STATS": "1",
                "HF_HUB_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }
        )
        env["PATH"] = (
            str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        )
        write_json(
            output / "controlled-environment.json",
            {
                name: env[name]
                for name in (
                    "CUDA_VISIBLE_DEVICES",
                    "CUDA_DEVICE_ORDER",
                    "PYTHONHASHSEED",
                    "PYTHONPATH",
                    "VLLM_BATCH_INVARIANT",
                    "VLLM_KV_CACHE_LAYOUT",
                    "VLLM_WORKER_MULTIPROC_METHOD",
                    "VLLM_ENABLE_V1_MULTIPROCESSING",
                    "VLLM_ALLOW_INSECURE_SERIALIZATION",
                    "CUBLAS_WORKSPACE_CONFIG",
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                )
            },
        )
        runs = [
            ("A_clean_control", 1.0, "clean-A"),
            ("B_clean_control", 2.0, "clean-B"),
            ("A_shared_cold", 1.0, "shared"),
            ("A_shared_restart", 1.0, "shared"),
            ("B_shared_isolation", 2.0, "shared"),
            ("B_shared_restart", 2.0, "shared"),
        ]
        results = {}
        for name, factor, root in runs:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                "--source",
                args.source,
                "--model",
                args.model,
                "--model-manifest",
                args.model_manifest,
                "--output",
                str(output),
                "--gpu",
                args.gpu,
                "--factor",
                str(factor),
                "--cache-root",
                str(output / root),
                "--result",
                str(output / f"{name}.json"),
            ]
            started = time.monotonic()
            print(f"START {name}: factor={factor}, root={root}", flush=True)
            with (
                (output / f"{name}.stdout.log").open("w") as stdout,
                (output / f"{name}.stderr.log").open("w") as stderr,
            ):
                process = subprocess.Popen(
                    command,
                    cwd=args.source,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                try:
                    code = process.wait(timeout=args.timeout)
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
                finally:
                    receipt = {
                        "command": command,
                        "pid": process.pid,
                        "returncode": process.returncode,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                    write_json(output / f"{name}.process.json", receipt)
                    summary["runs"][name] = receipt
                    write_json(output / "summary.json", summary)
            assert code == 0, f"{name} failed; inspect its raw logs and partial JSON"
            result = results[name] = json.loads((output / f"{name}.json").read_text())
            assert result["status"] == "complete"
            expected_read = name in {"A_shared_restart", "B_shared_restart"}
            if expected_read:
                assert result["read_after"]["sample_present"]
                assert result["fs_read_bytes"] > 0, (name, result["fs_read_bytes"])
                assert result["num_cached_tokens"] > 0, name
            else:
                assert result["no_fs_read_events_observed"], (
                    name,
                    result["read_after"],
                )
                assert (result["write_after"]["value"] or 0) > 0, name
                assert result["num_cached_tokens"] == 0, name
            if name not in {"A_clean_control", "B_clean_control"}:
                control = results[
                    "A_clean_control" if factor == 1 else "B_clean_control"
                ]
                comparison = compare(control, result)
                summary["comparisons"][name] = comparison
                write_json(output / "summary.json", summary)
                assert all(comparison.values()), (name, comparison)
            assert all(
                p in result["files_after"]
                and all(
                    result["files_after"][p][key] == info[key]
                    for key in ("size", "sha256")
                )
                for p, info in result["files_before"].items()
            ), (name, "Existing cache files changed")
            print(f"DONE {name}: FS read bytes={result['fs_read_bytes']}", flush=True)
        assert all(
            result["prompt_token_ids"] == results["A_clean_control"]["prompt_token_ids"]
            for result in results.values()
        ), "All six processes must receive identical tokenized prompts"
        a = results["A_shared_cold"]["run_configs"]
        b = results["B_shared_isolation"]["run_configs"]
        assert len(a) == 1 and len(b) == 2 and set(a) < set(b)
        assert results["A_shared_restart"]["run_configs"] == a
        assert results["B_shared_restart"]["run_configs"] == b
        assert len({config["model_config_hash"] for config in b.values()}) == 2
        assert set(results["A_clean_control"]["run_configs"]) == set(a)
        assert set(results["B_clean_control"]["run_configs"]) == set(b) - set(a)
        verify_source(args.source)
        summary["fs_reads"] = {
            name: value["fs_read_bytes"] for name, value in results.items()
        }
        summary["namespace_configs"] = b
        summary["status"] = "passed"
    except BaseException:
        summary["status"] = "failed"
        summary["failure"] = traceback.format_exc()
        raise
    finally:
        write_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
