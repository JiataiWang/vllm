# vLLM #56689 — current-head A100 validation

The complete ordinary weight-transfer test module passes with GPU integration
enabled. Real NCCL A→B→A updates also match cold controls on both requested
topologies, and the unmodified upstream base reproduces the stale output head.
No repository source changes were needed in this validation session.

- PR head: `2f25de1c9d3a31160b7f2e789428f5129dd006dd`
- PR tree: `eb833d3579382cb99305773e2876cc211dabeb02`
- Unmodified before-fix base: `fe77d0906be0c38c5e2c575cd61b545a2fa687e4`
- Hardware: eight NVIDIA A100-SXM4-40GB GPUs with NV12 interconnects.
- Ubuntu 22.04; driver 580.178.04; Python 3.12.16; PyTorch 2.14.0+cu130;
  Transformers 5.20.2; Ray 2.56.1; FlashInfer 0.7.2.post1.

## Ordinary pytest and the before-fix regression

- Full unchanged `tests/distributed/test_weight_transfer.py`, GPUs enabled:
  **117 passed, 23 skipped in 105.91 seconds**. All 23 skips are optional
  Modelexpress tests because `modelexpress_rl` is not installed. Both NCCL
  process-transfer variants and both Ray/HTTP IPC variants pass.
- Full unchanged module, GPUs hidden: **97 passed, 43 skipped in 43.62 seconds**.
- The unchanged PR test file, collected normally against production imports
  from the unmodified base: **4 failed, 1 passed in 3.45 seconds** for
  `TestModuleSource`. All four failures are the missing tied-name regression.
  These same cases pass in the current-head whole-module runs.

These are normal pytest collections with Ray installed. No definitions are
extracted with AST, no Ray stub is used, and no production methods are patched.
The GPU run uses the same `VLLM_ALLOW_INSECURE_SERIALIZATION=1` opt-in as the
repository's distributed Buildkite selection, on this trusted single host.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 RAY_TMPDIR=/tmp/vlray-repro \
  PYTHONPATH=. .venv/bin/python -m pytest -q -rs \
  tests/distributed/test_weight_transfer.py
```

The original failed GPU-module attempt is retained: its four Ray integration
cases failed before Ray startup because the task's long temporary-directory
name exceeded Linux's Unix-socket path limit. A short task-specific `/tmp`
directory resolves that environment error without changing tests or assertions.
The passing run retains pytest-asyncio's configuration deprecation warning and
a CUDA-IPC producer-shutdown warning; the final process/GPU audit finds no
remaining validation jobs or GPU allocations.

## Real model updates and cold controls

Model: `Qwen/Qwen3-0.6B`, revision
`c1899de289a04d12100db370d81485cdf75e47ca`, with pinned runtime files verified.
BF16, eager FA2, batch-invariant execution, two fixed tokenized prompts and eight
greedy continuation tokens per prompt. The native HTTP/NCCL control plane pauses
with `mode=wait&clear_cache=true`, sends weights, commits the weight version,
resumes and observes outputs.

Each topology has fresh cold A, cold B and warm A engines. B deterministically
scales the tied embedding, changes Q/O sentinels on layers 0/27 and halves the
final norm. Its checkpoint contains an explicit equal output head. It is a
controlled correctness input, not a trained-model quality or speed benchmark.

| Source / topology | B update vs fresh B | Maximum prompt-logprob difference | Maximum returned token-logprob difference |
| --- | --- | ---: | ---: |
| Before fix, TP2/PP2 packed | Both last-stage heads remain A | 2.0645949840545654 | 2.0912771224975586 |
| Current PR, TP2/PP2 packed | Every inspected full shard matches | 0.0 | 0.0 |
| Current PR, TP2/PP1 unpacked | Every inspected full shard matches | 0.0 | 0.0 |

Initial A and restored A match their cold control exactly in all three runs.
Both fixed topologies also match cold B exactly for generated IDs and every
captured logprob entry. The before-fix run reports weight version B while the
two output-head shards still hash to A; the other inspected shards match B.
Greedy token IDs happen to agree in this reproducer, so the parameter hashes
and probabilities are necessary to expose the defect.

The read-only worker census hashes complete selected shards: the first/last
attention projections in each pipeline partition, input embeddings, output
heads and final norms. There are 22 selected full shards in TP2/PP2 and 14 in
TP2/PP1. Worker PIDs and all inspected storage pointers remain unchanged across
both updates; the PP1 embedding/output-head alias remains shared.

An independent raw-result audit verifies all recorded observations, frozen
scripts, source identities, metadata versions, full-shard hashes, process
receipts, all captured probabilities and stable storage. The B checkpoint is
byte-identical across all three runs:
`511e702cec518602c5054a310cfa5cdff1109a0f0152b77a33e1d6097f9dc049`.

## Reproduction and provenance

Use a clean checkout of the PR head and the frozen scripts from each result
directory. Both fixed runs use an independent trainer GPU. TP2/PP2 takes four
inference GPUs plus one trainer; TP2/PP1 takes two plus one.

```bash
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
  VLLM_PRECOMPILED_WHEEL_COMMIT=fe77d0906be0c38c5e2c575cd61b545a2fa687e4 \
  uv pip install --editable . --torch-backend=cu130

CUDA_VISIBLE_DEVICES=0,1,2,3,4 TIED_SYNC_GPUS=0,1,2,3,4 \
  TIED_SYNC_TP=2 TIED_SYNC_PP=2 TIED_SYNC_PACKED=1 \
  VLLM_BATCH_INVARIANT=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_HOME=/usr/local/cuda-13.0 \
  PYTHONPATH="$PWD:/path/to/frozen-scripts" \
  .venv/bin/python /path/to/frozen-scripts/launch_tied_weight_sync_validation.py \
  --output /path/to/new-output
```

For TP2/PP1, use three listed GPUs, `TIED_SYNC_PP=1` and `TIED_SYNC_PACKED=0`.
The scripts' task-root/model paths must point to the equivalent local pinned
snapshot and checkout. Each output directory contains the exact scripts used,
commands, environment, full tracked-source manifest, model checksums and
frontend/worker package/source/native-library provenance.

Official precompiled vLLM extensions come from the common upstream base
`fe77d0906be0c38c5e2c575cd61b545a2fa687e4`. CUDA 13.0 tools compile dependency
JIT kernels as needed. This is exact Python-head validation with recorded
official native binaries, not a rebuild of all native sources at the PR SHA.
No performance, model-quality, multi-node or untested-backend claim is made.

The first model attempt hit its 240-second startup deadline during fresh-server
initialization. Its logs are retained. The successful complete run allows
900 seconds for readiness and 1,800 seconds per arm, retains exact-zero output
checks and records the compiler activity and thread settings. No numerical
tolerance, test skip or test-source change was used to obtain a pass.

The completed archive omits large model payloads but retains their checksums,
checkpoint configs, all raw output/probability/shard records, scripts, native
hashes, node-level pytest receipts and Ray runtime logs. The immutable A snapshot
and the frozen deterministic mutation reproduce B.

## Published files

- [Complete original logs, results, manifests, scripts and runtime records](raw-evidence.tar.xz?raw=1)
- [Archive/checksum receipt](archive-receipt.json)
- [Directly readable result files](data)
- [Frozen reproduction scripts](scripts)

Published archive: `raw-evidence.tar.xz`, 677,848 bytes.
SHA256: `59dfd50d604b404a2cf553f2172f21cc6ee4a7f65ccfbec5738c7f22151ed06f`. Its uncompressed tar bytes are
identical to the verified local gzip archive; the receipt records both formats
and the common tar checksum. Large model/KV payloads are excluded, with their
observed hashes and deterministic reproduction inputs retained.

```bash
mkdir -p evidence-pr56689
tar -xJf raw-evidence.tar.xz -C evidence-pr56689
```

The extracted directory preserves all original filenames and recorded host paths.
Configure equivalent checkout/model paths when reproducing, and retain the pinned
source commit/tree and model revision.

The full model-result audit can also be checked offline without a GPU:

```bash
.venv/bin/python scripts/audit_weight_results.py --root evidence-pr56689
```

The local and server audits produced identical receipts before publication.
