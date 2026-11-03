# vLLM #56331 — current-head A100 validation

The six fresh-process Qwen cache-isolation and restart checks pass on the current
published PR head. No repository source changes were needed for this validation.

- Head: `ba29c63bbaa130128268b9b65c0a0e4cfd85a38f`
- Tree: `852ce8296e1c0399bb970a383a0c1fb299443203`
- Hardware: one NVIDIA A100-SXM4-40GB, physical GPU 7 of the supplied eight-GPU host.
- Driver: 580.178.04; Ubuntu 22.04; Python 3.12.16; PyTorch 2.14.0+cu130;
  Transformers 5.20.2; FlashInfer 0.7.2.post1.
- Model: `Qwen/Qwen3-0.6B`, revision
  `c1899de289a04d12100db370d81485cdf75e47ca`. All seven required runtime files
  were checked against their pinned sizes and Git/LFS content hashes.

## Filesystem controls and restarts

BF16, eager FA2, batch-invariant execution, fixed prompt token IDs, 48 greedy
continuation tokens and top-20 logprobs. A uses linear RoPE factor 1, B factor 2.
Each row runs in a new process; the four shared runs use the same filesystem root.

| Run | Recorded FS read bytes | Recorded FS write bytes | Cached tokens |
| --- | ---: | ---: | ---: |
| A clean control | null | 133,955,584 | 0 |
| B clean control | null | 133,955,584 | 0 |
| A shared cold | null | 133,955,584 | 0 |
| A shared restart | 133,955,584 | null | 1,168 |
| B shared isolation | null | 133,955,584 | 0 |
| B shared restart | 133,955,584 | null | 1,168 |

All four shared runs match their corresponding clean control exactly for prompt
IDs, generated IDs, all captured logprob entries and cumulative logprob. The two
configuration namespaces are distinct; pre-existing cache payloads retain their
recorded sizes and hashes. Both restarts have actual positive FS-read and
cached-token measurements.

`null` preserves the registered lazy counter's absence of an increment sample.
It is not a fabricated zero measurement. The runner uses the real
`llm.sleep(level=0, mode="wait")` I/O barrier, drains final scheduler metrics and
shuts down EngineCore before examining cache files. No fixed wait or dummy
inference substitutes for completion.

## CPU regressions

All nine existing connector/offloading suites pass on this same clean head:
**492 passed, no skips, one pytest import-rewrite warning, 172.56 seconds**.
CUDA remains uninitialized before and after the CPU run. The old macOS
`/dev/shm` skip runs successfully on this Linux server.

The suite list, node-level outcomes, exact command and source identity are in
`evidence-pr56331/evidence/cpu-pr56331-receipt.json`; raw stdout is in
`evidence-pr56331/logs/cpu-pr56331.log`.

## Reproduction and provenance

```bash
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
  VLLM_PRECOMPILED_WHEEL_COMMIT=ed1a7307950ab5e6593d25123579ed7ca998b350 \
  uv pip install --editable . --torch-backend=cu130
CUDA_HOME=/usr/local/cuda-13.0 \
  .venv/bin/python /path/to/validate_fs_rope.py \
  --source "$PWD" --model /path/to/pinned-Qwen-snapshot \
  --model-manifest /path/to/model-revision-manifest.json \
  --gpu 7 --output /path/to/fresh-evidence-directory
```

The runner verifies HEAD/tree, clean tracked source, all tracked-file hashes,
frontend and worker import paths, package versions and loaded native-library
hashes. Official precompiled native extensions come from the merged upstream
base `ed1a7307950ab5e6593d25123579ed7ca998b350`. This is Python-source validation
with recorded official binaries, not an exact-source native rebuild or a
performance benchmark. It covers TP1 filesystem behavior; actual P2P/RDMA and
object-store deployments are outside this run.

The same-host worker-provenance callback explicitly uses
`VLLM_ALLOW_INSECURE_SERIALIZATION=1`, retained in `controlled-environment.json`.
The new server initially lacked `nvcc`; that first startup failed before model
evaluation. Its raw failure is retained. After installing CUDA 13.0 compilation
tools, the complete six-process run above passed without source patches or
relaxed numerical checks.

## Published files

- [Complete original logs, results, manifests, scripts and runtime records](raw-evidence.tar.xz?raw=1)
- [Archive/checksum receipt](archive-receipt.json)
- [Directly readable result files](data)
- [Frozen reproduction scripts](scripts)

Published archive: `raw-evidence.tar.xz`, 445,116 bytes.
SHA256: `fd7eae9b12a860416c39706570ad439440c17c13acb875b794650f3d1dae043b`. Its uncompressed tar bytes are
identical to the verified local gzip archive; the receipt records both formats
and the common tar checksum. Large model/KV payloads are excluded, with their
observed hashes and deterministic reproduction inputs retained.

```bash
mkdir -p evidence-pr56331
tar -xJf raw-evidence.tar.xz -C evidence-pr56331
```

The extracted directory preserves all original filenames and recorded host paths.
Configure equivalent checkout/model paths when reproducing, and retain the pinned
source commit/tree and model revision.
