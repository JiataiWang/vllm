# vLLM PR validation follow-up, November 14, 2026

PR #56331 is synchronized with main and pushed at `46849702cef6c3d94df6a71569f2675652129945`,
tree `8c337984a2f2001a51bd1d88b3618b57493aae09`. The merged upstream is
`a58360ca30e84c70a89a676a38b3779a9fe9f110`. Both PRs merge without conflicts
against subsequently advanced main `9bc16ae6a34b1e0fe260e8d12b98568478bf38c3`.

## Changes and regression coverage

The merge keeps upstream cache-access policy and new per-layer cache-group APIs.
The startup model-configuration hash function is unchanged from the prior
A100-validated head. Test fixtures use real `KVCacheLayerConfig` objects and
the shared cache-group helpers; an obsolete registry patch is removed.
All existing assertions are retained. No new skip, retry, xfail or tolerance
change was introduced.

| Validation | Result |
| --- | --- |
| Nine complete suites, Linux CPU | 541 passed, 11 existing skips, 114.47 s |
| FS suite with existing macOS native I/O extension | 48 passed, no skips, 11.82 s |
| Engine snapshot/auto-fit regressions | 8 passed |
| Applicable hooks on all 16 PR paths | 18 passed, including Ruff 0.17.2, mypy and test tethering |
| Canonical MkDocs, API generation excluded | Passed, 47.10 s |
| Standalone strict feature docs | Passed |
| Rendered in-page links | 240 valid anchors in each page |

All 11 Linux skips report `fs_io_C extension not built`; XML node IDs verify
that each skipped case passes in the macOS FS run. This is complementary
platform coverage, not a claim that the Linux run had no skips. The Linux run
covers 552 distinct cases. Every one of 9,326 tracked source blobs is checked
against the staged Git tree before and after pytest. CUDA stays uninitialized.
The source checkout was mounted read-only.

Linux: Python 3.12.15, CPU Torch 2.13.0+cpu, Transformers 5.15.1.
macOS: Python 3.12.7, CPU Torch 2.13.0. The existing FS extension is recorded
by path and SHA-256 in `coverage-receipt.json`; it was not rebuilt in this run.
The Linux environment is based on the cached official CPU image
`vllm/vllm-openai-cpu:v0.32.0-arm64`, with isolated test dependencies.

The first macOS merge run has 9 failed, 542 passed and 1 existing skip: eight
failures expose old fixture interfaces, and one uses Linux `/dev/shm`.
The intermediate engine-only run has 1 failed and 7 passed because its obsolete
registry patch remains. Both raw failures are retained. After adapting the
fixtures, the eight engine cases pass; Linux runs the `/dev/shm` case normally.

## Raw records and reproduction

Download [the raw validation archive](pr56331-validation.tar.xz) and verify it
using [SHA256SUMS](SHA256SUMS). It contains the source manifest, unchanged runner,
actual environment setup log, raw pytest logs and XML, receipts, hooks, docs,
and final patches. The source manifest binds these records to the tree above.

From that exact checkout in the recorded Linux environment, the ordinary pytest
selection equivalent to the runner is:

```bash
VLLM_TARGET_DEVICE=cpu PYTHONPATH=. .venv/bin/python -m pytest -q -rs \
  tests/v1/kv_connector/unit/offloading_connector/test_config.py \
  tests/v1/kv_offload/test_file_mapper.py \
  tests/v1/kv_offload/tiering/test_fs_tier.py \
  tests/v1/kv_offload/tiering/test_obj_tier.py \
  tests/v1/kv_offload/test_factory.py \
  tests/v1/kv_connector/unit/test_handshake_pp_aggregation.py \
  tests/v1/kv_offload/tiering/p2p/test_sessions.py \
  tests/v1/kv_offload/cpu/test_capacity.py \
  tests/v1/kv_offload/tiering/test_tiering_offloading.py
```

The exact executed runner is `run-linux-tests.py` in the archive: it verifies
`/reports/source-manifest.json`, imports `/workspace/vllm`, then calls pytest
with the paths above, `--junitxml=/reports/tests-linux.xml` and
`-o cache_dir=/tmp/pr56331-pytest-cache`. The Docker command mounts this checkout
read-only at `/workspace` and the extracted `pr56331` records at `/reports`,
sets `VLLM_TARGET_DEVICE=cpu`, `PYTHONPATH=/workspace`,
`PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, uses
`--cpus 3 --memory 3g --shm-size 1g`, and runs:

```bash
mkdir -p /task
ln -s /opt/venv /task/.venv
uv pip install --python /task/.venv/bin/python -r /reports/linux-test-dependencies.txt
/task/.venv/bin/python /reports/run-linux-tests.py
```

The macOS supplement executes the unmodified FS module:

```bash
VLLM_TARGET_DEVICE=cpu PYTHONPATH=. .venv/bin/python -m pytest -q -rs \
  tests/v1/kv_offload/tiering/test_fs_tier.py
```

## GPU provenance and limits

No GPU tests were rerun for the new #56331 merge. The six A100 Qwen3-0.6B
cache-isolation/restart controls remain attached to `ba29c63bbaa130128268b9b65c0a0e4cfd85a38f`,
not the new commit. [Prior immutable GPU evidence](https://github.com/JiataiWang/vllm/blob/3fa2e165ebc3be80cb072ff870dd45a2b3015d44/contribution-validation/2026-11-01/pr56331/README.md)
is preserved. No fresh accuracy, performance, P2P/RDMA, object-store or native
source build result is claimed by this merge validation.

PR #56689 remains at `2f25de1c9d3a31160b7f2e789428f5129dd006dd`; its source and
[complete-module A100 and native NCCL model evidence](https://github.com/JiataiWang/vllm/blob/3fa2e165ebc3be80cb072ff870dd45a2b3015d44/contribution-validation/2026-11-01/pr56689/README.md)
are unchanged. It has no conflict or new actionable review thread.

## Hosted CI

The account still has two merged vLLM PRs, below the pre-run policy threshold
of four, without a qualifying maintainer label. Both prior hosted pre-run jobs
failed this eligibility check. ReadtheDocs logs confirm that
`docs/pre_run_check.sh` exits on that prerequisite before MkDocs starts.
This is separate from the local docs results above. The original gate excerpts
are included in the archive. Maintainer review and CI eligibility remain.
No qualifying label, review ping, self-approval or CI-policy change was made.
