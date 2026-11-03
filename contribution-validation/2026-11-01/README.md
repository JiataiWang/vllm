# vLLM current-head A100 validation

Reports and complete raw evidence for two existing vLLM contributions, collected
on eight A100-SXM4-40GB GPUs on November 1–2, 2026. Each report pins the tested
Python source commit/tree, model snapshot and native-library provenance.

| PR | Tested source head | Result |
| --- | --- | --- |
| [56331](https://github.com/vllm-project/vllm/pull/56331) | `ba29c63bbaa130128268b9b65c0a0e4cfd85a38f` | 492 CPU tests; six filesystem isolation/restart processes pass |
| [56689](https://github.com/vllm-project/vllm/pull/56689) | `2f25de1c9d3a31160b7f2e789428f5129dd006dd` | Full GPU test module: 117 passed, 23 optional Modelexpress skips; TP2/PP2 packed and TP2/PP1 unpacked A→B→A match cold controls exactly |

- [PR 56331 report, reproduction and downloadable evidence](pr56331/README.md)
- [PR 56689 report, before-fix reproduction and downloadable evidence](pr56689/README.md)
- [Per-file SHA256 and Git blob manifest](manifest.json)

The unmodified before-fix base reproduces both stale last-stage output-head
shards and nonzero logprob error. Frozen scripts, raw numerical records, setup
failures, warnings and package/native provenance are included in the small XZ
archives. Their uncompressed tar bytes match the locally verified original
archives. The independent model-result audit passes remotely and after local
archive extraction.

This branch stores contribution-validation artifacts. Source PR branches retain
their existing tested heads. Hosted CI eligibility and maintainer review remain
separate from these local validation records.
