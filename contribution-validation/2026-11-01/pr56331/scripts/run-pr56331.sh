#!/bin/bash
set -euo pipefail
task_root=/home/shadeform/vllm-pr-validation-nov1
trap 'result=$?; printf "%s\n" "$result" > "$task_root/logs/gpu-pr56331.exit"' EXIT
cd "$task_root/pr56331"
export PATH="$PWD/.venv/bin:/usr/local/cuda-13.0/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.0
export PYTHONPATH="$PWD"
date -u
.venv/bin/python "$task_root/scripts/validate_fs_rope.py" \
  --source "$PWD" --model "$task_root/models/Qwen3-0.6B-c1899de" \
  --model-manifest "$task_root/scripts/model-revision-manifest.json" \
  --gpu 7 --output "$task_root/evidence/pr56331-fs-attempt2"
date -u
