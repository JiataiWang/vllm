"""Run the unchanged complete weight-transfer module with GPU integration enabled."""

import json
import os
from pathlib import Path
import subprocess
import sys

# This is the same opt-in used by the repository's weight-transfer CI step.
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'

import torch
import vllm
import pytest

root = Path('/home/shadeform/vllm-pr-validation-nov1')
source = root / 'pr56689'
os.chdir(source)


def git(*args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


assert git('rev-parse', 'HEAD') == '2f25de1c9d3a31160b7f2e789428f5129dd006dd'
assert git('rev-parse', 'HEAD^{tree}') == 'eb833d3579382cb99305773e2876cc211dabeb02'
assert not git('status', '--porcelain', '--untracked-files=no')
assert Path(vllm.__file__).resolve() == source / 'vllm/__init__.py'
assert torch.cuda.is_available() and torch.cuda.device_count() == 8
rows = []


class Recorder:
    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or report.skipped:
            rows.append({'nodeid': report.nodeid, 'outcome': report.outcome,
                         'duration': report.duration})


receipt = {'head': git('rev-parse', 'HEAD'), 'tree': git('rev-parse', 'HEAD^{tree}'),
           'python': sys.executable, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
           'vllm_import': str(vllm.__file__), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
           'allow_insecure_serialization': os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'],
           'ray_tmpdir': os.environ.get('RAY_TMPDIR'),
           'cuda_initialized_before': torch.cuda.is_initialized(),
           'tests': ['tests/distributed/test_weight_transfer.py']}
print(json.dumps(receipt), flush=True)
code = pytest.main(['-q', '-rs', '--timeout=300',
                    '--junitxml=' + str(root / 'evidence/gpu-weight-module.xml'),
                    *receipt['tests']], plugins=[Recorder()])
receipt.update(pytest_exit_code=int(code), tests_results=rows,
               cuda_initialized_after=torch.cuda.is_initialized())
(root / 'evidence/gpu-weight-module-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
import ray
if ray.is_initialized():
    ray.shutdown()
assert not git('status', '--porcelain', '--untracked-files=no')
raise SystemExit(code)
