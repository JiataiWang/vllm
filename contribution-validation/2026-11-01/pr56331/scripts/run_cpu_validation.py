"""Run ordinary pytest against a checked, clean source checkout."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--head', required=True)
parser.add_argument('--receipt', type=Path, required=True)
parser.add_argument('--suite', choices=('offloading', 'aliases'), required=True)
parser.add_argument('--alias-test', type=Path)
parser.add_argument('--expect-alias-failure', action='store_true')
args = parser.parse_args()
source = args.source.resolve()
os.chdir(source)
sys.path.insert(0, str(source))


def git(*values):
    return subprocess.check_output(['git', '-C', str(source), *values], text=True).strip()


assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert git('rev-parse', 'HEAD') == args.head
assert not git('status', '--porcelain', '--untracked-files=no')
import torch
import vllm
import pytest
from vllm.distributed.weight_transfer.base import ModuleSource

assert Path(vllm.__file__).resolve() == source / 'vllm/__init__.py'
assert not torch.cuda.is_initialized()
rows = []


class Recorder:
    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or report.skipped:
            rows.append({'nodeid': report.nodeid, 'outcome': report.outcome,
                         'duration': report.duration})


if args.suite == 'offloading':
    tests = [
        'tests/v1/kv_connector/unit/offloading_connector/test_config.py',
        'tests/v1/kv_offload/test_file_mapper.py',
        'tests/v1/kv_offload/tiering/test_fs_tier.py',
        'tests/v1/kv_offload/tiering/test_obj_tier.py',
        'tests/v1/kv_offload/test_factory.py',
        'tests/v1/kv_connector/unit/test_handshake_pp_aggregation.py',
        'tests/v1/kv_offload/tiering/p2p/test_sessions.py',
        'tests/v1/kv_offload/cpu/test_capacity.py',
        'tests/v1/kv_offload/tiering/test_tiering_offloading.py',
    ]
else:
    assert args.alias_test and args.alias_test.is_file()
    original = source.parent / 'pr56689/tests/distributed/test_weight_transfer.py'
    assert args.alias_test.read_bytes() == original.read_bytes()
    tests = [str(args.alias_test.resolve()) + '::TestModuleSource']
receipt = {'head': args.head, 'tree': git('rev-parse', 'HEAD^{tree}'),
           'source': str(source), 'vllm_import': str(vllm.__file__),
           'python': sys.executable, 'torch': torch.__version__,
           'cuda_before': torch.cuda.is_initialized(), 'tests': tests,
           'argv': sys.argv, 'expected_alias_failure': args.expect_alias_failure,
           'base_py_sha256': hashlib.sha256((source / 'vllm/distributed/weight_transfer/base.py').read_bytes()).hexdigest()}
if args.alias_test:
    receipt['test_sha256'] = hashlib.sha256(args.alias_test.read_bytes()).hexdigest()
print(json.dumps(receipt), flush=True)
code = pytest.main(['-q', '-rs', '--junitxml=' + str(args.receipt.with_suffix('.xml')), *tests], plugins=[Recorder()])
receipt.update(pytest_exit_code=int(code), cuda_after=torch.cuda.is_initialized(), tests_results=rows)
args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
assert not torch.cuda.is_initialized()
assert not git('status', '--porcelain', '--untracked-files=no')
if args.expect_alias_failure:
    failed = [row for row in rows if row['outcome'] == 'failed']
    assert int(code) == 1 and len(failed) == 4, receipt
    assert all('test_tied_names_are_transferred_each_round' in row['nodeid'] for row in failed)
    assert sum(row['outcome'] == 'passed' for row in rows) == 1
    print('Confirmed: all four unchanged alias regressions fail against the unmodified base.')
else:
    raise SystemExit(code)
