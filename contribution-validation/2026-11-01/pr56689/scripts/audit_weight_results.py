"""Audit recorded model outputs, full inspected shards and provenance offline."""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(actual, reference, exact):
    changed = []
    shard_count = 0
    for row, target in zip(actual['census'], reference['census'], strict=True):
        assert row['rank'] == target['rank']
        assert row['selected'].keys() == target['selected'].keys()
        for name, tensor in row['selected'].items():
            shard_count += 1
            assert tensor['shape'] == target['selected'][name]['shape']
            if tensor['sha256'] != target['selected'][name]['sha256']:
                changed.append({'rank': row['rank'], 'name': name})
    max_delta = 0.0
    finite_count = top_count = 0
    token_ids_exact = logprobs_exact = True
    for observed, expected in zip(actual['responses'], reference['responses'], strict=True):
        assert observed['usage'] == expected['usage']
        a, b = observed['choices'][0], expected['choices'][0]
        token_ids_exact &= a['token_ids'] == b['token_ids']
        logprobs_exact &= a['logprobs'] == b['logprobs']
        for x, y in zip(a['logprobs']['token_logprobs'][1:], b['logprobs']['token_logprobs'][1:], strict=True):
            assert math.isfinite(x) and math.isfinite(y)
            finite_count += 1
            max_delta = max(max_delta, abs(x-y))
        for values in a['logprobs']['top_logprobs']:
            if values is not None:
                assert all(math.isfinite(value) for value in values.values())
                top_count += len(values)
        if exact:
            for field in ('text','token_ids','prompt_token_ids','logprobs','prompt_logprobs','finish_reason','stop_reason'):
                assert a[field] == b[field], field
    if exact:
        assert not changed and token_ids_exact and logprobs_exact and max_delta == 0.0
    return {'changed_shards': changed, 'full_shards_compared': shard_count,
            'token_logprobs_compared': finite_count, 'top_logprob_entries_compared': top_count,
            'returned_token_ids_exact': token_ids_exact, 'all_logprobs_exact': logprobs_exact,
            'max_token_logprob_delta': max_delta}


def audit(directory, baseline=False):
    provenance = read(directory/'provenance.json')
    expected_head = ('fe77d0906be0c38c5e2c575cd61b545a2fa687e4' if baseline
                     else '2f25de1c9d3a31160b7f2e789428f5129dd006dd')
    source_manifest = read(directory/'source-manifest.json')
    assert provenance['head'] == source_manifest['head'] == expected_head
    assert provenance['native_wheel_commit'] == 'fe77d0906be0c38c5e2c575cd61b545a2fa687e4'
    assert not provenance['git_status']
    for name, digest in provenance['harness_sha256'].items():
        assert sha(directory/name) == digest
    assert sha(directory/'source.patch') == provenance['source_patch_sha256']
    assert sha(directory/'tested-base.py') == provenance['base_py_sha256']
    assert source_manifest['tracked_files']['vllm/distributed/weight_transfer/base.py']['sha256'] == provenance['base_py_sha256']
    assert read(directory/'source-after.json') == {'patch_unchanged': True, 'base_py_unchanged': True}
    tp, pp = provenance['tensor_parallel_size'], provenance['pipeline_parallel_size']
    assert tp == 2 and pp in (1,2)
    assert not list(directory.glob('*-live-server.json'))
    for arm in ('prepare','cold_A','cold_B','warm'):
        receipt = read(directory/f'{arm}-exit.json')
        assert receipt['returncode'] == 0 and receipt['error'] is None
    for arm in ('cold_A','cold_B','warm'):
        text = (directory/f'{arm}-server.log').read_text()
        shutdown = text.find('[shutdown]')
        assert shutdown >= 0
        before = text[:shutdown]
        assert ' ERROR ' not in before and 'EngineDeadError' not in before
    prompts = read(directory/'prompts.json')
    observations = {}
    for label in ('cold_A','cold_B','warm_A_initial','warm_B','warm_A_restored'):
        observed = observations[label] = read(directory/f'{label}.json')
        assert observed['census'] == read(directory/f'{label}-census.json')
        assert len(observed['census']) == tp*pp
        assert len(observed['responses']) == len(prompts) == 2
        for ids, response in zip(prompts, observed['responses'], strict=True):
            choice = response['choices'][0]
            assert response['usage']['prompt_tokens'] == len(ids)
            assert response['usage']['completion_tokens'] == 8
            assert choice['prompt_token_ids'] == ids
            assert len(choice['logprobs']['token_logprobs']) == len(ids)+8
            assert choice['logprobs']['token_logprobs'][0] is None
            if len(choice['token_ids']) != 8:
                assert len(choice['token_ids']) == len(ids)+8
                assert choice['token_ids'][:len(ids)] == ids
        for rank, row in enumerate(observed['census']):
            assert row['rank'] == rank and row['tp_rank'] == rank%tp and row['pp_rank'] == rank//tp
            runtime = row['runtime_provenance']
            assert runtime['source']['head'] == expected_head
            assert runtime['source']['tree'] == source_manifest['tree']
            assert runtime['source']['path'] == provenance['source']
            assert runtime['native_libraries']
            for name, module in runtime['modules'].items():
                if name.startswith('vllm') and module['path'].endswith('.py'):
                    assert PurePosixPath(module['path']).is_relative_to(PurePosixPath(provenance['source'])/'vllm')
            assert row['base_py_sha256'] == provenance['base_py_sha256']
            assert row['probe_sha256'] == provenance['harness_sha256']['tied_weight_probe.py']
            if pp == 1:
                assert row['input_output_same_parameter'] is True
                assert row['selected']['model.embed_tokens.weight']['data_ptr'] == row['selected']['lm_head.weight']['data_ptr']
        expected_version = 'B' if label == 'warm_B' else 'A_restored' if label == 'warm_A_restored' else 'default'
        assert observed['weight_info']['weight_version'] == expected_version
    for label in ('warm_B','warm_A_restored'):
        for row, initial in zip(observations[label]['census'], observations['warm_A_initial']['census'], strict=True):
            assert row['pid'] == initial['pid']
            assert all(value['data_ptr'] == initial['selected'][name]['data_ptr'] for name,value in row['selected'].items())
    checks = {
        'initial_A': compare(observations['warm_A_initial'], observations['cold_A'], True),
        'updated_B': compare(observations['warm_B'], observations['cold_B'], not baseline),
        'restored_A': compare(observations['warm_A_restored'], observations['cold_A'], True),
        'negative_B_vs_A': compare(observations['cold_B'], observations['cold_A'], False),
    }
    assert checks['negative_B_vs_A']['changed_shards']
    assert checks['negative_B_vs_A']['max_token_logprob_delta'] > 0.02
    if baseline:
        assert checks['updated_B']['changed_shards'] == [{'rank':rank,'name':'lm_head.weight'} for rank in (2,3)]
        for row, original in zip(observations['warm_B']['census'], observations['cold_A']['census'], strict=True):
            if row['pp_rank'] == 1:
                assert row['selected']['lm_head.weight']['sha256'] == original['selected']['lm_head.weight']['sha256']
        assert read(directory/'expected-baseline-failure.json')['status'] == 'bug_reproduced'
    return {'directory':directory.name,'head':expected_head,'tree':source_manifest['tree'],
            'baseline':baseline,'tp':tp,'pp':pp,'packed':provenance['packed'],
            'checks':checks,'B_weights_sha256':read(directory/'prepared-evidence.json')['files']['B-checkpoint/model.safetensors']['sha256']}


parser = argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
args = parser.parse_args()
results = [audit(args.root/'evidence'/name, baseline) for name,baseline in (
    ('pr56689-tp2-pp2-packed-attempt2',False),
    ('pr56689-tp2-pp1-unpacked-attempt2',False),
    ('base56689-tp2-pp2-packed-attempt1',True),
)]
assert len({row['B_weights_sha256'] for row in results}) == 1
receipt = {'status':'passed','auditor_sha256':sha(__file__),'runs':results}
(args.root/'evidence/weight-results-independent-audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
