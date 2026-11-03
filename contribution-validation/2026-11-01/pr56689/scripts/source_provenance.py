"""Record exact source and native-library provenance without changing vLLM."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path('/home/shadeform/vllm-pr-validation-nov1')
SOURCE = Path(os.environ.get('TIED_SYNC_SOURCE', ROOT / 'pr56689')).resolve()
HEAD = os.environ.get('TIED_SYNC_HEAD', '2f25de1c9d3a31160b7f2e789428f5129dd006dd')
TREE = os.environ.get('TIED_SYNC_TREE', 'eb833d3579382cb99305773e2876cc211dabeb02')


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(SOURCE), *args], text=True).strip()


def verify_source(full=False):
    assert git('rev-parse', 'HEAD') == HEAD
    assert git('rev-parse', 'HEAD^{tree}') == TREE
    assert not git('status', '--porcelain', '--untracked-files=no')
    result = {'path': str(SOURCE), 'head': HEAD, 'tree': TREE}
    if full:
        names = subprocess.check_output(['git', '-C', str(SOURCE), 'ls-files', '-z']).decode().split('\0')
        result['tracked_files'] = {
            name: ({'symlink': os.readlink(SOURCE / name)} if (SOURCE / name).is_symlink()
                   else {'sha256': sha_file(SOURCE / name), 'size': (SOURCE / name).stat().st_size})
            for name in names if name
        }
    return result


def verify_model():
    from validate_fs_rope import verify_model as verify_pinned_model

    return verify_pinned_model(ROOT / 'models/Qwen3-0.6B-c1899de',
                               Path(__file__).with_name('model-revision-manifest.json'))


def runtime_provenance():
    import torch
    import vllm

    assert Path(vllm.__file__).resolve() == SOURCE / 'vllm/__init__.py'
    native = set()
    modules = {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not name.startswith(('vllm', 'flash_attn')) or not filename:
            continue
        path = Path(filename).resolve()
        if not path.is_file():
            continue
        if name.startswith('vllm') and path.suffix == '.py':
            assert path.is_relative_to(SOURCE / 'vllm'), (name, path)
        modules[name] = {'path': str(path), 'sha256': sha_file(path)}
        if '.so' in path.name:
            native.add(path)
    maps = Path('/proc/self/maps')
    if maps.exists():
        for line in maps.read_text().splitlines():
            path = line.split()[-1]
            if path.startswith('/') and '.so' in path and any(
                fragment in path for fragment in ('/vllm/', '/flash_attn/', 'vllm_flash_attn', 'libnccl', 'libcudart')
            ) and Path(path).is_file():
                native.add(Path(path).resolve())
    packages = {}
    for name in ('vllm', 'torch', 'transformers', 'ray', 'tokenizers', 'triton',
                 'numpy', 'safetensors', 'flashinfer-python', 'nvidia-nccl-cu13'):
        try:
            dist = importlib.metadata.distribution(name)
            packages[name] = {'version': dist.version, 'location': str(dist.locate_file(''))}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'source': verify_source(), 'python': sys.version, 'executable': sys.executable,
            'pid': os.getpid(), 'torch_cuda': torch.version.cuda, 'packages': packages,
            'modules': modules, 'native_libraries': {str(p): sha_file(p) for p in sorted(native)},
            'cuda_initialized': torch.cuda.is_initialized()}
