"""Apply the production deterministic NCCL settings before any CUDA init."""

import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

import torch
import vllm
from vllm.model_executor.determinism.batch_invariant import override_envs_for_invariance


from source_provenance import ROOT as root, SOURCE as source, verify_source

verify_source()
assert Path(vllm.__file__).resolve().is_relative_to(source)
gpus = [int(value) for value in os.environ["TIED_SYNC_GPUS"].split(",")]
tp, pp = int(os.environ["TIED_SYNC_TP"]), int(os.environ["TIED_SYNC_PP"])
assert tp in (1, 2) and pp in (1, 2)
assert len(gpus) == 1 + tp * pp and len(set(gpus)) == len(gpus)
assert all(0 <= gpu <= 7 for gpu in gpus)
assert os.environ["TIED_SYNC_PACKED"] in ("0", "1")
assert os.environ["CUDA_VISIBLE_DEVICES"] == ",".join(map(str, gpus))
assert os.environ["VLLM_BATCH_INVARIANT"] == "1"
assert not torch.cuda.is_initialized()
before = dict(os.environ)
override_envs_for_invariance()
assert not torch.cuda.is_initialized()
assert shutil.which("ninja") == str(Path(sys.executable).parent / "ninja")
output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
function_source = inspect.getsource(override_envs_for_invariance)
selected = {
    key: value for key, value in os.environ.items()
    if key.startswith(("NCCL_", "TIED_SYNC_")) or key in (
        "VLLM_ALLREDUCE_USE_SYMM_MEM", "CUBLAS_WORKSPACE_CONFIG",
        "VLLM_USE_AOT_COMPILE", "VLLM_BATCH_INVARIANT", "PATH",
        "NO_PROXY", "no_proxy", "CUDA_VISIBLE_DEVICES", "PYTHONPATH",
    )
}
manifest = {
    "argv": sys.argv,
    "python": sys.executable,
    "pid": os.getpid(),
    "cuda_initialized": torch.cuda.is_initialized(),
    "env_changed_by_production_helper": {
        key: value for key, value in os.environ.items() if before.get(key) != value
    },
    "selected_launch_env": selected,
    "function_source_file": inspect.getsourcefile(override_envs_for_invariance),
    "function_source": function_source,
    "function_source_sha256": hashlib.sha256(function_source.encode()).hexdigest(),
    "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
output.with_name(output.name + "-launch.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
driver = Path(__file__).with_name("run_tied_weight_sync_validation.py")
sys.argv[0] = str(driver)
runpy.run_path(str(driver), run_name="__main__")
