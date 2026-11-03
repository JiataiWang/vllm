"""Read-only rank-local weight census for the tied-weight sync experiment."""

import hashlib
import os
from pathlib import Path

import torch

from source_provenance import SOURCE, runtime_provenance, verify_source


class TiedWeightProbe:
    def tied_weight_census(self):
        import vllm
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        source = SOURCE
        verify_source()
        assert Path(vllm.__file__).resolve().is_relative_to(source)
        model = self.model_runner.model
        params = dict(model.named_parameters(remove_duplicate=False))
        start, end = model.model.start_layer, model.model.end_layer
        names = [
            f"model.layers.{layer}.self_attn.{projection}.weight"
            for layer in sorted({start, end - 1})
            for projection in ("qkv_proj", "o_proj")
        ]
        if get_pp_group().is_first_rank:
            names.append("model.embed_tokens.weight")
        if get_pp_group().is_last_rank:
            names.extend(("lm_head.weight", "model.norm.weight"))
        selected = {}
        for name in names:
            param = params[name]
            raw = param.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            selected[name] = {
                "shape": list(param.shape),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "first": float(param.flatten()[0].item()),
                "data_ptr": param.data_ptr(),
            }
        engine = getattr(self, "weight_transfer_engine", None)
        group = getattr(engine, "model_update_group", None)
        config = self.vllm_config.model_config
        return {
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device": torch.cuda.current_device(),
            "rank": self.vllm_config.parallel_config.rank,
            "tp_rank": get_tp_group().rank_in_group,
            "pp_rank": get_pp_group().rank_in_group,
            "start_layer": start,
            "end_layer": end,
            "tie_word_embeddings": config.hf_config.tie_word_embeddings,
            "word_embeddings_untied_by_checkpoint": config.word_embeddings_untied_by_checkpoint,
            "input_output_same_parameter": (
                model.model.embed_tokens.weight is model.lm_head.weight
                if get_pp_group().is_first_rank and get_pp_group().is_last_rank
                else None
            ),
            "update_group": None if group is None else {
                "rank": group.rank, "world_size": group.world_size,
            },
            "selected": selected,
            "source": vllm.__file__,
            "base_py_sha256": hashlib.sha256(
                (source / "vllm/distributed/weight_transfer/base.py").read_bytes()
            ).hexdigest(),
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime_provenance": runtime_provenance(),
        }
