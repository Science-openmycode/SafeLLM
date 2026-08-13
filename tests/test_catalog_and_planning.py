from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from aloepri.adapters.base import TensorInventory
from aloepri.adapters.registry import default_adapter_registry
from aloepri.catalog.models import MatchStatus
from aloepri.jobs.store import JobState, JobStore
from aloepri.planning import ConversionPlan, build_local_plan


def _inventory(*names: str, dtype: str = "BF16") -> TensorInventory:
    return TensorInventory(
        frozenset(names),
        {name: (2, 2) for name in names},
        {name: dtype for name in names},
    )


def test_adapter_detection_uses_structure_not_repo_name() -> None:
    registry = default_adapter_registry()
    distilled = {
        "model_type": "qwen2",
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
    }
    match = registry.detect(distilled)
    assert match.status == MatchStatus.SUPPORTED
    assert match.adapter_id == "qwen2"
    assert match.fingerprint.attention == "gqa"


def test_deepseek_v3_rejects_missing_fp8_scale_and_mtp() -> None:
    config = {
        "model_type": "deepseek_v3",
        "num_hidden_layers": 1,
        "num_nextn_predict_layers": 1,
        "n_routed_experts": 2,
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    }
    inventory = _inventory("model.layers.0.self_attn.o_proj.weight", dtype="F8_E4M3")
    registry = default_adapter_registry()
    match = registry.detect(config, inventory)
    assert match.status == MatchStatus.INCOMPLETE_CHECKPOINT
    assert "MTP" in match.reasons[0]


def test_unknown_model_never_enters_conversion() -> None:
    match = default_adapter_registry().detect({"model_type": "untrusted_remote_code"})
    assert match.status == MatchStatus.INCOMPATIBLE
    assert match.adapter_id is None


def test_local_qwen_plan_reads_headers_without_loading_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "num_hidden_layers": 0,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "torch_dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(4, 2),
            "model.norm.weight": torch.ones(2),
            "lm_head.weight": torch.zeros(4, 2),
        },
        model_dir / "model.safetensors",
    )
    plan = build_local_plan(model_dir, output_uri=str(tmp_path / "private"))
    assert plan.adapter == "qwen2"
    assert plan.coverage["pass"] is True
    plan_path = tmp_path / "plan.yaml"
    plan.save(plan_path)
    assert ConversionPlan.load(plan_path).job_id == plan.job_id


def test_job_store_enforces_transitions_and_tile_resume(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "state.db")
    store.create("job-1", {"schema_version": 1})
    store.transition("job-1", JobState.PREFLIGHT)
    store.transition("job-1", JobState.CONVERTING)
    store.record_tile("job-1", "weight", 0, "abc")
    store.transition("job-1", JobState.PAUSED, progress={"tile": 1})
    assert store.resume("job-1") == JobState.CONVERTING
    assert store.completed_tiles("job-1", "weight") == {0: "abc"}
    with pytest.raises(ValueError, match="illegal job transition"):
        store.transition("job-1", JobState.RUNNING)
