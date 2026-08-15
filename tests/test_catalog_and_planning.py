from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from aloepri.adapters.base import TensorInventory
from aloepri.adapters.registry import default_adapter_registry
from aloepri.catalog.models import MatchStatus
from aloepri.catalog.registry import builtin_catalog
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


def test_catalog_groups_multiple_architecture_families_without_overclaiming() -> None:
    entries = {entry.catalog_id: entry for entry in builtin_catalog().list()}
    assert entries["qwen2.5-0.5b-instruct"].family_name == "Qwen2 / Qwen2.5"
    assert entries["deepseek-v2-lite-chat"].family_name == "DeepSeek MLA / MoE"
    assert entries["deepseek-v2-lite-chat"].conversion_ready is True
    assert entries["glm-4-9b-chat-hf"].family_name == "GLM"
    assert entries["glm-4-9b-chat-hf"].conversion_ready is True
    assert entries["glm-4-9b-chat-hf"].deployment_ready is False
    assert entries["qwen3-8b"].conversion_ready is True
    assert entries["qwen3-8b"].deployment_ready is False
    assert entries["glm-4.7-fp8"].conversion_ready is True
    assert entries["glm-4.7-fp8"].deployment_ready is False
    assert entries["kimi-k2-instruct"].conversion_ready is True
    assert entries["kimi-k2-instruct"].deployment_ready is False
    assert entries["kimi-k2.6"].family_name == "Kimi"
    assert entries["kimi-k2.6"].conversion_ready is True
    assert entries["kimi-k2.6"].deployment_ready is False
    assert entries["qwen2.5-0.5b-instruct"].deployment_ready is True
    assert entries["glm-4-9b-0414"].conversion_ready is False
    assert entries["glm-4-9b-0414"].max_stage == "inspect"
    assert entries["qwen3-4b"].max_stage == "convert"
    openseek = entries["openseek-small-v1-sft"]
    assert openseek.adapter_id == "deepseek_v3"
    assert openseek.conversion["expansion_h"] == 128
    assert openseek.conversion["estimated_output_ratio"] >= 1.2
    assert openseek.status == "supported"
    assert openseek.deployment_ready is True
    assert openseek.max_stage == "chat"


def test_new_families_have_explicit_conversion_readiness() -> None:
    registry = default_adapter_registry()
    glm = registry.detect(
        {
            "model_type": "glm",
            "num_hidden_layers": 1,
            "num_key_value_heads": 2,
        }
    )
    assert glm.status == MatchStatus.SUPPORTED
    assert glm.adapter_id == "glm_dense"
    glm4 = registry.detect(
        {
            "model_type": "glm4",
            "num_hidden_layers": 1,
            "num_key_value_heads": 2,
        }
    )
    assert glm4.status == MatchStatus.EXPERIMENTAL
    assert glm4.adapter_id == "glm_dense"
    kimi = registry.detect(
        {
            "model_type": "kimi_k25",
            "text_config": {
                "model_type": "kimi_k2",
                "n_routed_experts": 384,
                "num_nextn_predict_layers": 0,
            },
        }
    )
    assert kimi.status == MatchStatus.SUPPORTED
    assert kimi.adapter_id == "kimi_k2"
    kimi_text = registry.detect(
        {
            "model_type": "kimi_k2",
            "n_routed_experts": 384,
            "num_nextn_predict_layers": 0,
        }
    )
    assert kimi_text.status == MatchStatus.SUPPORTED
    assert kimi_text.adapter_id == "kimi_k2"


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
