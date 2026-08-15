from __future__ import annotations

from aloepri.adapters.base import TensorInventory
from aloepri.adapters.families import GLM4MoEFamilyAdapter, KimiK2FamilyAdapter


def _inventory(*names: str) -> TensorInventory:
    return TensorInventory(
        frozenset(names),
        {name: (2, 2) for name in names},
        {name: "BF16" for name in names},
    )


def test_pure_kimi_packed_int4_weights_are_validated_as_logical_weights() -> None:
    inventory = _inventory(
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight_packed",
        "lm_head.weight_scale",
        "lm_head.weight_shape",
    )
    report = KimiK2FamilyAdapter().validate_inventory(
        {
            "model_type": "kimi_k2",
            "num_hidden_layers": 0,
            "num_nextn_predict_layers": 0,
        },
        inventory,
    )
    assert report.pass_ is True


def test_glm_mtp_accepts_shared_global_embedding_and_head_layout() -> None:
    report = GLM4MoEFamilyAdapter().validate_inventory(
        {
            "num_hidden_layers": 0,
            "num_nextn_predict_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "first_k_dense_replace": 1,
            "n_routed_experts": 0,
        },
        _inventory(),
    )
    assert "model.layers.0.embed_tokens.weight" not in report.missing
    assert "model.layers.0.shared_head.head.weight" not in report.missing
    assert "model.layers.0.enorm.weight" in report.missing
    assert "model.layers.0.eh_proj.weight" in report.missing
