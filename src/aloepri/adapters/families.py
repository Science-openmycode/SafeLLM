from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aloepri.adapters.base import CoverageReport, ModelFamilyAdapter, TensorInventory
from aloepri.catalog.models import AdapterMatch, ArchitectureFingerprint, MatchStatus

_GLOBAL_OPTIONAL = frozenset({"model.rotary_emb.inv_freq"})


def _coverage(inventory: TensorInventory, expected: set[str], optional: set[str]) -> CoverageReport:
    names = set(inventory.names)
    recognized = names.intersection(expected | optional)
    return CoverageReport(
        frozenset(expected),
        frozenset(recognized),
        tuple(sorted(expected - names)),
        tuple(sorted(names - recognized)),
    )


def _add_fp8_scales(
    expected: set[str], inventory: TensorInventory, *, only_existing_weights: bool = False
) -> None:
    weights = (
        {name for name in inventory.names if name.endswith(".weight")}
        if only_existing_weights
        else {name for name in expected if name.endswith(".weight")}
    )
    for name in weights:
        dtype = inventory.dtypes.get(name, "")
        if dtype.upper().startswith("F8"):
            expected.add(f"{name.removesuffix('.weight')}.weight_scale_inv")


class Qwen2FamilyAdapter(ModelFamilyAdapter):
    adapter_id = "qwen2"

    def match(
        self,
        config: Mapping[str, Any],
        inventory: TensorInventory | None,
        fingerprint: ArchitectureFingerprint,
    ) -> AdapterMatch:
        supported = fingerprint.model_type in {"qwen2", "qwen2_5", "aloepri_qwen2"}
        return AdapterMatch(
            MatchStatus.SUPPORTED if supported else MatchStatus.INCOMPATIBLE,
            self.adapter_id if supported else None,
            fingerprint,
        )

    def validate_inventory(
        self, config: Mapping[str, Any], inventory: TensorInventory
    ) -> CoverageReport:
        expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
        for layer in range(int(config["num_hidden_layers"])):
            prefix = f"model.layers.{layer}"
            expected.update(
                {
                    f"{prefix}.input_layernorm.weight",
                    f"{prefix}.post_attention_layernorm.weight",
                    f"{prefix}.self_attn.q_proj.weight",
                    f"{prefix}.self_attn.q_proj.bias",
                    f"{prefix}.self_attn.k_proj.weight",
                    f"{prefix}.self_attn.k_proj.bias",
                    f"{prefix}.self_attn.v_proj.weight",
                    f"{prefix}.self_attn.v_proj.bias",
                    f"{prefix}.self_attn.o_proj.weight",
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight",
                    f"{prefix}.mlp.down_proj.weight",
                }
            )
        optional = set(_GLOBAL_OPTIONAL)
        if not bool(config.get("attention_bias", True)):
            for name in tuple(expected):
                if name.endswith("_proj.bias"):
                    expected.remove(name)
                    optional.add(name)
        return _coverage(inventory, expected, optional)


def _deepseek_layer_names(
    config: Mapping[str, Any], layer: int, *, force_moe: bool = False
) -> set[str]:
    prefix = f"model.layers.{layer}"
    names = {
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
        f"{prefix}.self_attn.kv_a_layernorm.weight",
        f"{prefix}.self_attn.kv_b_proj.weight",
        f"{prefix}.self_attn.o_proj.weight",
    }
    if config.get("q_lora_rank") is None:
        names.add(f"{prefix}.self_attn.q_proj.weight")
    else:
        names.update(
            {
                f"{prefix}.self_attn.q_a_proj.weight",
                f"{prefix}.self_attn.q_a_layernorm.weight",
                f"{prefix}.self_attn.q_b_proj.weight",
            }
        )
    dense_until = int(config.get("first_k_dense_replace") or 0)
    frequency = int(config.get("moe_layer_freq") or 1)
    is_moe = force_moe or (
        layer >= dense_until and (layer - dense_until) % frequency == 0
    )
    if is_moe and int(config.get("n_routed_experts") or 0):
        names.add(f"{prefix}.mlp.gate.weight")
        for expert in range(int(config["n_routed_experts"])):
            expert_prefix = f"{prefix}.mlp.experts.{expert}"
            names.update(
                {
                    f"{expert_prefix}.gate_proj.weight",
                    f"{expert_prefix}.up_proj.weight",
                    f"{expert_prefix}.down_proj.weight",
                }
            )
        if int(config.get("n_shared_experts") or 0):
            names.update(
                {
                    f"{prefix}.mlp.shared_experts.gate_proj.weight",
                    f"{prefix}.mlp.shared_experts.up_proj.weight",
                    f"{prefix}.mlp.shared_experts.down_proj.weight",
                }
            )
    else:
        names.update(
            {
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            }
        )
    return names


class DeepSeekV2FamilyAdapter(ModelFamilyAdapter):
    adapter_id = "deepseek_v2"

    def match(
        self,
        config: Mapping[str, Any],
        inventory: TensorInventory | None,
        fingerprint: ArchitectureFingerprint,
    ) -> AdapterMatch:
        supported = fingerprint.model_type == "deepseek_v2" and fingerprint.mtp_layers == 0
        return AdapterMatch(
            MatchStatus.SUPPORTED if supported else MatchStatus.INCOMPATIBLE,
            self.adapter_id if supported else None,
            fingerprint,
        )

    def validate_inventory(
        self, config: Mapping[str, Any], inventory: TensorInventory
    ) -> CoverageReport:
        expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
        for layer in range(int(config["num_hidden_layers"])):
            expected.update(_deepseek_layer_names(config, layer))
        optional = set(_GLOBAL_OPTIONAL)
        optional.update(
            name.replace(".mlp.gate.weight", ".mlp.gate.e_score_correction_bias")
            for name in expected
            if name.endswith(".mlp.gate.weight")
        )
        return _coverage(inventory, expected, optional)


class DeepSeekV3FamilyAdapter(DeepSeekV2FamilyAdapter):
    adapter_id = "deepseek_v3"

    def match(
        self,
        config: Mapping[str, Any],
        inventory: TensorInventory | None,
        fingerprint: ArchitectureFingerprint,
    ) -> AdapterMatch:
        if fingerprint.model_type not in {"deepseek_v3", "aloepri_deepseek_v3"}:
            return AdapterMatch(MatchStatus.INCOMPATIBLE, None, fingerprint)
        reasons: list[str] = []
        if fingerprint.weight_format == "fp8_block" and fingerprint.weight_block_size != (
            128,
            128,
        ):
            reasons.append("DeepSeek-V3 FP8 requires 128x128 weight blocks")
        if inventory is not None and fingerprint.mtp_layers:
            layer = int(config["num_hidden_layers"])
            if not any(name.startswith(f"model.layers.{layer}.") for name in inventory.names):
                reasons.append("config declares MTP but checkpoint has no MTP tensors")
        status = MatchStatus.INCOMPLETE_CHECKPOINT if reasons else MatchStatus.SUPPORTED
        return AdapterMatch(
            status,
            self.adapter_id if not reasons else None,
            fingerprint,
            tuple(reasons),
        )

    def validate_inventory(
        self, config: Mapping[str, Any], inventory: TensorInventory
    ) -> CoverageReport:
        expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
        layers = int(config["num_hidden_layers"])
        for layer in range(layers):
            expected.update(_deepseek_layer_names(config, layer))
        for offset in range(int(config.get("num_nextn_predict_layers") or 0)):
            layer = layers + offset
            prefix = f"model.layers.{layer}"
            expected.update(
                {
                    f"{prefix}.enorm.weight",
                    f"{prefix}.hnorm.weight",
                    f"{prefix}.eh_proj.weight",
                    f"{prefix}.embed_tokens.weight",
                    f"{prefix}.shared_head.norm.weight",
                    f"{prefix}.shared_head.head.weight",
                }
            )
            expected.update(_deepseek_layer_names(config, layer, force_moe=True))
        optional = set(_GLOBAL_OPTIONAL)
        optional.update(
            name.replace(".mlp.gate.weight", ".mlp.gate.e_score_correction_bias")
            for name in expected
            if name.endswith(".mlp.gate.weight")
        )
        if (config.get("quantization_config") or {}).get("quant_method") == "fp8":
            _add_fp8_scales(expected, inventory)
        return _coverage(inventory, expected, optional)
