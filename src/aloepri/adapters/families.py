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
        optional = set(_GLOBAL_OPTIONAL)
        if bool(config.get("tie_word_embeddings", False)):
            expected.remove("lm_head.weight")
            optional.add("lm_head.weight")
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
        if not bool(config.get("attention_bias", True)):
            for name in tuple(expected):
                if name.endswith("_proj.bias"):
                    expected.remove(name)
                    optional.add(name)
        return _coverage(inventory, expected, optional)


class GLMDenseFamilyAdapter(ModelFamilyAdapter):
    """Recognize the standard Transformers GLM dense checkpoint layout.

    GLM uses a fused SwiGLU gate/up projection.  It is deliberately a distinct
    adapter from Qwen2 so the product cannot silently run Qwen's converter over
    an incompatible tensor layout.
    """

    adapter_id = "glm_dense"

    def match(
        self,
        config: Mapping[str, Any],
        inventory: TensorInventory | None,
        fingerprint: ArchitectureFingerprint,
    ) -> AdapterMatch:
        matches = fingerprint.model_type == "glm" and fingerprint.ffn == "dense"
        return AdapterMatch(
            MatchStatus.SUPPORTED if matches else MatchStatus.INCOMPATIBLE,
            self.adapter_id if matches else None,
            fingerprint,
        )

    def validate_inventory(
        self, config: Mapping[str, Any], inventory: TensorInventory
    ) -> CoverageReport:
        expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
        optional = set(_GLOBAL_OPTIONAL)
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
                    f"{prefix}.mlp.gate_up_proj.weight",
                    f"{prefix}.mlp.down_proj.weight",
                }
            )
        return _coverage(inventory, expected, optional)


class Qwen3DenseFamilyAdapter(ModelFamilyAdapter):
    adapter_id = "qwen3_dense"

    def match(
        self,
        config: Mapping[str, Any],
        inventory: TensorInventory | None,
        fingerprint: ArchitectureFingerprint,
    ) -> AdapterMatch:
        matches = fingerprint.model_type == "qwen3" and fingerprint.ffn == "dense"
        return AdapterMatch(
            MatchStatus.SUPPORTED if matches else MatchStatus.INCOMPATIBLE,
            self.adapter_id if matches else None,
            fingerprint,
            (),
        )

    def validate_inventory(
        self, config: Mapping[str, Any], inventory: TensorInventory
    ) -> CoverageReport:
        expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
        optional = set(_GLOBAL_OPTIONAL)
        for layer in range(int(config["num_hidden_layers"])):
            prefix = f"model.layers.{layer}"
            expected.update(
                {
                    f"{prefix}.input_layernorm.weight",
                    f"{prefix}.post_attention_layernorm.weight",
                    f"{prefix}.self_attn.q_norm.weight",
                    f"{prefix}.self_attn.k_norm.weight",
                    f"{prefix}.self_attn.q_proj.weight",
                    f"{prefix}.self_attn.k_proj.weight",
                    f"{prefix}.self_attn.v_proj.weight",
                    f"{prefix}.self_attn.o_proj.weight",
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight",
                    f"{prefix}.mlp.down_proj.weight",
                }
            )
        return _coverage(inventory, expected, optional)


class GLM4MoEFamilyAdapter(ModelFamilyAdapter):
    adapter_id = "glm4_moe"

    def match(
        self,
        config: Mapping[str, Any],
        inventory: TensorInventory | None,
        fingerprint: ArchitectureFingerprint,
    ) -> AdapterMatch:
        matches = fingerprint.model_type in {"glm4_moe", "glm_moe_dsa"}
        return AdapterMatch(
            MatchStatus.SUPPORTED if matches else MatchStatus.INCOMPATIBLE,
            self.adapter_id if matches else None,
            fingerprint,
            (),
        )

    def validate_inventory(
        self, config: Mapping[str, Any], inventory: TensorInventory
    ) -> CoverageReport:
        expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
        optional = set(_GLOBAL_OPTIONAL)
        main_layers = int(config["num_hidden_layers"])
        mtp_layers = int(config.get("num_nextn_predict_layers") or 0)
        for layer in range(main_layers + mtp_layers):
            prefix = f"model.layers.{layer}"
            expected.update(
                {
                    f"{prefix}.input_layernorm.weight",
                    f"{prefix}.post_attention_layernorm.weight",
                    f"{prefix}.self_attn.q_proj.weight",
                    f"{prefix}.self_attn.k_proj.weight",
                    f"{prefix}.self_attn.v_proj.weight",
                    f"{prefix}.self_attn.o_proj.weight",
                }
            )
            if bool(config.get("attention_bias", False)):
                expected.update(
                    {
                        f"{prefix}.self_attn.q_proj.bias",
                        f"{prefix}.self_attn.k_proj.bias",
                        f"{prefix}.self_attn.v_proj.bias",
                    }
                )
            if bool(config.get("use_qk_norm", False)):
                expected.update(
                    {
                        f"{prefix}.self_attn.q_norm.weight",
                        f"{prefix}.self_attn.k_norm.weight",
                    }
                )
            if layer < int(config.get("first_k_dense_replace") or 0):
                expected.update(
                    {
                        f"{prefix}.mlp.gate_proj.weight",
                        f"{prefix}.mlp.up_proj.weight",
                        f"{prefix}.mlp.down_proj.weight",
                    }
                )
            else:
                expected.update(
                    {
                        f"{prefix}.mlp.gate.weight",
                        f"{prefix}.mlp.gate.e_score_correction_bias",
                        f"{prefix}.mlp.shared_experts.gate_proj.weight",
                        f"{prefix}.mlp.shared_experts.up_proj.weight",
                        f"{prefix}.mlp.shared_experts.down_proj.weight",
                    }
                )
                for expert in range(int(config["n_routed_experts"])):
                    expert_prefix = f"{prefix}.mlp.experts.{expert}"
                    expected.update(
                        {
                            f"{expert_prefix}.gate_proj.weight",
                            f"{expert_prefix}.up_proj.weight",
                            f"{expert_prefix}.down_proj.weight",
                        }
                    )
            if layer >= main_layers:
                expected.update(
                    {
                        f"{prefix}.embed_tokens.weight",
                        f"{prefix}.enorm.weight",
                        f"{prefix}.hnorm.weight",
                        f"{prefix}.eh_proj.weight",
                        f"{prefix}.shared_head.head.weight",
                        f"{prefix}.shared_head.norm.weight",
                    }
                )
        for name in tuple(expected):
            if not name.endswith(".weight"):
                continue
            scale_name = f"{name.removesuffix('.weight')}.weight_scale"
            if inventory.dtypes.get(name, "").upper().startswith("F8"):
                expected.add(scale_name)
            else:
                optional.add(scale_name)
        return _coverage(inventory, expected, optional)


class KimiK2FamilyAdapter(ModelFamilyAdapter):
    adapter_id = "kimi_k2"

    def match(
        self,
        config: Mapping[str, Any],
        inventory: TensorInventory | None,
        fingerprint: ArchitectureFingerprint,
    ) -> AdapterMatch:
        text = config.get("text_config")
        text_type = str(text.get("model_type", "")) if isinstance(text, Mapping) else ""
        matches = fingerprint.model_type in {"kimi_k2", "kimi_k25"} or text_type == "kimi_k2"
        pure_text = fingerprint.model_type == "kimi_k2" and not isinstance(text, Mapping)
        return AdapterMatch(
            MatchStatus.SUPPORTED if matches else MatchStatus.INCOMPATIBLE,
            self.adapter_id if matches else None,
            fingerprint,
            (
                "Kimi-K2.6 is converted as its complete text backbone; the vision tower "
                "is retained outside the private token-ID runtime boundary",
            )
            if matches and not pure_text
            else (),
        )

    def validate_inventory(
        self, config: Mapping[str, Any], inventory: TensorInventory
    ) -> CoverageReport:
        if config.get("model_type") == "kimi_k2" and not isinstance(
            config.get("text_config"), Mapping
        ):
            normalized = dict(config)
            normalized["model_type"] = "deepseek_v3"
            return DeepSeekV3FamilyAdapter().validate_inventory(normalized, inventory)
        text = config.get("text_config")
        if not isinstance(text, Mapping):
            return CoverageReport(frozenset(), frozenset(), ("text_config",), ())
        normalized_names: set[str] = set()
        normalized_shapes: dict[str, tuple[int, ...]] = {}
        normalized_dtypes: dict[str, str] = {}
        for name in inventory.names:
            if not name.startswith("language_model."):
                continue
            canonical = name.removeprefix("language_model.")
            if canonical.endswith(".weight_packed"):
                virtual = f"{canonical.removesuffix('.weight_packed')}.weight"
                normalized_names.add(virtual)
                normalized_shapes[virtual] = inventory.shapes[name]
                scale_name = f"{name.removesuffix('.weight_packed')}.weight_scale"
                normalized_dtypes[virtual] = inventory.dtypes.get(scale_name, "BF16")
            elif canonical.endswith((".weight_scale", ".weight_shape")):
                packed_name = f"{name.rsplit('.', 1)[0]}.weight_packed"
                if packed_name in inventory.names:
                    continue
                normalized_names.add(canonical)
                normalized_shapes[canonical] = inventory.shapes[name]
                normalized_dtypes[canonical] = inventory.dtypes[name]
            else:
                normalized_names.add(canonical)
                normalized_shapes[canonical] = inventory.shapes[name]
                normalized_dtypes[canonical] = inventory.dtypes[name]
        normalized_config = dict(text)
        normalized_config["model_type"] = "deepseek_v3"
        normalized_inventory = TensorInventory(
            frozenset(normalized_names), normalized_shapes, normalized_dtypes
        )
        text_coverage = DeepSeekV3FamilyAdapter().validate_inventory(
            normalized_config, normalized_inventory
        )
        allowed_non_text = {
            name
            for name in inventory.names
            if name.startswith(("vision_tower.", "mm_projector."))
        }
        physical_text = {
            name for name in inventory.names if name.startswith("language_model.")
        }
        recognized = physical_text | allowed_non_text
        unknown = set(inventory.names) - recognized
        return CoverageReport(
            frozenset(f"language_model.{name}" for name in text_coverage.expected),
            frozenset(recognized),
            text_coverage.missing,
            tuple(sorted(unknown | set(text_coverage.unknown))),
        )


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


def _replace_individual_experts_with_fused(
    expected: set[str], inventory: TensorInventory, layer: int
) -> None:
    prefix = f"model.layers.{layer}.mlp.experts"
    gate_up = f"{prefix}.gate_up_proj"
    down = f"{prefix}.down_proj"
    if {gate_up, down}.issubset(inventory.names):
        expected.difference_update(
            name for name in tuple(expected) if name.startswith(f"{prefix}.")
        )
        expected.update({gate_up, down})


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
            _replace_individual_experts_with_fused(expected, inventory, layer)
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
            _replace_individual_experts_with_fused(expected, inventory, layer)
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
            _replace_individual_experts_with_fused(expected, inventory, layer)
        optional = set(_GLOBAL_OPTIONAL)
        optional.update(
            name.replace(".mlp.gate.weight", ".mlp.gate.e_score_correction_bias")
            for name in expected
            if name.endswith(".mlp.gate.weight")
        )
        if (config.get("quantization_config") or {}).get("quant_method") == "fp8":
            _add_fp8_scales(expected, inventory)
        return _coverage(inventory, expected, optional)
