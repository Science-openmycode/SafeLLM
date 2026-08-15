from __future__ import annotations

from typing import Any

from transformers import Qwen2Config


class AloePriQwen2Config(Qwen2Config):
    model_type = "aloepri_qwen2"

    def __init__(
        self,
        *,
        plain_hidden_size: int = 896,
        expansion_h: int = 128,
        head_dim: int = 64,
        aloepri_transform_version: int = 2,
        aloepri_rms_mode: str = "paper_kappa",
        aloepri_rms_representation: str = "gram",
        aloepri_attention_compute_dtype: str = "float32",
        aloepri_rope_style: str = "qwen_half_split",
        aloepri_rope_block_orders: list[list[list[int]]] | None = None,
        **kwargs: Any,
    ) -> None:
        private_hidden_size = plain_hidden_size + 2 * expansion_h
        supplied_hidden_size = kwargs.pop("hidden_size", private_hidden_size)
        if supplied_hidden_size != private_hidden_size:
            raise ValueError(
                f"hidden_size must equal plain_hidden_size + 2*expansion_h: "
                f"{supplied_hidden_size} != {private_hidden_size}"
            )
        kwargs["hidden_size"] = private_hidden_size
        kwargs["tie_word_embeddings"] = False
        super().__init__(**kwargs)
        self.plain_hidden_size = plain_hidden_size
        self.expansion_h = expansion_h
        self.head_dim = head_dim
        self.aloepri_transform_version = aloepri_transform_version
        if aloepri_rms_mode not in {"paper_kappa", "exact_metric"}:
            raise ValueError(f"unsupported AloePri RMS mode: {aloepri_rms_mode}")
        self.aloepri_rms_mode = aloepri_rms_mode
        if aloepri_rms_representation not in {"gram", "stable_factor"}:
            raise ValueError(
                "unsupported AloePri RMS representation: "
                f"{aloepri_rms_representation}"
            )
        self.aloepri_rms_representation = aloepri_rms_representation
        if aloepri_attention_compute_dtype not in {
            "float32",
            "float64",
            "float64_scores",
        }:
            raise ValueError(
                "unsupported AloePri attention compute dtype: "
                f"{aloepri_attention_compute_dtype}"
            )
        self.aloepri_attention_compute_dtype = aloepri_attention_compute_dtype
        if aloepri_rope_style not in {
            "qwen_half_split",
            "glm_interleaved_partial",
            "qwen3_qk_norm",
        }:
            raise ValueError(f"unsupported AloePri RoPE style: {aloepri_rope_style}")
        self.aloepri_rope_style = aloepri_rope_style
        self.aloepri_rope_block_orders = aloepri_rope_block_orders

    @classmethod
    def from_qwen2_config(
        cls,
        base: Qwen2Config,
        *,
        expansion_h: int,
        transform_version: int = 2,
        rms_mode: str = "paper_kappa",
        rms_representation: str = "gram",
        attention_compute_dtype: str = "float32",
    ) -> AloePriQwen2Config:
        values = base.to_dict()
        values.pop("model_type", None)
        values.pop("architectures", None)
        values.pop("hidden_size", None)
        values.pop("head_dim", None)
        values.pop("tie_word_embeddings", None)
        plain_hidden_size = base.hidden_size
        head_dim = getattr(base, "head_dim", plain_hidden_size // base.num_attention_heads)
        return cls(
            plain_hidden_size=plain_hidden_size,
            expansion_h=expansion_h,
            head_dim=head_dim,
            aloepri_transform_version=transform_version,
            aloepri_rms_mode=rms_mode,
            aloepri_rms_representation=rms_representation,
            aloepri_attention_compute_dtype=attention_compute_dtype,
            aloepri_rope_style=(
                "glm_interleaved_partial"
                if getattr(base, "aloepri_source_family", None) == "glm_dense"
                else (
                    "qwen3_qk_norm"
                    if (
                        getattr(base, "aloepri_source_family", None) == "qwen3_dense"
                        or getattr(base, "model_type", None) == "qwen3"
                    )
                    else "qwen_half_split"
                )
            ),
            **values,
        )
