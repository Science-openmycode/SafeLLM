from __future__ import annotations

from typing import Any

from transformers import Glm4MoeConfig


class AloePriGlm4MoeConfig(Glm4MoeConfig):
    """GLM4-MoE configuration with an expanded private residual stream."""

    model_type = "aloepri_glm4_moe"

    def __init__(
        self,
        *,
        plain_hidden_size: int = 5120,
        expansion_h: int = 128,
        aloepri_rms_mode: str = "exact_metric",
        aloepri_transform_version: str = "glm4-moe-paper-complete-v1",
        **kwargs: Any,
    ) -> None:
        private_hidden_size = plain_hidden_size + 2 * expansion_h
        supplied = int(kwargs.pop("hidden_size", private_hidden_size))
        if supplied != private_hidden_size:
            raise ValueError(
                "hidden_size must equal plain_hidden_size + 2*expansion_h: "
                f"{supplied} != {private_hidden_size}"
            )
        if aloepri_rms_mode not in {"paper_kappa", "exact_metric"}:
            raise ValueError(f"unsupported GLM4-MoE RMS mode: {aloepri_rms_mode}")
        kwargs["hidden_size"] = private_hidden_size
        kwargs["tie_word_embeddings"] = False
        super().__init__(**kwargs)
        self.plain_hidden_size = plain_hidden_size
        self.expansion_h = expansion_h
        self.aloepri_rms_mode = aloepri_rms_mode
        self.aloepri_transform_version = aloepri_transform_version

    @classmethod
    def from_glm4_moe_config(
        cls,
        base: Glm4MoeConfig,
        *,
        expansion_h: int,
        rms_mode: str = "exact_metric",
    ) -> AloePriGlm4MoeConfig:
        values = base.to_dict()
        for name in ("model_type", "architectures", "hidden_size", "tie_word_embeddings"):
            values.pop(name, None)
        return cls(
            plain_hidden_size=base.hidden_size,
            expansion_h=expansion_h,
            aloepri_rms_mode=rms_mode,
            **values,
        )
