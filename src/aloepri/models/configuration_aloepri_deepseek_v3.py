from __future__ import annotations

from typing import Any

from transformers import DeepseekV3Config


class AloePriDeepseekV3Config(DeepseekV3Config):  # type: ignore[no-untyped-call]
    """DeepSeek-V3 configuration with AloePri residual-coordinate expansion."""

    model_type = "aloepri_deepseek_v3"

    def __init__(
        self,
        *,
        plain_hidden_size: int = 1280,
        expansion_h: int = 128,
        aloepri_transform_version: str = "deepseek-v3-paper-complete-v1",
        aloepri_rms_mode: str = "paper_kappa",
        aloepri_rope_pair_order: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        private_hidden_size = plain_hidden_size + 2 * expansion_h
        supplied_hidden_size = int(kwargs.pop("hidden_size", private_hidden_size))
        if supplied_hidden_size != private_hidden_size:
            raise ValueError(
                "hidden_size must equal plain_hidden_size + 2*expansion_h: "
                f"{supplied_hidden_size} != {private_hidden_size}"
            )
        if aloepri_rms_mode not in {"paper_kappa", "exact_metric"}:
            raise ValueError(f"unsupported AloePri RMS mode: {aloepri_rms_mode}")
        kwargs["hidden_size"] = private_hidden_size
        # Embedding and LM Head receive independent paper noise and therefore
        # must not be re-tied by Transformers after checkpoint loading.
        kwargs["tie_word_embeddings"] = False
        super().__init__(**kwargs)
        self.plain_hidden_size = plain_hidden_size
        self.expansion_h = expansion_h
        self.aloepri_transform_version = aloepri_transform_version
        self.aloepri_rms_mode = aloepri_rms_mode
        self.aloepri_rope_pair_order = aloepri_rope_pair_order

    @classmethod
    def from_deepseek_v3_config(
        cls,
        base: DeepseekV3Config,
        *,
        expansion_h: int,
        rms_mode: str,
        rope_pair_order: list[int] | None,
        transform_version: str = "deepseek-v3-paper-complete-v1",
    ) -> AloePriDeepseekV3Config:
        values = base.to_dict()
        values.pop("model_type", None)
        values.pop("architectures", None)
        values.pop("hidden_size", None)
        values.pop("tie_word_embeddings", None)
        return cls(
            plain_hidden_size=base.hidden_size,
            expansion_h=expansion_h,
            aloepri_transform_version=transform_version,
            aloepri_rms_mode=rms_mode,
            aloepri_rope_pair_order=rope_pair_order,
            **values,
        )
