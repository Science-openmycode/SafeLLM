"""Required vLLM registration entry point for expanded AloePri Qwen2 checkpoints.

The residual width is expanded while the attention head width remains fixed, so vLLM's
built-in Qwen2 implementation is not shape-compatible. Register this adapter before
loading checkpoints whose architecture is ``AloePriQwen2ForCausalLM``.
"""

from __future__ import annotations


def register() -> None:
    from transformers import AutoConfig
    from vllm import ModelRegistry

    from aloepri.models.configuration_aloepri_qwen2 import AloePriQwen2Config

    AutoConfig.register(AloePriQwen2Config.model_type, AloePriQwen2Config, exist_ok=True)
    ModelRegistry.register_model(
        "AloePriQwen2ForCausalLM",
        "aloepri.serving.vllm_qwen2:AloePriQwen2ForCausalLM",
    )
