from __future__ import annotations

from pathlib import Path

import torch
from lm_eval.models.huggingface import HFLM
from safetensors.torch import load_file


class NonEmptyContextHFLM(HFLM):
    """Work around whitespace-only contexts in lm-eval multiple-choice tasks."""

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        context_ids, continuation_ids = super()._encode_pair(context, continuation)
        if not context_ids:
            context_ids = [self.prefix_token_id]
        if not continuation_ids:
            continuation_ids = self.tok_encode(continuation, add_special_tokens=False)
        if not continuation_ids:
            raise ValueError(
                "lm-eval produced an empty continuation after boundary tokenization: "
                f"context={context!r}, continuation={continuation!r}"
            )
        return context_ids, continuation_ids


class PrivateTokenHFLM(NonEmptyContextHFLM):
    def __init__(self, *, key_path: Path, **kwargs: object) -> None:
        super().__init__(**kwargs)
        key = load_file(key_path, device="cpu")
        self.tau = key["tau"].to(self.device)
        self.inverse_tau = key["inverse_tau"].to(self.device)

    @property
    def eot_token_id(self) -> int:
        return int(self.tau[self.tokenizer.eos_token_id])

    @property
    def prefix_token_id(self) -> int:
        original = self.custom_prefix_token_id
        if original is None:
            original = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        return int(self.tau[original])

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        left_truncate_len: int | None = None,
        **kwargs: object,
    ) -> list[int]:
        plain = super().tok_encode(
            string,
            add_special_tokens=add_special_tokens,
            left_truncate_len=left_truncate_len,
            **kwargs,
        )
        return self.tau[torch.tensor(plain, device=self.device)].cpu().tolist()

    def tok_batch_encode(
        self,
        strings: list[str],
        padding_side: str = "left",
        left_truncate_len: int | None = None,
        truncation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        plain_ids, attention_mask = super().tok_batch_encode(
            strings,
            padding_side=padding_side,
            left_truncate_len=left_truncate_len,
            truncation=truncation,
        )
        return self.tau[plain_ids.to(self.device)], attention_mask.to(self.device)

    def tok_decode(self, tokens, skip_special_tokens: bool = True):
        private = torch.as_tensor(tokens, dtype=torch.int64, device=self.device)
        plain = self.inverse_tau[private].cpu().tolist()
        return self.tokenizer.decode(plain, skip_special_tokens=skip_special_tokens)

    def _model_generate(
        self,
        context: torch.Tensor,
        max_length: int,
        stop: list[str],
        **generation_kwargs: object,
    ) -> torch.Tensor:
        del stop
        if generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
            generation_kwargs.setdefault("do_sample", False)
        generation_kwargs.pop("pad_token_id", None)
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            pad_token_id=int(self.tau[self.tokenizer.pad_token_id]),
            use_cache=True,
            **generation_kwargs,
        )
