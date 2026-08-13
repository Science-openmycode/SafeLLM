from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from lm_eval.models.huggingface import HFLM
from safetensors.torch import load_file
from tqdm import tqdm


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

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[Any, list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
    ) -> list[tuple[float, bool]]:
        """Score causal-LM continuations without materializing full log-softmax.

        For a target token ``y``, ``log_softmax(z)[y]`` is exactly
        ``z[y] - logsumexp(z)``.  Computing only that value avoids a second
        sequence-by-vocabulary tensor, which is material for the expanded
        private checkpoint on a 6 GiB GPU.
        """

        if self.backend != "causal":
            return super()._loglikelihood_tokens(requests, disable_tqdm=disable_tqdm)
        configured_batch_size = override_bs if override_bs is not None else self.batch_size
        batch_size = 1 if configured_batch_size == "auto" else int(configured_batch_size)
        if batch_size < 1:
            raise ValueError("loglikelihood batch size must be positive")
        indexed_requests = sorted(
            enumerate(requests),
            key=lambda row: -(len(row[1][1]) + len(row[1][2])),
        )
        results: list[tuple[float, bool] | None] = [None] * len(requests)
        progress = tqdm(
            total=len(requests),
            disable=disable_tqdm or self.rank != 0,
            desc="Running memory-efficient loglikelihood requests",
        )
        for start in range(0, len(indexed_requests), batch_size):
            chunk = indexed_requests[start : start + batch_size]
            prepared: list[tuple[int, Any, list[int], list[int]]] = []
            max_input_len = 0
            max_continuation_len = 0
            for original_index, (request_str, context_enc, continuation_enc) in chunk:
                if not context_enc or not continuation_enc:
                    raise ValueError("loglikelihood requires non-empty context and continuation")
                combined = (context_enc + continuation_enc)[-(self.max_length + 1) :]
                input_tokens = combined[:-1]
                prepared.append(
                    (original_index, request_str, input_tokens, continuation_enc)
                )
                max_input_len = max(max_input_len, len(input_tokens))
                max_continuation_len = max(
                    max_continuation_len, len(continuation_enc)
                )

            input_ids = torch.full(
                (len(prepared), max_input_len),
                fill_value=self.eot_token_id,
                dtype=torch.int64,
                device=self.device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, (_, _, input_tokens, _) in enumerate(prepared):
                token_tensor = torch.tensor(
                    input_tokens, dtype=torch.int64, device=self.device
                )
                input_ids[row, -len(input_tokens) :] = token_tensor
                attention_mask[row, -len(input_tokens) :] = 1
            position_ids = attention_mask.cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

            with torch.inference_mode():
                selected_batch = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    logits_to_keep=max_continuation_len,
                ).logits
            if selected_batch.shape[:2] != (len(prepared), max_continuation_len):
                raise RuntimeError("model did not honor batched logits_to_keep")

            for row, (original_index, request_str, _, continuation_enc) in enumerate(
                prepared
            ):
                selected = selected_batch[row, -len(continuation_enc) :]
                target = torch.tensor(
                    continuation_enc, dtype=torch.int64, device=selected.device
                )
                greedy_match = bool((selected.argmax(dim=-1) == target).all())
                target_logits = selected.gather(1, target.unsqueeze(1)).squeeze(1).float()
                log_normalizer = torch.logsumexp(selected.float(), dim=-1)
                answer = (float((target_logits - log_normalizer).sum()), greedy_match)
                results[original_index] = answer
                if request_str is not None:
                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                progress.update(1)
        progress.close()
        if any(result is None for result in results):
            raise RuntimeError("loglikelihood batching did not produce every result")
        return [result for result in results if result is not None]


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
