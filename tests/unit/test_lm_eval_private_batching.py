from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from aloepri.eval.lm_eval_private import NonEmptyContextHFLM


class _CacheHook:
    def __init__(self) -> None:
        self.rows: list[tuple[str, object, tuple[float, bool]]] = []

    def add_partial(self, method: str, key: object, value: tuple[float, bool]) -> None:
        self.rows.append((method, key, value))


class _DeterministicCausalLM:
    def __init__(self, vocab_size: int = 19) -> None:
        self.vocab_size = vocab_size
        self.batch_sizes: list[int] = []

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        use_cache: bool,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        assert not use_cache
        assert input_ids.shape == attention_mask.shape == position_ids.shape
        self.batch_sizes.append(input_ids.shape[0])
        tokens = input_ids[:, -logits_to_keep:]
        positions = position_ids[:, -logits_to_keep:]
        preferred = (tokens + positions + 1) % self.vocab_size
        vocabulary = torch.arange(self.vocab_size).view(1, 1, -1)
        logits = -(vocabulary - preferred.unsqueeze(-1)).abs().float()
        return SimpleNamespace(logits=logits)


def _evaluator(batch_size: int) -> NonEmptyContextHFLM:
    evaluator = NonEmptyContextHFLM.__new__(NonEmptyContextHFLM)
    evaluator.batch_size_per_gpu = batch_size
    evaluator._max_length = 8
    evaluator._device = torch.device("cpu")
    evaluator._rank = 0
    evaluator._model = _DeterministicCausalLM()
    evaluator.tokenizer = SimpleNamespace(eos_token_id=0)
    evaluator.cache_hook = _CacheHook()
    return evaluator


def _serial_reference(
    evaluator: NonEmptyContextHFLM,
    requests: list[tuple[object, list[int], list[int]]],
) -> list[tuple[float, bool]]:
    results = []
    vocab = torch.arange(evaluator.model.vocab_size)
    for _, context, continuation in requests:
        combined = (context + continuation)[-(evaluator.max_length + 1) :]
        inputs = torch.tensor(combined[:-1])
        positions = torch.arange(len(inputs))
        preferred = (inputs[-len(continuation) :] + positions[-len(continuation) :] + 1) % len(
            vocab
        )
        logits = -(vocab.view(1, -1) - preferred.view(-1, 1)).abs().float()
        target = torch.tensor(continuation)
        score = logits.gather(1, target.unsqueeze(1)).squeeze(1) - torch.logsumexp(
            logits, dim=-1
        )
        results.append((float(score.sum()), bool((logits.argmax(-1) == target).all())))
    return results


def test_batched_loglikelihood_matches_serial_reference_and_restores_order() -> None:
    requests = [
        (("short", " a"), [2, 3], [4]),
        (("truncated", " b"), list(range(1, 11)), [11, 12]),
        (("medium", " c"), [5, 6, 7, 8], [9, 10, 11]),
        (("other", " d"), [3, 1, 4], [1]),
    ]
    evaluator = _evaluator(batch_size=3)

    actual = evaluator._loglikelihood_tokens(requests, disable_tqdm=True)
    expected = _serial_reference(evaluator, requests)

    assert [row[0] for row in actual] == pytest.approx([row[0] for row in expected])
    assert [row[1] for row in actual] == [row[1] for row in expected]
    assert evaluator.model.batch_sizes == [3, 1]
    assert [row[1] for row in evaluator.cache_hook.rows] == [
        requests[1][0],
        requests[2][0],
        requests[3][0],
        requests[0][0],
    ]


def test_batched_loglikelihood_rejects_empty_context_or_continuation() -> None:
    evaluator = _evaluator(batch_size=2)
    with pytest.raises(ValueError, match="non-empty"):
        evaluator._loglikelihood_tokens([(("", "x"), [], [1])], disable_tqdm=True)


def test_batched_loglikelihood_honors_override_batch_size() -> None:
    evaluator = _evaluator(batch_size=1)
    requests = [((str(i), " x"), [1, 2], [3]) for i in range(5)]
    evaluator._loglikelihood_tokens(requests, disable_tqdm=True, override_bs=4)
    assert evaluator.model.batch_sizes == [4, 1]
