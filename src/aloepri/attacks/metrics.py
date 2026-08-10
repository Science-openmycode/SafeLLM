from __future__ import annotations

import math
from collections import Counter


def _ngrams(tokens: list[int], order: int) -> Counter[tuple[int, ...]]:
    return Counter(tuple(tokens[start : start + order]) for start in range(len(tokens) - order + 1))


def corpus_bleu4(
    references: list[list[int]],
    hypotheses: list[list[int]],
    *,
    smoothing_epsilon: float = 0.1,
) -> float:
    """Compute single-reference corpus BLEU-4 on a 0-100 scale.

    Clipped n-gram counts are aggregated over the corpus. Zero numerators use the
    same epsilon-over-denominator smoothing convention as NLTK method 1.
    """

    if len(references) != len(hypotheses) or not references:
        raise ValueError("BLEU requires equally sized, non-empty reference and hypothesis lists")
    if smoothing_epsilon <= 0:
        raise ValueError("smoothing_epsilon must be positive")
    numerators = [0, 0, 0, 0]
    denominators = [0, 0, 0, 0]
    reference_length = 0
    hypothesis_length = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        reference_length += len(reference)
        hypothesis_length += len(hypothesis)
        for order in range(1, 5):
            reference_counts = _ngrams(reference, order)
            hypothesis_counts = _ngrams(hypothesis, order)
            numerators[order - 1] += sum(
                min(count, reference_counts[ngram]) for ngram, count in hypothesis_counts.items()
            )
            denominators[order - 1] += max(0, len(hypothesis) - order + 1)
    if hypothesis_length == 0:
        return 0.0
    precisions = [
        (numerator / denominator if numerator else smoothing_epsilon / denominator)
        if denominator
        else smoothing_epsilon
        for numerator, denominator in zip(numerators, denominators, strict=True)
    ]
    brevity_penalty = (
        1.0
        if hypothesis_length >= reference_length
        else math.exp(1.0 - reference_length / hypothesis_length)
    )
    return 100.0 * brevity_penalty * math.exp(sum(math.log(value) for value in precisions) / 4)
