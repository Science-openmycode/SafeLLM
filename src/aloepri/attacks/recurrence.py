from __future__ import annotations

from collections import Counter


def frequency_rank_encode(sequence: list[int]) -> list[int]:
    """Replace symbols by frequency rank; ties use first occurrence then symbol id."""
    counts = Counter(sequence)
    first = {token: sequence.index(token) for token in counts}
    ordered = sorted(counts, key=lambda token: (-counts[token], first[token], token))
    rank = {token: index + 1 for index, token in enumerate(ordered)}
    return [rank[token] for token in sequence]
