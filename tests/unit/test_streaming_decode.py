from __future__ import annotations

from aloepri.client.streaming_decode import (
    IncrementalTokenDecoder,
    build_byte_level_token_table,
)


class _ByteTokenizer:
    all_special_ids = [3]
    _tokens = ["H", "i", "Ġ", "<special>"]

    def __len__(self) -> int:
        return len(self._tokens)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self._tokens[token_id]

    def decode(self, token_ids: list[int], **_kwargs: object) -> str:
        return "".join(" " if value == 2 else self._tokens[value] for value in token_ids)


class _FallbackTokenizer:
    all_special_ids: list[int] = []

    def __len__(self) -> int:
        return 2

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return ["▁hello", "world"][token_id]

    def decode(self, token_ids: list[int], **_kwargs: object) -> str:
        return ",".join(str(value) for value in token_ids)


def test_byte_level_decoder_emits_exact_deltas_and_skips_specials() -> None:
    tokenizer = _ByteTokenizer()
    table = build_byte_level_token_table(tokenizer)
    assert table is not None
    decoder = IncrementalTokenDecoder(
        tokenizer, token_bytes=table, build_if_missing=False
    )
    assert [decoder.push(value) for value in [0, 1, 2, 3]] == ["H", "i", " ", ""]
    assert decoder.finish() == ""
    assert decoder.text == "Hi "
    assert decoder.optimized is True


def test_non_byte_tokenizer_uses_exact_cumulative_fallback() -> None:
    tokenizer = _FallbackTokenizer()
    assert build_byte_level_token_table(tokenizer) is None
    decoder = IncrementalTokenDecoder(tokenizer)
    assert decoder.push(0) == "0"
    assert decoder.push(1) == ",1"
    assert decoder.text == "0,1"
    assert decoder.optimized is False
