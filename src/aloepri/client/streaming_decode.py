from __future__ import annotations

import codecs
from typing import Any


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    characters = values.copy()
    offset = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(256 + offset)
            offset += 1
    return dict(zip(values, (chr(value) for value in characters), strict=True))


def build_byte_level_token_table(
    tokenizer: Any,
) -> tuple[bytes | None, ...] | None:
    """Return an exact byte table for GPT-2/Qwen tokenizers, otherwise None."""

    inverse_bytes = {
        character: value for value, character in _gpt2_bytes_to_unicode().items()
    }
    if not hasattr(tokenizer, "convert_ids_to_tokens") or not hasattr(
        tokenizer, "all_special_ids"
    ):
        return None
    special_ids = set(int(value) for value in tokenizer.all_special_ids)
    table: list[bytes | None] = []
    for token_id in range(len(tokenizer)):
        if token_id in special_ids:
            table.append(None)
            continue
        token = tokenizer.convert_ids_to_tokens(token_id)
        if not isinstance(token, str) or any(
            character not in inverse_bytes for character in token
        ):
            return None
        table.append(bytes(inverse_bytes[character] for character in token))
    return tuple(table)


class IncrementalTokenDecoder:
    """Decode output IDs incrementally without repeatedly decoding the full prefix.

    Qwen/GPT-2 byte-level tokenizers use a true O(output tokens) UTF-8 stream.
    Other tokenizer families retain the exact cumulative fallback until a family-
    specific incremental decoder is available.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        token_bytes: tuple[bytes | None, ...] | None = None,
        build_if_missing: bool = True,
        skip_special_tokens: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.token_bytes = (
            build_byte_level_token_table(tokenizer)
            if token_bytes is None and build_if_missing
            else token_bytes
        )
        self._utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._ids: list[int] = []
        self._text = ""

    @property
    def text(self) -> str:
        return self._text

    @property
    def optimized(self) -> bool:
        return self.token_bytes is not None

    def push(self, token_id: int) -> str:
        self._ids.append(token_id)
        if self.token_bytes is not None:
            if not 0 <= token_id < len(self.token_bytes):
                raise ValueError("output token id is outside the tokenizer vocabulary")
            piece = self.token_bytes[token_id]
            delta = "" if piece is None else self._utf8.decode(piece, final=False)
            self._text += delta
            return delta
        decoded = self.tokenizer.decode(
            self._ids,
            skip_special_tokens=self.skip_special_tokens,
        )
        if not isinstance(decoded, str):
            raise TypeError("tokenizer.decode returned a batch")
        delta = decoded[len(self._text) :] if decoded.startswith(self._text) else decoded
        self._text = decoded
        return delta

    def finish(self) -> str:
        if self.token_bytes is None:
            return ""
        delta = self._utf8.decode(b"", final=True)
        self._text += delta
        return delta
