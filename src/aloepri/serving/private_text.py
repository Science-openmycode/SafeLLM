from __future__ import annotations

import base64
import hashlib
import hmac
import struct
from collections.abc import Iterable

PREFIX = "apids1."
MAGIC = b"APID\x01"
CHECKSUM_BYTES = 8


def encode_private_id_text(token_ids: Iterable[int]) -> str:
    """Encode private token IDs as reversible, tokenizer-free ASCII text."""
    values = list(token_ids)
    if not values:
        raise ValueError("private token text requires at least one token ID")
    if any(value < 0 or value > 0xFFFFFFFF for value in values):
        raise ValueError("private token ID is outside uint32 range")
    payload = MAGIC + struct.pack(">I", len(values)) + b"".join(
        struct.pack(">I", value) for value in values
    )
    checksum = hashlib.sha256(payload).digest()[:CHECKSUM_BYTES]
    encoded = base64.urlsafe_b64encode(payload + checksum).decode("ascii").rstrip("=")
    return PREFIX + encoded


def decode_private_id_text(value: str, *, max_tokens: int = 1_000_000) -> list[int]:
    """Decode and authenticate text produced by :func:`encode_private_id_text`."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if not value.startswith(PREFIX):
        raise ValueError("unsupported private token text version")
    encoded = value[len(PREFIX) :]
    if not encoded:
        raise ValueError("private token text payload is empty")
    padding = "=" * (-len(encoded) % 4)
    try:
        raw = base64.b64decode(encoded + padding, altchars=b"-_", validate=True)
    except ValueError as error:
        raise ValueError("private token text is not valid Base64URL") from error
    minimum = len(MAGIC) + 4 + CHECKSUM_BYTES
    if len(raw) < minimum or raw[: len(MAGIC)] != MAGIC:
        raise ValueError("private token text has an invalid header")
    payload, supplied_checksum = raw[:-CHECKSUM_BYTES], raw[-CHECKSUM_BYTES:]
    expected_checksum = hashlib.sha256(payload).digest()[:CHECKSUM_BYTES]
    if not hmac.compare_digest(supplied_checksum, expected_checksum):
        raise ValueError("private token text checksum mismatch")
    count = struct.unpack(">I", payload[len(MAGIC) : len(MAGIC) + 4])[0]
    if count < 1 or count > max_tokens:
        raise ValueError("private token text token count is outside configured limits")
    expected_bytes = len(MAGIC) + 4 + count * 4
    if len(payload) != expected_bytes:
        raise ValueError("private token text length does not match its token count")
    offset = len(MAGIC) + 4
    return [struct.unpack_from(">I", payload, offset + index * 4)[0] for index in range(count)]
