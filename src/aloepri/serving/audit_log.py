from __future__ import annotations

import json
import logging
from typing import Any

_FORBIDDEN_FIELDS = {
    "input_ids",
    "output_ids",
    "output_id",
    "prompt",
    "text",
    "tau",
    "inverse_tau",
}


def audit_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Write a structured event after rejecting token, text, and key material."""
    leaked = _FORBIDDEN_FIELDS.intersection(fields)
    if leaked:
        raise ValueError(f"sensitive audit fields are forbidden: {sorted(leaked)}")
    logger.info(json.dumps({"event": event, **fields}, separators=(",", ":"), default=str))
