from __future__ import annotations

from typing import Any

_SENSITIVE_KEYS = {
    "authorization",
    "bearer_token",
    "hf_token",
    "input_ids",
    "inverse_tau",
    "offline_master_key",
    "output_ids",
    "password",
    "private_key",
    "private_key_passphrase",
    "prompt",
    "response_body",
    "secret",
    "tau",
    "token",
}


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if str(key).lower() in _SENSITIVE_KEYS
                else redact_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secrets(item) for item in value]
    return value
