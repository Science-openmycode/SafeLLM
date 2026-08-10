from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

SECRET_METADATA_FIELDS = frozenset(
    {
        "seed",
        "embedding_noise_seed",
        "head_noise_seed",
        "noise_seed",
        "tau",
        "inverse_tau",
        "p",
        "q",
    }
)


def strip_secret_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return metadata that is safe to place in a server-side checkpoint."""
    return {
        str(key): value
        for key, value in metadata.items()
        if str(key).lower() not in SECRET_METADATA_FIELDS
        and not str(key).lower().endswith("_seed")
    }


def conversion_fingerprint(specification: Mapping[str, Any]) -> str:
    """Hash the complete conversion specification for safe resume checks."""
    canonical = json.dumps(
        dict(specification), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
