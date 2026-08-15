from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aloepri.catalog.download import find_catalog_entry


def require_validated_hf_deployment(plan: Mapping[str, Any]) -> None:
    """Reject catalog deployments without checkpoint-specific HF acceptance.

    Architecture compatibility is enough to build a conversion plan, but it is
    not evidence that a particular checkpoint loads and generates on the HF
    runtime.  The release catalog records that distinction explicitly.
    """

    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("deployment plan has no source metadata")
    repo_id = source.get("repo_id")
    revision = source.get("revision")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError(
            "local/imported packages require a signed deployment acceptance record"
        )
    entry = find_catalog_entry(repo_id)
    if not entry.deployment_ready:
        raise ValueError(
            f"{entry.display_name} is conversion-compatible but has not passed "
            "checkpoint-specific deployment acceptance"
        )
    if entry.max_stage != "chat":
        raise ValueError(f"{entry.display_name} is not released for private chat")
    if entry.last_validated_revision != revision:
        raise ValueError(
            "model revision differs from the checkpoint-specific deployment acceptance"
        )
    preferred = str(entry.runtime.get("preferred", ""))
    if preferred != "hf":
        raise ValueError(
            f"{entry.display_name} requires the {preferred or 'declared'} runtime; "
            "the current one-click deployer only supports validated HF deployments"
        )
