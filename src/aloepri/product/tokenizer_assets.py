from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from aloepri.planning import ConversionPlan


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return normalized[:80] or "model"


def _candidate_roots(plan: ConversionPlan) -> tuple[Path, ...]:
    candidates: list[Path] = []
    output_uri = plan.output.get("uri")
    if output_uri:
        candidates.append(Path(str(output_uri)).resolve())
    source_root = plan.source.get("cache_path") or plan.source.get("path")
    if source_root:
        resolved = Path(str(source_root)).resolve()
        if resolved not in candidates:
            candidates.append(resolved)
    return tuple(candidates)


def _load_local_tokenizer(root: Path) -> Any:
    return AutoTokenizer.from_pretrained(
        root,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )


def materialize_local_tokenizer(
    plan: ConversionPlan, *, destination_root: Path
) -> Path:
    """Persist a self-contained tokenizer for the trusted local client.

    Converted output is checked before the source checkpoint. Some upstream
    checkpoints require repository Python code for tokenization, while AloePri's
    conversion emits a standard ``tokenizer.json`` which is safe to load locally.
    """

    selected: Path | None = None
    tokenizer: Any | None = None
    failures: list[str] = []
    for candidate in _candidate_roots(plan):
        if not candidate.is_dir():
            failures.append(f"{candidate}: directory is missing")
            continue
        try:
            tokenizer = _load_local_tokenizer(candidate)
        except (OSError, TypeError, ValueError) as error:
            failures.append(f"{candidate}: {error}")
            continue
        selected = candidate
        break
    if selected is None or tokenizer is None:
        detail = "; ".join(failures) if failures else "no tokenizer directory is configured"
        raise ValueError(f"no self-contained local tokenizer is available: {detail}")

    model_id = str(plan.output.get("model_id", selected.name))
    revision = str(plan.source.get("revision", plan.job_id))
    identity = hashlib.sha256(f"{model_id}\0{revision}".encode()).hexdigest()[:16]
    destination = destination_root / f"{_safe_component(model_id)}-{identity}"
    if destination.is_dir():
        try:
            _load_local_tokenizer(destination)
            return destination.resolve()
        except (OSError, TypeError, ValueError):
            shutil.rmtree(destination)

    destination_root.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    try:
        tokenizer.save_pretrained(partial)
        chat_template = selected / "chat_template.jinja"
        if chat_template.is_file() and not (partial / chat_template.name).is_file():
            shutil.copy2(chat_template, partial / chat_template.name)
        manifest = {
            "schema_version": 1,
            "model_id": model_id,
            "revision": revision,
            "source": str(selected),
            "files": sorted(path.name for path in partial.iterdir() if path.is_file()),
        }
        (partial / "yinbian_tokenizer.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _load_local_tokenizer(partial)
        os.replace(partial, destination)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return destination.resolve()
