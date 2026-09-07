from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_tee_split_qwen import DEFAULT_PROMPTS, load_prompts


def test_default_prompts_are_available() -> None:
    assert load_prompts(None) == DEFAULT_PROMPTS


def test_prompt_manifest_accepts_strings_and_records(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps({"prompts": ["问题一", {"prompt": "Question two"}]}),
        encoding="utf-8",
    )
    assert load_prompts(path) == ("问题一", "Question two")


def test_prompt_manifest_rejects_empty_prompt(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(["  "]), encoding="utf-8")
    with pytest.raises(ValueError, match="empty prompt"):
        load_prompts(path)
