from __future__ import annotations

import json
from pathlib import Path

from aloepri.conversion.resource_estimate import estimate_deepseek_host_memory


def test_official_v3_plan_stays_inside_declared_host_budget() -> None:
    config = json.loads(
        Path("tests/fixtures/deepseek-v3-official/config.json").read_text(encoding="utf-8")
    )
    result = estimate_deepseek_host_memory(config, expansion_h=128)
    assert result.pass_ is True
    assert result.estimated_peak_gib <= 11.0
    assert result.vocabulary_transform_peak_gib > result.expert_transform_peak_gib
