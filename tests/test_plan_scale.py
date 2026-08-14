from __future__ import annotations

import json
import time
from pathlib import Path

from aloepri.adapters.deepseek_plan import build_deepseek_v3_static_plan


def test_static_planner_handles_more_than_one_hundred_thousand_modules() -> None:
    config = json.loads(
        Path("tests/fixtures/deepseek-v3-official/config.json").read_text(encoding="utf-8")
    )
    config["n_routed_experts"] = 600
    started = time.perf_counter()
    plan = build_deepseek_v3_static_plan(config).to_dict()
    elapsed = time.perf_counter() - started
    assert len(plan["modules"]) > 100_000
    assert elapsed < 15.0
