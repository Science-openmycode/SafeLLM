from __future__ import annotations

import pytest

from scripts import build_deepseek_v2_lite_acceptance as acceptance


def test_utility_gate_checks_every_task_and_confidence_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(acceptance, "paired_runs_ok", lambda _report: True)
    report = {
        "dtype_match": True,
        "provenance": {"formal_run_binding": True},
        "tasks": [
            {
                "task": "mmlu",
                "absolute_change": -0.01,
                "paired_change_95_percent_ci": [-0.03, 0.01],
            }
        ],
    }
    assert acceptance.utility_check(report) == (True, [])
    report["tasks"][0]["paired_change_95_percent_ci"] = [-0.04, 0.01]
    passed, failures = acceptance.utility_check(report)
    assert passed is False
    assert "mmlu" in failures[0]


def test_generation_gates_require_full_sample_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(acceptance, "paired_runs_ok", lambda _report: True)
    ifeval = {
        "sample_len": 541,
        "dtype_match": True,
        "provenance": {"formal_run_binding": True},
        "metrics": {
            "prompt_level_strict_acc": {
                "absolute_change": 0.0,
                "paired_change_95_percent_ci": [0.0, 0.0],
            }
        },
    }
    humaneval = {
        "sample_len": 164,
        "dtype_match": True,
        "provenance": {"formal_run_binding": True},
        "absolute_change": 0.0,
        "paired_change_95_percent_ci": [0.0, 0.0],
    }
    assert acceptance.ifeval_check(ifeval) == (True, [])
    assert acceptance.humaneval_check(humaneval) == (True, [])
    ifeval["sample_len"] = 540
    humaneval["sample_len"] = 163
    assert acceptance.ifeval_check(ifeval)[0] is False
    assert acceptance.humaneval_check(humaneval)[0] is False
