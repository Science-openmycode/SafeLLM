from __future__ import annotations

from scripts.build_qwen05b_v47_acceptance import evaluate_rule, nested_value


def test_nested_acceptance_rule_evaluation() -> None:
    payload = {"metric": {"drop": 0.02, "ci": [-0.01, 0.03]}, "ok": True}
    assert nested_value(payload, "metric.drop") == 0.02
    assert nested_value(payload, "metric.ci.0") == -0.01
    assert evaluate_rule(payload, {"path": "metric.drop", "maximum": 0.035})["pass"]
    assert evaluate_rule(payload, {"path": "ok", "equals": True})["pass"]


def test_missing_acceptance_field_fails() -> None:
    result = evaluate_rule({}, {"path": "missing", "equals": True})
    assert result == {"path": "missing", "status": "MISSING_FIELD", "pass": False}
