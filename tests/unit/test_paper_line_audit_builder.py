from __future__ import annotations

from scripts.build_paper_line_audit import _symbol_override


def test_symbol_override_prefers_exact_symbol() -> None:
    overrides = {
        "DemoGateway": {"status": "ENGINEERING_SUBSTITUTE"},
        "DemoGateway.generate": {"status": "VERIFICATION"},
    }

    result = _symbol_override("DemoGateway.generate", overrides)

    assert result == {"status": "VERIFICATION"}


def test_symbol_override_inherits_closest_enclosing_symbol() -> None:
    overrides = {
        "DemoGateway": {"status": "ENGINEERING_SUBSTITUTE"},
    }

    result = _symbol_override("DemoGateway.generate", overrides)

    assert result == {"status": "ENGINEERING_SUBSTITUTE"}
