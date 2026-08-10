from __future__ import annotations

import logging

import pytest

from aloepri.serving.audit_log import audit_event


def test_audit_log_contains_metadata_only(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="test.audit"):
        audit_event(logging.getLogger("test.audit"), "done", request_id="r1", input_tokens=3)
    assert '"request_id":"r1"' in caplog.text
    assert '"input_tokens":3' in caplog.text


@pytest.mark.parametrize("field", ["input_ids", "output_ids", "prompt", "text", "tau"])
def test_audit_log_rejects_sensitive_fields(field: str) -> None:
    with pytest.raises(ValueError, match="sensitive audit fields"):
        audit_event(logging.getLogger("test.audit"), "bad", **{field: [1, 2, 3]})
