from __future__ import annotations

from pathlib import Path

from aloepri.product.redaction import redact_secrets
from aloepri.product.state import ProductJobStatus, ProductStore


def test_recursive_redaction_keeps_identifiers_but_removes_secret_values() -> None:
    payload = redact_secrets(
        {
            "key_id": "safe-id",
            "password": "secret-password",
            "nested": {"input_ids": [1, 2], "tokenizer": "safe-path"},
        }
    )
    assert payload == {
        "key_id": "safe-id",
        "password": "[REDACTED]",
        "nested": {"input_ids": "[REDACTED]", "tokenizer": "safe-path"},
    }


def test_product_events_redact_sensitive_diagnostics(tmp_path: Path) -> None:
    store = ProductStore(tmp_path / "state.db")
    store.create_job("job", {"password": "never-store", "model_id": "model"})
    assert store.get_job("job")["plan"]["password"] == "[REDACTED]"
    store.transition_job(
        "job",
        ProductJobStatus.AWAITING_CONFIRMATION,
        progress={"prompt": "private text", "item": "metadata"},
    )
    assert store.events("job")[-1]["payload"]["prompt"] == "[REDACTED]"
