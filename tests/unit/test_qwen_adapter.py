from __future__ import annotations

from aloepri.adapters.base import TensorInventory
from aloepri.adapters.families import Qwen2FamilyAdapter


def test_tied_qwen_checkpoint_does_not_require_duplicate_lm_head() -> None:
    inventory = TensorInventory(
        frozenset({"model.embed_tokens.weight", "model.norm.weight"}),
        {},
        {},
    )
    report = Qwen2FamilyAdapter().validate_inventory(
        {"num_hidden_layers": 0, "tie_word_embeddings": True}, inventory
    )
    assert report.pass_ is True


def test_untied_qwen_checkpoint_requires_lm_head() -> None:
    inventory = TensorInventory(
        frozenset({"model.embed_tokens.weight", "model.norm.weight"}),
        {},
        {},
    )
    report = Qwen2FamilyAdapter().validate_inventory(
        {"num_hidden_layers": 0, "tie_word_embeddings": False}, inventory
    )
    assert report.missing == ("lm_head.weight",)
