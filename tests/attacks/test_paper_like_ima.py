import json

import torch
from transformers import Qwen2Config

from aloepri.attacks.corpus import fixed_token_windows, load_corpus_texts
from aloepri.attacks.ima import (
    PaperLikeIMAInverter,
    build_paper_like_inverter_config,
    topk_embedding_recovery,
)


def test_corpus_loader_and_disjoint_windows(tmp_path) -> None:
    (tmp_path / "a.json").write_text(
        json.dumps([{"prompt": "one"}, {"text": "two"}]), encoding="utf-8"
    )
    (tmp_path / "b.txt").write_text("three", encoding="utf-8")
    assert load_corpus_texts([tmp_path / "a.json", tmp_path / "b.txt"]) == [
        "one",
        "two",
        "three",
    ]
    windows = fixed_token_windows(list(range(40)), sequence_length=4, count=5, seed=7)
    assert len(windows) == 5
    assert all(len(window) == 4 for window in windows)


def test_paper_like_ima_is_two_layer_eight_head_qwen(tmp_path) -> None:
    Qwen2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=32,
    ).save_pretrained(tmp_path)
    config = build_paper_like_inverter_config(
        tmp_path, observed_hidden_size=32, vocab_size=32
    )
    assert config.num_hidden_layers == 2
    assert config.num_attention_heads == 8
    assert config.num_key_value_heads == 8
    model = PaperLikeIMAInverter(config, target_embedding_dim=24)
    assert model(torch.randn(2, 4, 32)).shape == (2, 4, 24)


def test_topk_embedding_recovery_recovers_exact_rows() -> None:
    embedding = torch.eye(8)
    truth = torch.tensor([[2, 5, 7]])
    result = topk_embedding_recovery(embedding[truth], truth, embedding, topk=3, chunk_size=2)
    assert result["token_top1_recovery_rate"] == 1.0
    assert result["token_topk_recovery_rate"] == 1.0
    assert result["embedding_cosine_similarity"] == 1.0
