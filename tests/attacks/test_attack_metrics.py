import pytest

from aloepri.attacks.metrics import corpus_bleu4


def test_corpus_bleu4_exact_match() -> None:
    assert corpus_bleu4([[1, 2, 3, 4]], [[1, 2, 3, 4]]) == pytest.approx(100.0)


def test_corpus_bleu4_rejects_mismatched_corpora() -> None:
    with pytest.raises(ValueError, match="equally sized"):
        corpus_bleu4([[1]], [])


def test_corpus_bleu4_penalizes_wrong_sequence() -> None:
    score = corpus_bleu4([[1, 2, 3, 4]], [[9, 8, 7, 6]])
    assert 0.0 < score < 100.0
