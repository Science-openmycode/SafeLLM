from scripts.run_product_question_gate import garbled, repetitive


def test_garbled_detects_replacement_and_private_use_characters() -> None:
    assert garbled("normal answer") is False
    assert garbled("broken \ufffd answer") is True
    assert garbled("private \ue000 glyph") is True


def test_repetitive_detects_token_and_phrase_loops() -> None:
    assert repetitive(list(range(20))) is False
    assert repetitive([7] * 16) is True
    assert repetitive([1, 2, 3, 4] * 5) is True


def test_repetitive_does_not_reject_short_answers() -> None:
    assert repetitive([42] * 8) is False
