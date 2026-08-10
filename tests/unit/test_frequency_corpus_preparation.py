from __future__ import annotations

from scripts.prepare_frequency_attack_corpora import (
    deterministic_partition,
    record_to_text,
    stream_processed_meddialog,
)


def test_record_to_text_flattens_dialogue_fields() -> None:
    record = {"question": "问", "answer": "答", "ignored": "不纳入"}
    assert record_to_text(record) == "问\n答"


def test_deterministic_partition_is_stable() -> None:
    assert deterministic_partition("固定文本") == deterministic_partition("固定文本")


def test_processed_meddialog_stream_parser(tmp_path) -> None:
    path = tmp_path / "processed.zh.json"
    path.write_text(
        '[\n  ["病人：甲", "医生：乙"],\n  ["病人：丙", "医生：丁"]\n]\n',
        encoding="utf-8",
    )
    assert list(stream_processed_meddialog(path)) == [
        {"utterances": ["病人：甲", "医生：乙"]},
        {"utterances": ["病人：丙", "医生：丁"]},
    ]
