from pathlib import Path

import pytest

from scripts.relocate_local_eval_tasks import relocate_task_config


def test_relocate_task_config_rewrites_windows_path(tmp_path: Path) -> None:
    dataset_root = tmp_path / "mmlu"
    subject = dataset_root / "abstract_algebra"
    subject.mkdir(parents=True)
    parquet = subject / "test-00000-of-00001.parquet"
    parquet.write_bytes(b"parquet")
    payload = {
        "task": "mmlu_local_abstract_algebra",
        "dataset_kwargs": {
            "data_files": {
                "test": r"E:\AloePri\data\eval\mmlu\abstract_algebra\test-00000-of-00001.parquet"
            }
        },
    }

    relocated = relocate_task_config(payload, family="mmlu", dataset_root=dataset_root)

    assert relocated["dataset_kwargs"]["data_files"]["test"] == str(parquet.resolve())


def test_relocate_task_config_rejects_missing_file(tmp_path: Path) -> None:
    payload = {
        "task": "ceval_local_accountant",
        "dataset_kwargs": {"data_files": {"validation": "val.parquet"}},
    }
    with pytest.raises(FileNotFoundError):
        relocate_task_config(payload, family="ceval", dataset_root=tmp_path)
