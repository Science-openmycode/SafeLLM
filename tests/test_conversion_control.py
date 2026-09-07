from __future__ import annotations

import json
from pathlib import Path

import pytest

from aloepri.conversion import executor
from aloepri.jobs.store import JobState, JobStore
from aloepri.planning import ConversionPlan


def _deepseek_config(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 16,
                "vocab_size": 32,
                "num_attention_heads": 2,
                "qk_nope_head_dim": 4,
                "qk_rope_head_dim": 4,
                "v_head_dim": 4,
                "kv_lora_rank": 8,
                "q_lora_rank": None,
                "moe_intermediate_size": 8,
                "n_routed_experts": 2,
            }
        ),
        encoding="utf-8",
    )


def test_deepseek_progress_callback_stops_at_a_paused_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _deepseek_config(source)
    plan = ConversionPlan(
        schema_version=1,
        job_id="pause-job",
        source={"type": "local", "path": str(source)},
        adapter="deepseek_v2",
        fingerprint={"weight_format": "bfloat16"},
        output={"type": "local", "uri": str(tmp_path / "private")},
        security={"vocab_permutation": True, "offline_key_encrypted": False},
    )
    store = JobStore(tmp_path / "state.db")
    store.create(plan.job_id, plan.to_dict())
    store.transition(plan.job_id, JobState.PREFLIGHT)
    store.transition(plan.job_id, JobState.CONVERTING)

    def fake_converter(**kwargs: object) -> dict[str, bool]:
        callback = kwargs["progress_callback"]
        callback("layer.0.weight", 0, "completed")  # type: ignore[operator]
        store.transition(plan.job_id, JobState.PAUSED)
        callback("layer.0.weight", 1, "starting")  # type: ignore[operator]
        return {"pass": True}

    monkeypatch.setattr(executor, "convert_deepseek_checkpoint", fake_converter)
    with pytest.raises(executor.ConversionPaused):
        executor._run_deepseek(plan, tmp_path / "private", store, source)  # type: ignore[attr-defined]
    assert store.get(plan.job_id)["state"] == "PAUSED"
    assert store.completed_tiles(plan.job_id, "layer.0.weight")[0]


def test_qwen_executor_uses_cli_enum_spellings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    plan = ConversionPlan(
        schema_version=1,
        job_id="qwen-args",
        source={"type": "local", "path": str(source)},
        adapter="qwen2",
        fingerprint={"weight_format": "float32"},
        output={"type": "local", "uri": str(tmp_path / "private")},
        security={"vocab_permutation": True, "offline_key_encrypted": False},
    )
    captured: list[str] = []
    captured_environment: dict[str, str] = {}

    def fake_run(command: list[str], **kwargs: object) -> None:
        captured.extend(command)
        captured_environment.update(kwargs["env"])  # type: ignore[arg-type]
        Path(command[command.index("--key-dir") + 1]).mkdir(parents=True)

    monkeypatch.setattr(executor.subprocess, "run", fake_run)
    monkeypatch.setattr(executor, "split_key_package", lambda *_: None)
    executor._run_qwen(plan, tmp_path / "private", source)  # type: ignore[attr-defined]
    assert captured[captured.index("--rms-mode") + 1] == "exact-metric"
    assert captured[captured.index("--rms-representation") + 1] == "stable-factor"
    assert captured_environment["PYTHONFAULTHANDLER"] == "1"
    assert captured_environment["OMP_NUM_THREADS"] == "4"
    assert "pythonw.exe" not in captured[0].casefold()


def test_qwen_executor_removes_only_empty_stale_partial_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "private"
    output_partial = tmp_path / "private.partial"
    key_partial = tmp_path / "private-keys" / "full.partial"
    output_partial.mkdir()
    key_partial.mkdir(parents=True)
    plan = ConversionPlan(
        schema_version=1,
        job_id="qwen-empty-partials",
        source={"type": "local", "path": str(source)},
        adapter="qwen2",
        fingerprint={"weight_format": "bfloat16"},
        output={"type": "local", "uri": str(output)},
        security={"vocab_permutation": True, "offline_key_encrypted": False},
    )
    def fake_run(command: list[str], **_kwargs: object) -> None:
        Path(command[command.index("--key-dir") + 1]).mkdir(parents=True)

    monkeypatch.setattr(executor.subprocess, "run", fake_run)
    monkeypatch.setattr(executor, "split_key_package", lambda *_args: None)
    executor._run_qwen(plan, output, source)  # type: ignore[attr-defined]
    assert not output_partial.exists()
    assert not key_partial.exists()


def test_qwen_executor_refuses_nonempty_stale_partial_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "private"
    output_partial = tmp_path / "private.partial"
    output_partial.mkdir()
    (output_partial / "unfinished.bin").write_bytes(b"partial")
    plan = ConversionPlan(
        schema_version=1,
        job_id="qwen-nonempty-partial",
        source={"type": "local", "path": str(source)},
        adapter="qwen2",
        fingerprint={"weight_format": "bfloat16"},
        output={"type": "local", "uri": str(output)},
        security={"vocab_permutation": True, "offline_key_encrypted": False},
    )
    with pytest.raises(FileExistsError, match="requires inspection"):
        executor._run_qwen(plan, output, source)  # type: ignore[attr-defined]


def test_qwen_executor_routes_tee_plan_without_online_token_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "private"
    tee_boundary = tmp_path / "trusted-boundary"
    plan = ConversionPlan(
        schema_version=1,
        job_id="qwen-tee",
        source={"type": "local", "path": str(source)},
        adapter="qwen2",
        fingerprint={"weight_format": "float32"},
        output={"type": "local", "uri": str(output)},
        keys={"tee": str(tee_boundary)},
        security={
            "vocab_permutation": False,
            "offline_key_encrypted": False,
            "security_mode": "tee_gm",
            "boundary_mode": "tee_split",
            "tee_backend": "software_sim",
        },
    )
    captured: list[str] = []

    def fake_run(command: list[str], **_kwargs: object) -> None:
        captured.extend(command)
        key_dir = Path(command[command.index("--key-dir") + 1])
        key_dir.mkdir(parents=True)
        (key_dir / "paper_key.safetensors").write_bytes(b"key")
        (key_dir / "key.json").write_text("{}", encoding="utf-8")
        Path(command[command.index("--tee-output") + 1]).mkdir(parents=True)
        Path(command[command.index("--output") + 1]).mkdir(parents=True)

    monkeypatch.setattr(executor.subprocess, "run", fake_run)
    result = executor._run_qwen(plan, output, source)  # type: ignore[attr-defined]
    assert captured[captured.index("--security-mode") + 1] == "tee-gm"
    assert captured[captured.index("--boundary-mode") + 1] == "tee-split"
    assert result["online_key"] is None
    assert result["tee_boundary"] == str(tee_boundary.resolve())
