from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from aloepri.cli import app
from aloepri.conversion import executor
from aloepri.jobs.store import JobStore
from aloepri.planning import ConversionPlan


def _tiny_qwen(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "num_hidden_layers": 0,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "torch_dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(4, 2),
            "model.norm.weight": torch.ones(2),
            "lm_head.weight": torch.zeros(4, 2),
        },
        path / "model.safetensors",
    )


def test_product_cli_plan_job_upload_and_mock_deploy(tmp_path: Path) -> None:
    runner = CliRunner()
    model = tmp_path / "model"
    _tiny_qwen(model)
    plan_path = tmp_path / "plan.yaml"
    environment = {"ALOEPRI_STATE_DB": str(tmp_path / "state.db")}
    result = runner.invoke(app, ["models", "inspect", "--model", str(model)], env=environment)
    assert result.exit_code == 0, result.output
    assert '"adapter_id": "qwen2"' in result.output
    result = runner.invoke(
        app,
        [
            "plan",
            "--model",
            str(model),
            "--output",
            str(plan_path),
            "--destination",
            str(tmp_path / "private"),
        ],
        env=environment,
    )
    assert result.exit_code == 0, result.output
    plan = ConversionPlan.load(plan_path)
    result = runner.invoke(
        app, ["convert", "--plan", str(plan_path), "--schedule-only"], env=environment
    )
    assert result.exit_code == 0, result.output
    assert '"state": "CONVERTING"' in result.output
    result = runner.invoke(
        app, ["upload", plan.job_id, "--cloud-profile", "mock"], env=environment
    )
    assert result.exit_code == 0, result.output
    assert '"environment": "mock-cloud"' in result.output
    result = runner.invoke(
        app, ["deploy", plan.job_id, "--cloud-profile", "mock"], env=environment
    )
    assert result.exit_code == 0, result.output
    assert '"real_cloud_validated": false' in result.output


def test_convert_requires_exactly_one_input() -> None:
    result = CliRunner().invoke(app, ["convert"])
    assert result.exit_code != 0
    assert "one of --plan or --config is required" in result.output


def test_completed_local_conversion_waits_for_explicit_upload(
    tmp_path: Path, monkeypatch: object
) -> None:
    plan = ConversionPlan(
        schema_version=1,
        job_id="job-execute",
        source={"type": "local", "path": str(tmp_path / "source")},
        adapter="qwen2",
        fingerprint={"weight_format": "float32"},
        output={"type": "local", "uri": str(tmp_path / "private")},
    )

    def fake_run(current: ConversionPlan, output: Path) -> dict[str, str]:
        assert current is plan
        return {"output": str(output)}

    monkeypatch.setattr(executor, "_run_qwen", fake_run)  # type: ignore[attr-defined]
    store = JobStore(tmp_path / "state.db")
    executor.execute_conversion_plan(plan, store)
    assert store.get(plan.job_id)["state"] == "UPLOADING"
