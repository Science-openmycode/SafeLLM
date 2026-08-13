import csv
from pathlib import Path

from scripts.summarize_cloud_runtime_cost import main


def test_runtime_cost_rounds_to_billing_hour(tmp_path: Path, monkeypatch) -> None:
    timings = tmp_path / "timings.tsv"
    with timings.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "stage",
                "started_utc",
                "finished_utc",
                "duration_seconds",
                "exit_code",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "stage": "test",
                "started_utc": "2026-08-12T00:00:00Z",
                "finished_utc": "2026-08-12T01:00:01Z",
                "duration_seconds": "3601",
                "exit_code": "0",
            }
        )
    output = tmp_path / "cost.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_cloud_runtime_cost.py",
            "--timings",
            str(timings),
            "--hourly-rate",
            "1.80",
            "--out",
            str(output),
        ],
    )
    main()
    payload = output.read_text(encoding="utf-8")
    assert '"billed_instance_hours": "2.0"' in payload
    assert '"cost": "3.60"' in payload
