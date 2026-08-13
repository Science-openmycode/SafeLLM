from __future__ import annotations

import argparse
import json
from pathlib import Path

from aloepri.packaging import verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify one or more AloePri key packages")
    parser.add_argument("directories", type=Path, nargs="+")
    args = parser.parse_args()
    reports = []
    for directory in args.directories:
        failures = verify_manifest(directory) if directory.is_dir() else ["directory missing"]
        reports.append(
            {"directory": str(directory.resolve()), "pass": not failures, "failures": failures}
        )
    passed = all(report["pass"] for report in reports)
    print(json.dumps({"schema_version": 1, "pass": passed, "packages": reports}, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
