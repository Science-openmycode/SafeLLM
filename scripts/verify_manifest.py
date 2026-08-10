from __future__ import annotations

import argparse
import json
from pathlib import Path

from aloepri.conversion.verify import verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    result = verify_manifest(args.model_dir)
    print(json.dumps({"checked": result.checked, "failures": result.failures, "ok": result.ok}))
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
