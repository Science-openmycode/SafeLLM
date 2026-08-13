from __future__ import annotations

import hashlib
import json
import os
import random
import runpy
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SEED = 20260803


def file_identity(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def output_path(argv: list[str]) -> Path:
    try:
        return Path(argv[argv.index("--out") + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError("deterministic lm-eval launcher requires --out PATH") from exc


def main() -> None:
    target = Path(__file__).with_name("run_lm_eval.py")
    if "--help" in sys.argv or "-h" in sys.argv:
        runpy.run_path(str(target), run_name="__main__")
        return

    import numpy as np
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    out = output_path(sys.argv)
    runpy.run_path(str(target), run_name="__main__")

    artifact = json.loads(out.read_text(encoding="utf-8"))
    artifact["run_provenance"]["deterministic_launcher"] = {
        "enabled": True,
        "seed": SEED,
        "torch_deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "allow_tf32": False,
        "script": file_identity(Path(__file__)),
    }
    out.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
