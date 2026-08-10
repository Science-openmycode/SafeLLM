from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from aloepri.evidence import file_identity, verify_file_identity, verify_model_identity

RUNNER = r'''
import builtins
import json
import os
import resource
import shutil
import socket
import sys

resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))

def blocked(*args, **kwargs):
    raise PermissionError("operation disabled by HumanEval runner")

os.system = blocked
os.kill = blocked
os.remove = blocked
os.unlink = blocked
os.rmdir = blocked
os.removedirs = blocked
os.rename = blocked
os.replace = blocked
shutil.rmtree = blocked
shutil.move = blocked
socket.socket = blocked
builtins.open = blocked

payload = json.loads(sys.stdin.read())
namespace = {}
try:
    exec(payload["code"] + "\n" + payload["test"], namespace)
    print(json.dumps({"passed": True}))
except BaseException as exc:
    print(json.dumps({"passed": False, "error": type(exc).__name__ + ": " + str(exc)}))
'''


def extract_samples(payload: dict) -> list[dict]:
    samples = payload.get("samples", {}).get("humaneval_generate_local", [])
    if not samples:
        raise ValueError("no HumanEval samples found")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--wsl-distro", default="Ubuntu-24.04")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    run_provenance = payload.get("run_provenance")
    if not isinstance(run_provenance, dict):
        raise ValueError("generation artifact has no run_provenance")
    if run_provenance.get("formal_run_binding") is not True:
        raise ValueError("generation artifact is not formally bound")
    if not verify_model_identity(run_provenance.get("model", {})):
        raise ValueError("generation model identity no longer verifies")
    key = run_provenance.get("key")
    if key is not None and not verify_file_identity(key):
        raise ValueError("generation key identity no longer verifies")
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="aloepri-humaneval-") as temp_dir:
        for sample in extract_samples(payload):
            doc = sample["doc"]
            filtered = sample.get("filtered_resps") or sample.get("resps")
            code = filtered[0] if isinstance(filtered[0], str) else filtered[0][0]
            test = f'{doc["test"]}\ncheck({doc["entry_point"]})'
            try:
                runner = (
                    ["wsl.exe", "-d", args.wsl_distro, "--", "python3", "-I", "-c", RUNNER]
                    if os.name == "nt"
                    else ["python3", "-I", "-c", RUNNER]
                )
                completed = subprocess.run(
                    runner,
                    input=json.dumps({"code": code, "test": test}),
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                    timeout=args.timeout,
                    cwd=None if os.name == "nt" else temp_dir,
                    check=False,
                )
                outcome = json.loads(completed.stdout.strip().splitlines()[-1])
            except subprocess.TimeoutExpired:
                outcome = {"passed": False, "error": "TimeoutExpired"}
            except Exception as exc:
                outcome = {"passed": False, "error": f"runner error: {exc}"}
            results.append({"task_id": doc["task_id"], **outcome})
    passed = sum(bool(result["passed"]) for result in results)
    report = {
        "sample_len": len(results),
        "passed": passed,
        "pass@1": passed / len(results),
        "results": results,
        "run_provenance": run_provenance,
        "evaluation_provenance": {
            "schema_version": 1,
            "formal_run_binding": True,
            "generation_artifact": file_identity(args.input),
            "evaluation_script": file_identity(Path(__file__)),
            "runner": {
                "platform": "wsl" if os.name == "nt" else "linux",
                "wsl_distro": args.wsl_distro if os.name == "nt" else None,
                "isolated_python": True,
                "timeout_seconds": args.timeout,
                "cpu_limit_seconds": 5,
                "address_space_limit_bytes": 1024 * 1024 * 1024,
                "file_size_limit_bytes": 1024 * 1024,
                "network_disabled": True,
                "filesystem_writes_disabled": True,
            },
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
