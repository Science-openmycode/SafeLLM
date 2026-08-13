from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import time
from pathlib import Path
from typing import Any

try:
    import paramiko
except ImportError as error:  # pragma: no cover - local operator dependency
    raise SystemExit("install the local upload dependency: uv pip install paramiko") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload and verify the Qwen0.5B cloud bundle")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", default="root")
    parser.add_argument("--bundle-dir", type=Path, default=Path("release/qwen05b-cloud"))
    parser.add_argument("--remote-root", default="/data")
    parser.add_argument("--remote-upload", default="/root/aloepri-upload")
    args = parser.parse_args()
    password_b64 = os.environ.pop("ALOEPRI_REMOTE_PASS_B64", None)
    if not password_b64:
        parser.error("ALOEPRI_REMOTE_PASS_B64 is required")
    password = base64.b64decode(password_b64).decode("utf-8")
    bundle_path = args.bundle_dir / "bundle-set.json"
    bundle: dict[str, Any] = json.loads(bundle_path.read_text(encoding="utf-8"))

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=password,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )

    def run(command: str, *, timeout: int = 1800) -> str:
        _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        code = stdout.channel.recv_exit_status()
        output = stdout.read().decode(errors="replace")
        error_output = stderr.read().decode(errors="replace")
        if code:
            raise RuntimeError(
                f"remote command failed ({code}): {command}\n{output}\n{error_output}"
            )
        return output.strip()

    root = shlex.quote(args.remote_root)
    upload = shlex.quote(args.remote_upload)
    run(f"mkdir -p {root} {upload} && chmod 700 {upload}")
    sftp = client.open_sftp()
    remote_bundle = f"{args.remote_upload}/bundle-set.json"
    sftp.put(str(bundle_path), remote_bundle)
    for record in bundle["archives"]:
        local = args.bundle_dir / Path(record["path"]).name
        remote = f"{args.remote_upload}/{local.name}"
        total = local.stat().st_size
        last_bucket = -1

        def progress(
            done: int,
            _total: int,
            *,
            upload_total: int = total,
            upload_name: str = local.name,
        ) -> None:
            nonlocal last_bucket
            bucket = int(done * 10 / max(upload_total, 1))
            if bucket != last_bucket:
                last_bucket = bucket
                print(
                    f"UPLOAD {upload_name}: {done / 1024**3:.2f}/"
                    f"{upload_total / 1024**3:.2f} GiB "
                    f"({done * 100 / upload_total:.0f}%)",
                    flush=True,
                )

        started = time.monotonic()
        sftp.put(str(local), remote, callback=progress)
        quoted_remote = shlex.quote(remote)
        actual = run(f"sha256sum {quoted_remote} | cut -d' ' -f1")
        if actual != record["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {local.name}: {actual}")
        elapsed = time.monotonic() - started
        print(f"VERIFIED {local.name} in {elapsed:.1f}s sha256={actual}", flush=True)
        run(f"tar -xzf {quoted_remote} -C {root}", timeout=3600)
        run(f"rm -f -- {quoted_remote}")
        free = run("df -BG --output=avail / | tail -1 | tr -d ' '")
        print(f"EXTRACTED {local.name}; free={free}", flush=True)
    sftp.close()
    aloepri_root = f"{args.remote_root}/AloePri"
    run(f"find {shlex.quote(aloepri_root + '/data/keys')} -type f -exec chmod 600 {{}} \\;")
    verification = f"""import json
from pathlib import Path
from scripts.verify_qwen05b_cloud_bundle_set import verify_extracted_manifests
payload=json.loads(Path({remote_bundle!r}).read_text(encoding='utf-8'))
failures=verify_extracted_manifests(payload,Path({aloepri_root!r}))
print(json.dumps({{'pass':not failures,'failures':failures}},indent=2))
raise SystemExit(0 if not failures else 2)
"""
    encoded = base64.b64encode(verification.encode()).decode()
    result = run(
        f"cd {shlex.quote(aloepri_root)} && echo {shlex.quote(encoded)} | base64 -d | python3 -",
        timeout=600,
    )
    print(f"POST_EXTRACT_VERIFY {result}", flush=True)
    print(f"REMOTE_DISK\n{run('df -h / ' + root)}", flush=True)
    client.close()


if __name__ == "__main__":
    main()
