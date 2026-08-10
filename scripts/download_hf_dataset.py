from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import requests
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable Hugging Face dataset downloader")
    parser.add_argument("repo_id")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--mmlu-configs",
        type=Path,
        help="Build the exact 57-subject MMLU file list without querying the Hub tree API.",
    )
    parser.add_argument("--subject-configs", type=Path)
    parser.add_argument("--config-glob", default="*.yaml")
    parser.add_argument("--splits", nargs="+", default=["dev", "test", "validation"])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    config_dir = args.subject_configs or args.mmlu_configs
    if config_dir:
        config_glob = "mmlu_*.yaml" if args.mmlu_configs else args.config_glob
        subjects = sorted(
            str(yaml.safe_load(path.read_text(encoding="utf-8"))["dataset_name"])
            for path in config_dir.glob(config_glob)
            if "dataset_name" in yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        files = [
            f"{subject}/{split}-00000-of-00001.parquet"
            for subject in subjects
            for split in args.splits
        ]
    else:
        api_url = f"{args.endpoint.rstrip('/')}/api/datasets/{args.repo_id}/tree/main"
        response = requests.get(
            api_url,
            params={"recursive": "true", "expand": "false"},
            timeout=60,
        )
        response.raise_for_status()
        files = [entry["path"] for entry in response.json() if entry.get("type") == "file"]
    session = requests.Session()
    completed: list[dict[str, object]] = []
    for index, relative in enumerate(files, start=1):
        target = args.out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size > 0:
            print(f"[{index}/{len(files)}] cached {relative}", flush=True)
        else:
            partial = target.with_name(target.name + ".partial")
            url = f"{args.endpoint.rstrip('/')}/datasets/{args.repo_id}/resolve/main/{relative}"
            for attempt in range(1, args.retries + 1):
                try:
                    with session.get(url, stream=True, timeout=(30, args.timeout)) as response:
                        response.raise_for_status()
                        with partial.open("wb") as handle:
                            for chunk in response.iter_content(1024 * 1024):
                                if chunk:
                                    handle.write(chunk)
                    if partial.stat().st_size == 0:
                        raise RuntimeError("downloaded file is empty")
                    partial.replace(target)
                    print(f"[{index}/{len(files)}] downloaded {relative}", flush=True)
                    break
                except Exception as exc:
                    if attempt == args.retries:
                        raise RuntimeError(
                            f"failed after {args.retries} attempts: {relative}"
                        ) from exc
                    print(f"[{index}/{len(files)}] retry {attempt}: {relative}: {exc}", flush=True)
                    time.sleep(min(2**attempt, 30))
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        completed.append({"path": relative, "bytes": target.stat().st_size, "sha256": digest})

    manifest = {
        "repo_id": args.repo_id,
        "endpoint": args.endpoint,
        "file_count": len(completed),
        "files": completed,
    }
    (args.out / "download-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"file_count": len(completed), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
