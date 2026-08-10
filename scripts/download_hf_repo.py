from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests
from huggingface_hub import HfApi


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    session: requests.Session,
    url: str,
    target: Path,
    retries: int,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.is_file() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if offset and not append:
                    offset = 0
                mode = "ab" if append else "wb"
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            partial.replace(target)
            return
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"failed to download {target.name}") from exc
            print(
                f"retry {attempt}/{retries} {target.name} at {offset} bytes: {exc}",
                flush=True,
            )
            time.sleep(min(2**attempt, 30))


def download_file_curl(url: str, target: Path, retries: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    command = [
        "curl.exe",
        "--fail",
        "--location",
        "--continue-at",
        "-",
        "--retry",
        str(retries),
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--speed-time",
        "60",
        "--speed-limit",
        "1024",
        "--output",
        str(partial),
        url,
    ]
    subprocess.run(command, check=True)
    partial.replace(target)


def remote_size(url: str) -> int:
    response = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
    response.raise_for_status()
    content_range = response.headers.get("content-range", "")
    response.close()
    if response.status_code != 206 or "/" not in content_range:
        raise RuntimeError(f"server did not honor range probe: {response.status_code}")
    return int(content_range.rsplit("/", 1)[1])


def download_file_chunked(
    url: str,
    target: Path,
    retries: int,
    chunk_size: int,
    workers: int,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(url)
    parts = target.with_name(target.name + ".parts")
    parts.mkdir(exist_ok=True)
    chunks = []
    for index, start in enumerate(range(0, total, chunk_size)):
        end = min(start + chunk_size, total) - 1
        expected = end - start + 1
        part = parts / f"{index:06d}.part"
        if part.is_file() and part.stat().st_size == expected:
            continue
        chunks.append((index, start, end, expected, part))

    def fetch_chunk(chunk: tuple[int, int, int, int, Path]) -> int:
        index, start, end, expected, part = chunk
        temporary = part.with_name(f"{part.name}.partial.{uuid.uuid4().hex}")
        command = [
            "curl.exe",
            "--fail",
            "--location",
            "--range",
            f"{start}-{end}",
            "--retry",
            str(retries),
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--speed-time",
            "60",
            "--speed-limit",
            "1024",
            "--silent",
            "--show-error",
            "--output",
            str(temporary),
            url,
        ]
        subprocess.run(command, check=True)
        if temporary.stat().st_size != expected:
            raise RuntimeError(
                f"range length mismatch for {target.name} {start}-{end}: "
                f"expected {expected}, got {temporary.stat().st_size}"
            )
        temporary.replace(part)
        return index

    chunk_count = (total + chunk_size - 1) // chunk_size
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            index = future.result()
            print(f"chunk {index + 1}/{chunk_count} {target.name}", flush=True)
    assembled = target.with_name(target.name + ".assembling")
    with assembled.open("wb") as output:
        for part in sorted(parts.glob("*.part")):
            with part.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    output.write(block)
    if assembled.stat().st_size != total:
        raise RuntimeError(f"assembled length mismatch for {target.name}")
    assembled.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable direct Hugging Face repo downloader")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--allow", nargs="+", default=["*"])
    parser.add_argument("--retries", type=int, default=12)
    parser.add_argument("--curl", action="store_true")
    parser.add_argument("--chunk-size-mb", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    api = HfApi(endpoint=args.endpoint)
    files = api.list_repo_files(args.repo, revision=args.revision)
    files = [
        path for path in files if any(fnmatch.fnmatch(path, pattern) for pattern in args.allow)
    ]
    session = requests.Session()
    manifest = []
    for index, relative in enumerate(files, start=1):
        target = args.out / relative
        if not target.is_file() or target.stat().st_size == 0:
            encoded = "/".join(quote(part, safe="") for part in relative.split("/"))
            url = f"{args.endpoint.rstrip('/')}/{args.repo}/resolve/{args.revision}/{encoded}"
            if args.chunk_size_mb:
                download_file_chunked(
                    url,
                    target,
                    args.retries,
                    args.chunk_size_mb * 1024 * 1024,
                    args.workers,
                )
            elif args.curl:
                download_file_curl(url, target, args.retries)
            else:
                download_file(session, url, target, args.retries)
        digest = sha256_file(target)
        manifest.append({"path": relative, "bytes": target.stat().st_size, "sha256": digest})
        print(f"[{index}/{len(files)}] {relative} {target.stat().st_size}", flush=True)
    receipt = {
        "repo": args.repo,
        "revision": args.revision,
        "endpoint": args.endpoint,
        "files": manifest,
    }
    (args.out / "download_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
