from __future__ import annotations

import os
import socket
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import uvicorn
from fastapi import FastAPI


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def run_desktop(
    title: str,
    factory: Callable[[], FastAPI],
    *,
    width: int = 1280,
    height: int = 820,
    initial_path: str = "/",
) -> None:
    if not initial_path.startswith("/") or initial_path.startswith("//"):
        raise ValueError("initial_path must be an absolute local application path")
    bundled_webview = Path(sys.executable).resolve().parent / "WebView2"
    if bundled_webview.is_dir():
        executable = next(bundled_webview.rglob("msedgewebview2.exe"), None)
        runtime_root = bundled_webview if executable is None else executable.parent
        os.environ.setdefault(
            "WEBVIEW2_BROWSER_EXECUTABLE_FOLDER", str(runtime_root)
        )
    import webview

    port = _free_port()
    config = uvicorn.Config(factory(), host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() >= deadline:
            raise TimeoutError("local desktop service did not start")
        time.sleep(0.05)
    try:
        webview.create_window(
            title,
            f"http://127.0.0.1:{port}{initial_path}",
            width=width,
            height=height,
            min_size=(1024, 700),
        )
        webview.start(debug=False, private_mode=False)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
