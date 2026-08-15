from __future__ import annotations

import argparse

from aloepri.desktop.chat_api import create_chat_desktop_app
from aloepri.desktop.common import run_desktop


def main() -> None:
    parser = argparse.ArgumentParser(prog="yinbian-chat")
    parser.add_argument(
        "--deployment",
        help="启动后自动选择的健康部署 ID",
    )
    args = parser.parse_args()
    run_desktop(
        "隐变智模对话",
        lambda: create_chat_desktop_app(initial_deployment_id=args.deployment),
    )


if __name__ == "__main__":
    main()
