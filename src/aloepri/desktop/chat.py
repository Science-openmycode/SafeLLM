from __future__ import annotations

from aloepri.desktop.chat_api import create_chat_desktop_app
from aloepri.desktop.common import run_desktop


def main() -> None:
    run_desktop("隐变智模对话", create_chat_desktop_app)


if __name__ == "__main__":
    main()
