from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    executable = Path(sys.executable).stem.lower()
    if "deploy" in executable or "部署" in executable:
        from aloepri.desktop.deploy import main as deploy_main

        deploy_main()
        return
    if "chat" in executable or "对话" in executable:
        from aloepri.desktop.chat import main as chat_main

        chat_main()
        return
    from aloepri.cli import app

    app(prog_name="yinbian")


if __name__ == "__main__":
    main()
