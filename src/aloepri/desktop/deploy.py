from __future__ import annotations

from aloepri.desktop.common import run_desktop
from aloepri.desktop.deploy_api import create_deploy_desktop_app


def main() -> None:
    run_desktop("隐变智模部署", create_deploy_desktop_app)


if __name__ == "__main__":
    main()
