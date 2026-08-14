# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

root = Path(SPEC).resolve().parents[2]
source = root / "src"
entry = source / "aloepri" / "desktop" / "launcher.py"
datas = [
    (str(source / "aloepri" / "demo" / "static"), "aloepri/demo/static"),
    (str(source / "aloepri" / "desktop" / "static"), "aloepri/desktop/static"),
    (str(root / "configs" / "catalog"), "configs/catalog"),
]

a = Analysis(
    [str(entry)],
    pathex=[str(source), str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "aloepri.models.configuration_aloepri_qwen2",
        "aloepri.models.modeling_aloepri_qwen2",
        "aloepri.serving.container_entrypoint",
        "webview.platforms.edgechromium",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)
deploy = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="隐变智模部署", debug=False, bootloader_ignore_signals=False,
    strip=False, upx=True, console=False, disable_windowed_traceback=False,
)
chat = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="隐变智模对话", debug=False, bootloader_ignore_signals=False,
    strip=False, upx=True, console=False, disable_windowed_traceback=False,
)
cli = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="yinbian", debug=False, bootloader_ignore_signals=False,
    strip=False, upx=True, console=True, disable_windowed_traceback=False,
)
coll = COLLECT(
    deploy, chat, cli, a.binaries, a.datas,
    strip=False, upx=True, upx_exclude=[], name="YinbianZhimo",
)
