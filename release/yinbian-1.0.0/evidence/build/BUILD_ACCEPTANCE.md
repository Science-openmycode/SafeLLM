# 隐变智模 1.0 Windows 开发构建证据

构建日期：2026-08-14（Asia/Shanghai）

产品代码提交：`708a516a6e862e3baaad81a38d59b0cfe980acd4`

| 项目 | 结果 |
|---|---|
| 产品版本 | `1.0.0` |
| PyInstaller | `6.22.0` |
| Python | `3.11.15` |
| CLI | `yinbian.exe --version` 返回 `1.0.0`；`--help` 退出码为 0 |
| 部署桌面程序 | 冻结 EXE 启动 35 秒后仍存活，测试进程随后关闭 |
| 对话桌面程序 | 冻结 EXE 启动 35 秒后仍存活，测试进程随后关闭 |
| 安装包 | `YinbianZhimo-1.0.0-Windows-x64-Offline.exe` |
| 字节数 | `3323353793` |
| SHA-256 | `2e96741b398197967ab655ae50c782451901a826254906f3ea64bdbf5289bf1f` |
| Authenticode | `NotSigned` |
| 冻结目录文件数 | `7567` |
| 稳定启动器 | CLI 启动器读取 `current.json` 后返回 `1.0.0`、退出码 0；GUI/CLI 快捷入口均指向 `launcher` 目录 |
| 模型/密钥扫描 | 0 个 `.safetensors`、`.ybkey`、`.dpapi`、`.aloepri-key` |
| 自动化测试 | 277 通过，1 跳过；JUnit 见 `../tests/pytest.xml` |

该构建只能用于内部开发验收。以下门禁没有在本机完成：

- 干净 Windows 10 虚拟机离线安装、更新、回退和卸载；
- 干净 Windows 11 虚拟机离线安装、更新、回退和卸载；
- 受控 Ubuntu 22.04 GPU 服务器真实上传、启动、问答和回滚；
- GHCR 运行镜像发布并固定到 image digest；
- Windows 代码签名和时间戳。
