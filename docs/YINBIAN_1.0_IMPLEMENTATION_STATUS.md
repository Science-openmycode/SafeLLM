# 隐变智模 1.0 实施状态

## 2026-08-15 可移植性与多架构修复

| 检查项 | 当前结果 |
|---|---|
| Ruff | 通过 |
| Mypy | 120 个源文件通过 |
| Pytest | 310 通过，1 跳过；跳过项需要显式加载真实 Qwen2.5-0.5B |
| Qwen2.5-0.5B 云端 | 一台 24 GiB GPU 租赁容器真实部署并通过健康检查 |
| 新机器流程 | 预检驱动、显存、内存、磁盘、身份和Python；统一原生运行时；依赖精确锁定 |
| 模型迁移 | 可把本地私有包复制到另一台服务器，不重新转换 |
| 架构目录 | Qwen2/2.5、DeepSeek、GLM、Qwen3、Kimi 按族展示 |

当前可执行权重转换的是 Qwen2/2.5、DeepSeek-V2 和 DeepSeek-V3 适配器路径。Qwen2.5-0.5B 是唯一完成产品部署验收的型号；OpenSeek完成真实转换验收。GLM、Qwen3和Kimi已经完成结构级识别，但由于融合FFN、Q/K Norm、多模态外壳和打包INT4/FP8格式差异，不标记为可转换。

## 2026-08-14 本机构建结果

| 检查项 | 当前结果 |
|---|---|
| Ruff | 通过 |
| Mypy | 119 个源文件通过 |
| Pytest | 277 通过，1 跳过；跳过项需要显式加载真实 Qwen2.5-0.5B |
| 冻结入口 | `yinbian.exe --version` 为 `1.0.0`；两个桌面入口启动 35 秒后均存活 |
| 开发安装包 | `dist/installer/YinbianZhimo-1.0.0-Windows-x64-Offline.exe` |
| 安装包大小 | 3,323,353,793 字节（3169.40 MiB） |
| 安装包 SHA-256 | `2e96741b398197967ab655ae50c782451901a826254906f3ea64bdbf5289bf1f` |
| 签名 | `NotSigned`，只能用于内部开发验收 |
| 包内容扫描 | 7,567 个运行时文件；0 个 `.safetensors`、`.ybkey`、`.dpapi` 或 AloePri 密钥包；另含 2 个无依赖稳定启动器 |

当前代码门禁通过，但正式 1.0 发布门禁尚未通过。还缺干净 Windows 10/11 安装、受控 Ubuntu 22.04 GPU 的真实部署/回滚、固定 digest 的已发布 HF 运行镜像，以及 Windows 代码签名。

## 已实现代码

| 部分 | 入口或目录 | 当前状态 |
|---|---|---|
| 品牌与兼容 CLI | `src/aloepri/cli.py`、`src/aloepri/product_cli.py` | `yinbian` 与旧命令共用核心代码 |
| 状态库 | `src/aloepri/product/state.py` | WAL、外键、迁移、任务/分片/部署分离 |
| 模型目录和下载 | `src/aloepri/catalog/` | 固定 commit、索引优先、断点下载、SHA-256 |
| 转换任务 | `src/aloepri/product/pipeline.py` | Qwen2.5-0.5B 正式路径；DeepSeek 使用已有 tile 执行器 |
| 密钥 | `src/aloepri/keys/` | 在线密钥目录使用 DPAPI 封装；Scrypt、AES-256-GCM 便携备份；旧格式导入 |
| SSH 部署 | `src/aloepri/cloud/ssh.py`、`hf_deployment.py` | 预检、上传、Compose、systemd、健康、回滚 |
| 本地隧道 | `src/aloepri/cloud/tunnel.py`、`product/tunnel_worker.py` | 桌面进程和 CLI 后台进程均可用 |
| 部署桌面程序 | `src/aloepri/desktop/deploy*` | 模型、许可证、任务、服务器、密钥备份和直接部署控制 |
| 对话桌面程序 | `src/aloepri/desktop/chat*` | 健康部署、多轮历史、SSE、本地 Token 恢复 |
| Windows 构建 | `build/windows/` | PyInstaller 与 Inno Setup 配置 |
| 服务镜像 | `deploy/runtime/Dockerfile` | HF 私有 Token-ID 服务入口 |

## 发布前必须补齐的物理验收

| 门禁 | 需要的环境 | 证据 |
|---|---|---|
| Windows 10 安装、更新、卸载 | 干净 Windows 10 x64 虚拟机 | 安装日志、截图、SHA-256 |
| Windows 11 安装、更新、卸载 | 干净 Windows 11 x64 虚拟机 | 安装日志、截图、SHA-256 |
| Ubuntu GPU 部署 | Ubuntu 22.04、NVIDIA 525+ | 预检、上传哈希、健康、回滚日志 |
| Qwen2.5-0.5B 端到端 | Windows 客户端和受控 GPU 服务器 | 转换、隧道、多轮问答、安全扫描 |
| 代码签名 | Windows 代码签名证书 | 签名验证和时间戳 |

完成上述证据前，不发布正式签名安装包。

## 本地检查命令

```powershell
uv sync --frozen
uv run ruff check src scripts tests
uv run mypy src
uv run pytest -q

uv run yinbian models list
uv run yinbian keys list
uv run yinbian servers list
uv run yinbian deploy list
uv run yinbian chat deployments
```

Windows 开发构建：

```powershell
.\build\windows\build_installer.ps1
```

该脚本要求 `build\windows\vendor\WebView2` 中已有固定版 WebView2 Runtime，并要求本机安装 Inno Setup 6。未配置签名变量时生成的安装包只供内部测试。
