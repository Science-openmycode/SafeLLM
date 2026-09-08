# 隐变智模 1.1 可复现说明

本文给出接收者从 GitHub 干净副本开始能够复现到什么程度，以及各层验证需要的额外资源。

## 1. 复现范围

| 层级 | 是否只靠 GitHub | 验收内容 |
|---|---:|---|
| 安装、静态检查、单元与模拟集成测试 | 是 | 397 条通过，1 条真实模型测试跳过 |
| Python wheel 与 CLI | 是 | 构建 sdist/wheel、运行 `yinbian --help`、导入包 |
| Qwen2.5-0.5B 私有 API 往返 | 否 | 需要固定原始模型、改造模型、在线密钥和 CUDA GPU |
| 云端一键部署 | 否 | 需要 Ubuntu GPU 服务器、SSH、足够磁盘和模型工件 |
| TEE 软件模式 | 部分 | 代码可测；完整演示需要先生成 TEE 边界包和主体包 |
| Intel TDX 硬件证明 | 否 | 尚需真实 TDX/DCAP 环境，不能由普通 RTX 服务器替代 |

“代码可复现”和“所有硬件场景均已复现”是两个结论。本仓库达到前者；真实 TDX 与未验收的大模型检查点不在当前通过范围内。

## 2. 固定版本

### 本地源码与桌面端

| 项目 | 版本/要求 |
|---|---|
| 操作系统 | Windows 10/11 x64 |
| Python | `3.11.*`；本次实测 `3.11.15` |
| uv | 本次实测 `0.12.1` |
| 项目 | `yinbian-zhimo 1.1.0` |
| PyTorch | 锁文件固定 `2.9.1+cu128` |
| Transformers | `5.12.0` |
| CUDA wheel | CUDA 12.8 构建；GPU 驱动必须满足该 wheel 要求 |

最终依赖及哈希以 `.python-version`、`pyproject.toml` 和 `uv.lock` 为准，不使用未锁定的 `pip install -U`。

### 云端原生推理环境

云端部署器使用与本地开发环境分开的运行依赖：

| 项目 | 固定值 |
|---|---|
| 操作系统 | Ubuntu 22.04 x86-64 |
| Python | 系统 `python3`；Ubuntu 22.04 的 Python 3.10 路径已兼容 |
| PyTorch | `2.5.1+cu121` |
| TorchVision | `0.20.1+cu121` |
| Transformers | `5.12.0` |
| NumPy | `2.2.6` |
| 服务监听 | 远端 `127.0.0.1`，客户端经 SSH 隧道访问 |

服务器预检结果是最终依据。部署器不安装 NVIDIA 驱动，也不把 CPU 回退标记成 GPU 部署成功。

## 3. 从 GitHub 干净复现

```powershell
git clone https://github.com/Science-openmycode/SafeLLM.git
cd SafeLLM
git checkout codex/project-organization-20260908

uv python install 3.11
uv sync --frozen --extra eval

uv run ruff check .
uv run mypy src/aloepri
uv run pytest -q
uv build
uv run yinbian --help
```

预期结果：

```text
Ruff: All checks passed
Mypy: no issues found
Pytest: 397 passed, 1 skipped
Build: yinbian_zhimo-1.1.0.tar.gz 和 yinbian_zhimo-1.1.0-py3-none-any.whl
```

跳过项 `tests/integration/test_real_private_api.py` 需要未上传 GitHub 的真实模型工件。首次冷安装会下载约 2.7 GiB 的 CUDA PyTorch wheel；建议仅 Python 环境预留至少 8 GiB 可用空间。Qwen2.5-0.5B 下载与转换另建议预留至少 12 GiB。

普通使用、不运行完整评测测试时可以执行：

```powershell
uv sync --frozen
uv run yinbian --help
```

## 4. 数据目录

```powershell
[Environment]::SetEnvironmentVariable("YINBIAN_HOME", "E:\YinbianRuntime", "User")
[Environment]::SetEnvironmentVariable("YINBIAN_DATA_DIR", "E:\YinbianData", "User")
```

模型、改造模型、密钥和证据不进入 Git。目录定义见 [代码与模型数据分离](guides/DATA_STORAGE.md)。

## 5. 固定模型来源

当前完整验收模型：

```text
模型：Qwen/Qwen2.5-0.5B-Instruct
Hugging Face commit：7ae557604adf67be50417f59c2c2f167def9a775
```

目录下载会验证 revision 必须是完整 commit，并校验 Hugging Face LFS SHA-256。模型许可证必须由复现者自行确认。

## 6. 真实模型 API 验收

准备好对应工件后：

```powershell
$env:ALOEPRI_RUN_MODEL_TESTS = "1"
$env:ALOEPRI_TEST_DEVICE = "cuda"
$env:ALOEPRI_SOURCE_MODEL = "E:\YinbianData\source-models\qwen2.5-0.5b"
$env:ALOEPRI_PRIVATE_MODEL = "E:\YinbianData\packages\qwen05b-product-v31-blockperm8"
$env:ALOEPRI_KEY_DIR = "E:\YinbianData\keys\qwen05b-product-v31-blockperm8-online"

uv run pytest tests/integration/test_real_private_api.py -q
```

该测试验证：私有模型加载、`/readyz` 单 Token 探针、普通生成、SSE 流式生成、输出 Token 恢复和错误密钥拒绝。本次整理后在 CUDA 环境实测结果为 `1 passed`。

## 7. Windows 安装包构建

安装包应在只执行过 `uv sync --frozen` 的干净基础环境中构建；不要在安装了
`eval` 等开发附加依赖的环境中制作正式安装包，否则无关评测依赖也可能进入产物。
此外还需要：

- Inno Setup 6；
- .NET Framework C# 编译器；
- `build/windows/vendor/WebView2` 中的 WebView2 Fixed Runtime x64；
- 可选的 Windows 代码签名证书。

```powershell
powershell -ExecutionPolicy Bypass -File build/windows/build_installer.ps1
```

输出文件为：

```text
dist/installer/YinbianZhimo-1.1.0-Windows-x64-Offline.exe
dist/installer/YinbianZhimo-1.1.0-Windows-x64-Offline.exe.sha256
```

无代码签名证书时只能作为内部开发安装包分发，不能宣称为正式签名发行版。
