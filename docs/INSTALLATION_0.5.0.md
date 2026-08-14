# AloePri 0.5.0安装手册

## 环境

- Windows 10/11或Linux；
- Python 3.11；
- `uv`；
- CUDA机器建议驱动支持当前PyTorch CUDA 12.8构建；
- 本地转换最低目标为RTX 3060 6GB、11GiB可用系统内存和50GiB临时空间；
- DeepSeek-V3完整权重目录约689GB，50GiB仅指转换过程额外临时空间，不含源模型、目标模型和对象存储容量。

## 安装

```powershell
cd E:\AloePri
uv sync --frozen
uv run aloepri --help
```

若正在运行旧AloePri服务，Windows可能锁定`.venv\Scripts\aloepri.exe`。不需要停止服务时可直接执行：

```powershell
.\.venv\Scripts\python.exe -m aloepri.cli --help
```

## 状态数据库

默认位置：

```text
Windows: %LOCALAPPDATA%\AloePri\state.db
Linux:   ~/.local/share/aloepri/state.db
```

测试或多实例隔离：

```powershell
$env:ALOEPRI_STATE_DB = "E:\AloePri\data\state\test.db"
```

数据库不保存明文prompt、对话历史、`tau`、`inverse_tau`或离线密钥。它保存plan、任务状态、tile记录、错误和deployment版本。

## 安装验收

```powershell
uv run ruff check src scripts tests
uv run mypy src
uv run pytest -q
```

0.5.0当前发布基线：`257 collected`，`256 passed, 1 skipped`。跳过项需要`ALOEPRI_RUN_MODEL_TESTS=1`才启动；当前Qwen和OpenSeek工件另有独立真实checkpoint前向证据，路径见`docs/ACCEPTANCE_0.5.0.md`。
