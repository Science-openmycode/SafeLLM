# 开发约定

## 从哪里开始

先阅读 [代码部件地图](docs/architecture/COMPONENT_MAP.md)，再根据任务进入对应目录。模型数据位置见
[代码与模型数据分离](docs/guides/DATA_STORAGE.md)。

## 命名规则

- 模块名描述职责，不使用 `utils.py`、`common.py`、`misc.py` 这类泛化名称。
- 类名描述承担者，例如 `TeeAttestationService`、`MaskedOutsourceHeadEngine`。
- 函数名描述动作，例如 `verify_manifest_files`、`inspect_tdx_environment`。
- `private` 表示 P/Q 或 Token 坐标下的数据；`plain` 表示恢复后的模型语义坐标。
- `source_model` 表示原始模型，`private_model` 表示改造结果，二者不得混用。
- 兼容模块只允许转发导入，不再承载新实现。

## 修改要求

1. 不提交模型、密钥、数据库、日志或生成证据。
2. 不修改已有工件格式，除非同时提供显式版本迁移。
3. 修改公共导入路径时保留兼容模块和回归测试。
4. 提交前运行：

```powershell
python -m pytest
python -m ruff check .
python -m mypy src/aloepri
```

真实模型、GPU、云服务器和 TDX 测试必须单独标注环境；软件模拟结果不得写成硬件证明结果。
