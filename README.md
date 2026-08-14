# AloePri 0.5.0

AloePri把“模型识别、权重私有化改造、密钥拆分、上传部署和本地问答”统一为一个可恢复任务流。0.5.0支持Qwen2/Qwen2.5、DeepSeek-V2/V2.5和DeepSeek-V3架构族；云端部分使用接口级Mock，不代表真实云集群或671B物理执行。

## 当前状态

| 项目 | 状态 |
|---|---|
| 版本 | `0.5.0` |
| 发布状态 | `CODE_COMPLETE_MOCK_CLOUD_PASS` |
| 自动化测试 | `257 collected`：`256 passed, 1 skipped`；另有当前Qwen/OpenSeek真实checkpoint前向证据 |
| Ruff / Mypy | PASS |
| 官方DeepSeek-V3索引 | 91,991张量；missing=0；unknown=0 |
| 真实Qwen2.5-0.5B | 转换完成；私有next-token经`inverse_tau`恢复为108386，与明文一致 |
| OpenSeek-Small-v1-SFT | MLA/MoE转换完成；私有next-token 123415恢复为9707，与明文一致 |
| 真实云平台 | `NOT_TESTED` |
| 真实DeepSeek-V3 671B转换 | `NOT_EXECUTED` |

## 安装与检查

```powershell
uv sync --frozen
uv run aloepri models list
uv run aloepri models recommend
uv run aloepri models inspect --model E:\models\Qwen2.5-0.5B-Instruct
uv run aloepri doctor --model E:\models\Qwen2.5-0.5B-Instruct
```

模型检查只读取`config.json`、Safetensors索引和头部，不执行模型仓库的远程代码。

## 本地计划与Mock Cloud闭环

```powershell
$env:ALOEPRI_OFFLINE_KEY_PASSWORD = "使用密码管理器生成的强密码"

uv run aloepri plan `
  --model E:\models\Qwen2.5-0.5B-Instruct `
  --output artifacts\plans\qwen05b.yaml `
  --destination artifacts\converted\qwen05b-private

uv run aloepri convert --plan artifacts\plans\qwen05b.yaml
uv run aloepri upload <job-id> --cloud-profile mock
uv run aloepri deploy <job-id> --cloud-profile mock
uv run aloepri studio --port 7861
```

浏览器打开`http://127.0.0.1:7861`。Studio只监听本地回环地址；任务状态保存到SQLite，聊天历史保存到浏览器本地存储。

## 官方DeepSeek-V3静态完整性

```powershell
uv run aloepri models inspect `
  --model-fixture tests\fixtures\deepseek-v3-official

uv run aloepri plan `
  --model-fixture tests\fixtures\deepseek-v3-official `
  --output artifacts\plans\deepseek-v3-official.yaml

uv run python scripts\audit_conversion_plan.py `
  --plan artifacts\plans\deepseek-v3-official.yaml `
  --require-layers 61 `
  --require-routed-experts 256 `
  --require-mtp-layers 1 `
  --require-fp8-block-size 128 128
```

## 文档

- [安装手册](docs/INSTALLATION_0.5.0.md)
- [模型选择手册](docs/MODEL_SELECTION_0.5.0.md)
- [RTX 3060转换手册](docs/RTX3060_CONVERSION_0.5.0.md)
- [Mock Cloud演示手册](docs/MOCK_CLOUD_DEMO_0.5.0.md)
- [架构适配器开发手册](docs/ADAPTER_DEVELOPMENT_0.5.0.md)
- [FP8与MTP实现说明](docs/FP8_MTP_IMPLEMENTATION_0.5.0.md)
- [密钥备份与换钥](docs/KEY_BACKUP_ROTATION_0.5.0.md)
- [故障恢复手册](docs/RECOVERY_0.5.0.md)
- [真实云端待验收清单](docs/REAL_CLOUD_ACCEPTANCE_TODO_0.5.0.md)
- [0.5.0验收报告](docs/ACCEPTANCE_0.5.0.md)
- [0.5.0可视化验收报告](docs/AloePri_0.5.0_Mock_Cloud_Acceptance.html)
- [版本说明](docs/VERSION_0.5.0.md)

旧Qwen、OpenSeek、论文复现、实验数据和错误推导文档仍保留在`docs/`与`artifacts/`中；0.5.0的Mock结果不得用于替代真实精度、攻击或云端性能数据。
