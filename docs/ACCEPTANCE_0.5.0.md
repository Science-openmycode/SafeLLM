# AloePri 0.5.0 Mock Cloud验收报告

验收日期：2026-08-14  
代码分支：`codex/aloepri-baseline`  
发布范围：本地转换、密钥拆分、Mock对象存储、Mock远程主机、Mock推理集群和本地Studio  
发布状态：`CODE_COMPLETE_MOCK_CLOUD_PASS`  
真实云状态：`NOT_TESTED`  
真实DeepSeek-V3 671B状态：`NOT_EXECUTED`

## 1. 代码入口

| 功能 | 文件 | 入口 |
|---|---|---|
| 模型目录 | `src/aloepri/catalog/registry.py` | `builtin_catalog()` |
| 固定revision下载 | `src/aloepri/catalog/download.py` | `download_pinned_snapshot()` |
| checkpoint检查 | `src/aloepri/catalog/inspect.py` | `inspect_local_checkpoint()` |
| 架构识别 | `src/aloepri/adapters/registry.py` | `AdapterRegistry.detect()` |
| Qwen/DeepSeek适配 | `src/aloepri/adapters/families.py` | 三个FamilyAdapter |
| 转换计划 | `src/aloepri/planning.py` | `build_local_plan()`、`build_catalog_plan()` |
| 统一执行器 | `src/aloepri/conversion/executor.py` | `execute_conversion_plan()` |
| DeepSeek流式转换 | `src/aloepri/conversion/deepseek_streaming.py` | `convert_deepseek_checkpoint()` |
| FP8 Codec | `src/aloepri/formats/deepseek_fp8.py` | block dequantize/requantize |
| Safetensors range I/O | `src/aloepri/tensor_io/safetensors_range.py` | `TensorSource`、`TensorSink` |
| SQLite状态 | `src/aloepri/jobs/store.py` | `JobStore` |
| Studio后台执行 | `src/aloepri/jobs/worker.py` | `ConversionWorker` |
| Mock上传部署 | `src/aloepri/workflow.py` | `MockCloudWorkflow` |
| Mock S3/SSH/Cluster | `src/aloepri/cloud/mock.py` | 三个Mock实现 |
| 生产接口 | `src/aloepri/cloud/production.py` | `S3ObjectStore`、`SSHRemoteHost` |
| 离线密钥加密 | `src/aloepri/keys/encryption.py` | AES-256-GCM＋Scrypt |
| 发布包 | `src/aloepri/release.py` | build/inspect release |
| 本地Studio | `src/aloepri/studio/app.py` | FastAPI、SSE和静态页面 |

## 2. 工作包验收

| 工作包 | 当前实现 | 验收证据 | 结果 |
|---|---|---|---|
| A 架构目录与适配器 | Qwen2、DeepSeek-V2、DeepSeek-V3；配置＋名称＋形状指纹；未知张量拒绝 | `tests/test_catalog.py`、`tests/test_plan_scale.py`、官方V3 plan audit | PASS |
| B CLI与任务状态 | CREATED到ROLLED_BACK；pause/cancel/resume；tile、事件、deployment SQLite持久化 | `tests/test_jobs.py`、`tests/test_conversion_control.py`、`tests/test_product_cli.py` | PASS |
| C Safetensors I/O | header解析、range read、预分配range write、partial/flush/SHA/原子提交 | `tests/test_safetensors_range.py`、中断恢复测试 | PASS |
| D 数学变换 | Qwen dense/GQA；DeepSeek MLA、dense/shared/routed/fused MoE；逐专家写出；确定性层密钥 | Qwen/OpenSeek真实转换；DeepSeek paper-complete单元与集成测试 | PASS |
| E FP8 | 128×128 E4M3FN；FP32变换；输出scale重算；边缘零填充；NaN/Inf拒绝 | `tests/unit/test_deepseek_fp8.py`、tiny FP8＋MTP转换 | PASS |
| F MTP | `enorm`、`hnorm`、`eh_proj`、MTP层、shared embedding/head、tiny候选验证 | `tests/unit/test_deepseek_paper_complete.py`；官方V3 MTP 1,564张量覆盖 | PASS |
| G 云接口 | 可恢复multipart、ETag/SHA、Mock SSH、健康检查、SSE、蓝绿回滚 | `tests/test_mock_cloud.py`、`tests/test_product_cli.py`、真实转换输出Mock上传部署 | PASS |
| H 发布与密钥 | online/offline/server三分；所有离线Safetensors加密；server秘密扫描 | 两个0.5.0 release inspect均PASS；明文离线Safetensors=0 | PASS |
| I Studio | 模型、任务、部署、聊天、隐私、设置；后台worker；刷新恢复；SSE | `tests/test_studio_api.py`、`tests/test_studio_assets.py` | PASS |
| J 发布验收 | Ruff、Mypy、Pytest、V3静态审计、资源审计、真实小模型前向 | 本报告第3至7节 | PASS |

## 3. 自动化检查

执行命令：

```powershell
.\.venv\Scripts\ruff.exe check src scripts tests
.\.venv\Scripts\mypy.exe src
.\.venv\Scripts\pytest.exe -q
```

| 检查 | 当前结果 |
|---|---:|
| Ruff | PASS |
| Mypy | 97个源文件，0 error |
| Pytest collected | 257 |
| Pytest passed | 256 |
| Pytest skipped | 1 |
| Pytest failed | 0 |

跳过项为`tests/integration/test_real_private_api.py`，需要显式设置`ALOEPRI_RUN_MODEL_TESTS=1`。当前发布checkpoint的真实前向不引用该跳过结果，使用第4节的独立工件。

## 4. 真实checkpoint结果

| 模型 | 结构 | 转换输出 | 明文next-token | 私有next-token | inverse_tau结果 | 一致 |
|---|---|---|---:|---:|---:|---|
| Qwen2.5-0.5B-Instruct | dense、GQA、RoPE、RMSNorm | `artifacts/converted/qwen05b-v050` | 108386 | 6287 | 108386 | PASS |
| OpenSeek-Small-v1-SFT | MLA、MoE、fused experts | `artifacts/converted/openseek-v050-release` | 9707 | 123415 | 9707 | PASS |

原始证据：

- `artifacts/audits/qwen05b-v050-forward.json`
- `artifacts/audits/openseek-v050-release-forward.json`

OpenSeek源配置声明1层MTP，但checkpoint没有任何MTP张量。转换器只在`normalization_manifest.json`明确记录且全部源文件SHA-256验证通过时，把有效MTP层数设为0；输出manifest同时保留“源声明1、有效0、依据verified-normalization-manifest”。没有伪造MTP权重。MTP执行能力由tiny真实Safetensors测试和官方V3静态索引单独验收。

## 5. DeepSeek-V3完整结构静态验收

固定来源：

```text
repo_id: deepseek-ai/DeepSeek-V3
revision: bb399fea3bbfbea55d71cb018e12cdfb6b215179
```

执行命令：

```powershell
.\.venv\Scripts\python.exe scripts/audit_conversion_plan.py `
  --plan artifacts/plans/deepseek-v3-official.yaml `
  --require-layers 61 `
  --require-routed-experts 256 `
  --require-mtp-layers 1 `
  --require-fp8-block-size 128 128
```

| 项目 | 结果 |
|---|---:|
| 索引张量 | 91,991 |
| MTP张量 | 1,564 |
| 主层 | 61 |
| routed experts/层 | 256 |
| MTP层 | 1 |
| FP8块 | 128×128 |
| missing | 0 |
| unknown | 0 |
| `copied_unknown` | 0 |
| 综合门禁 | PASS |

静态审计证明转换计划能够认领官方权重结构；它不是671B物理转换或推理结果。

## 6. 3060资源预算

执行命令：

```powershell
.\.venv\Scripts\python.exe scripts/audit_resource_budget.py `
  --config tests/fixtures/deepseek-v3-official/config.json `
  --expansion-h 128 `
  --host-budget-gib 11 `
  --output artifacts/audits/deepseek-v3-resource-budget.json
```

| 阶段 | 估算峰值GiB |
|---|---:|
| Algorithm 1 / 层密钥生成 | 5.524 |
| 词表权重变换 | 10.165 |
| Attention/MLA变换 | 3.657 |
| 单专家变换 | 2.310 |
| 计划host上限 | 11.000 |

估算假设：一次常驻一个层密钥；FP8不生成完整padding/restored副本；fused experts逐专家读写；噪声使用固定4,096元素随机块和受调用者限制的更新tile；不计操作系统page cache。执行器在计划峰值超过host上限时拒绝启动。源权重、目标权重和对象存储容量不包含在50GiB临时空间内。

## 7. Mock上传、部署与发布包

当前Mock闭环使用真实转换后的本地文件，不使用空目录或虚构文件：

| 模型 | upload job | deployment | 结果 |
|---|---|---|---|
| OpenSeek | `10b18e50-ceb7-403c-863b-f8aecff1c0b4` | `mock-1049d436-4956-462b-8b5b-8bc62896d185` | 上传、校验、部署、恢复RUNNING |
| Qwen | `9f07161c-2054-42f6-9f1d-595ae797b9a6` | `mock-0f1ba994-b496-49d6-9d9b-3c737e3051e5` | 上传、校验、部署、回滚 |

每个上传对象记录part编号、ETag、字节数和SHA-256；新进程可读取`.multipart`状态继续上传。部署证据固定包含：

```json
{
  "environment": "mock-cloud",
  "real_cloud_validated": false
}
```

发布包：

```text
release/qwen05b-v050/0.5.0-current
release/openseek-v050-release/0.5.0-current
```

两个目录的release manifest、文件集合、字节数、SHA-256和server package秘密扫描均为PASS。Qwen和OpenSeek最终离线密钥目录中的明文`*.safetensors`数量均为0。

## 8. 从零执行

```powershell
$env:ALOEPRI_OFFLINE_KEY_PASSWORD = "<从密码管理器取得>"

uv sync --frozen
uv run aloepri models list
uv run aloepri models inspect --model <本地checkpoint目录或catalog-id>
uv run aloepri doctor --model <本地checkpoint目录或catalog-id>
uv run aloepri plan --model <本地checkpoint目录或catalog-id> `
  --output artifacts/plans/private-model.yaml `
  --destination artifacts/converted/private-model
uv run aloepri convert --plan artifacts/plans/private-model.yaml
uv run aloepri upload <job-id> --cloud-profile mock
uv run aloepri deploy <job-id> --cloud-profile mock
uv run aloepri studio --port 7861
```

暂停、恢复和取消：

```powershell
uv run aloepri jobs pause <job-id>
uv run aloepri jobs resume <job-id>
uv run aloepri jobs cancel <job-id>
```

`resume`会重新读取原plan和SQLite断点；DeepSeek逐tensor/逐expert记录已完成tile；Mock multipart只补传缺失part。损坏SHA、转换参数变化、缺FP8 scale、缺MTP权重、未知张量或model/key版本不匹配均拒绝继续。

## 9. 发布边界

本版本已经完成代码与Mock Cloud验收。以下项目没有实测数据，不写入通过项：真实S3/SSH/TLS/IAM、完整DeepSeek-V3 671B权重物理转换、真实多节点SGLang加载、MTP加速、671B TTFT/TPOT/吞吐、真实云攻击实验。
