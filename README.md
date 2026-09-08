# 隐变智模

隐变智模是一套面向私有大模型推理的模型改造、部署和对话工具。项目提供两种相互独立的安全模式：

- `permutation`：本地保存 Token 置换关系，云端运行同步改造后的模型；
- `tee_gm`：原始 Token 通过国密信道进入 TEE，TEE 负责 Embedding 和 LM Head，GPU 运行 P/Q 坐标下的 Transformer 主体。

Windows 桌面端覆盖模型选择、下载、改造、服务器检查、部署、对话和运行原理展示；命令行使用相同的状态库与核心实现。

## 当前验收边界

| 能力 | 当前状态 |
|---|---|
| Qwen2.5-0.5B-Instruct 置换模式 | 已完成转换、部署和对话闭环 |
| Qwen2/Qwen2.5、Qwen3、DeepSeek、GLM、Kimi 架构识别 | 已纳入统一模型目录，按各版本门禁开放检查或转换 |
| TEE 拆分与 Local TEE Head | Qwen2.5-0.5B 软件环境闭环通过 |
| 掩码外包 LM Head | 软件参考实现与安全回退已接入 |
| Intel TDX、DCAP Quote、RFC 8998 物理互操作 | 尚需真实 TDX 硬件验收 |

架构兼容不等于每个大模型检查点都完成了真实 GPU 部署。产品只会开放已达到相应门禁的操作；未知权重、资源不足或运行依赖不满足时会明确拒绝。

## 阅读代码

从 [代码部件地图](docs/architecture/COMPONENT_MAP.md) 开始。主要目录如下：

```text
src/aloepri/
├── desktop/       Windows 部署端与对话端
├── product/       任务、状态、路径和流程编排
├── catalog/       模型目录、下载与结构识别
├── adapters/      模型架构适配器
├── conversion/    转换执行与工件生成
├── transforms/    P/Q 与各层权重变换
├── keys/          密钥和凭据管理
├── cloud/         SSH、服务器预检与部署
├── serving/       模型服务运行入口
├── client/        置换模式客户端
├── tee/           国密、证明与可信边界
└── secure_head/   LM Head 安全外包
```

TEE 相关实现采用明确的职责名称，例如 `gm_cryptography.py`、`attestation_service.py`、`package_integrity.py` 和 `trusted_boundary.py`。旧模块名仍作为兼容转发保留，不影响既有调用。

## 代码与数据分离

模型权重、改造模型、密钥、数据库和运行证据不会上传 GitHub。建议将运行状态与大体积数据放在两个外部目录：

```powershell
$env:YINBIAN_HOME = "E:\YinbianRuntime"
$env:YINBIAN_DATA_DIR = "E:\YinbianData"
```

默认结构：

```text
E:\YinbianRuntime\   状态库、聊天记录、凭据和日志
E:\YinbianData\
├── source-models\   原始模型
├── private-models\  改造后的模型
├── cache\           下载缓存
└── evidence\        验收证据
```

完整说明与旧目录兼容方式见 [代码与模型数据分离](docs/guides/DATA_STORAGE.md)。仓库中的 `data/` 只允许保留明确白名单内的小型评测样本。

## 安装

源码环境要求 Python 3.11 和 Windows 10/11 x64。当前锁文件实测组合为 Python
3.11.15、uv 0.12.1、PyTorch 2.9.1+cu128、Transformers 5.12.0。云端部署使用独立的
Ubuntu 22.04、PyTorch 2.5.1+cu121 运行环境，不与本地开发环境混用。

```powershell
uv sync --frozen
uv run yinbian --help
```

桌面程序：

```powershell
uv run yinbian-deploy
uv run yinbian-chat
```

构建后的 Windows 程序名为 `隐变智模部署.exe` 和 `隐变智模对话.exe`。详细步骤见
[安装与运行手册](docs/YINBIAN_1.0_INSTALLATION.md)；从 GitHub 干净克隆、安装、测试、
构建和真实模型验收的完整口径见 [1.1 可复现说明](docs/REPRODUCIBILITY_1.1.md)。

## 常用流程

```powershell
# 查看模型与推荐
uv run yinbian models recommend

# 只下载并检查固定版本元数据
uv run yinbian models download --model qwen2.5-0.5b-instruct --metadata-only

# 添加并检查 GPU 服务器
uv run yinbian servers add --name gpu --host 203.0.113.10 `
  --auth-type private_key --private-key C:\Keys\id_ed25519
uv run yinbian servers check <server-id> --trust-host-key

# 创建任务并执行转换
uv run yinbian plan create --model qwen2.5-0.5b-instruct `
  --mode direct-deploy --device auto --server <server-id>
uv run yinbian convert --plan <plan-path> --accept-license
```

部署服务只监听远端 `127.0.0.1`，对话程序通过 SSH 隧道访问。候选服务必须实际完成一次私有 Token 解码才能标记健康。

## 安全模式

模式组合由三个字段控制：

```yaml
security_mode: permutation | tee_gm
boundary_mode: in_model | tee_split
tee_backend: software_sim | intel_tdx
```

旧配置缺少这些字段时自动使用 `permutation + in_model`。TEE 模式的边界、国密通信、软件模拟与真实 TDX 门禁见 [TEE 国密模式说明](docs/guides/TEE_GM_MODE.md)。普通 RTX 服务器不能作为 TDX 硬件证明证据。

## 开发与验证

```powershell
uv sync --frozen --extra eval
uv run pytest
uv run ruff check .
uv run mypy src/aloepri
uv build
```

修改前请阅读 [开发约定](CONTRIBUTING.md)。版本状态、模型族边界和历史研究证据分别见：

- [1.1 模型族与一键部署](docs/YINBIAN_1.1_MODEL_FAMILIES_AND_ONE_CLICK.md)
- [1.1 发布说明](docs/RELEASE_NOTES_1.1.0.md)
- [模型与部署审计](docs/MODEL_SUPPORT_AUDIT_2026-08-15.md)
- [历史工件政策](docs/LEGACY_ARTIFACT_POLICY.md)

`aloepri` 命令和 `src/aloepri` Python 包名为兼容名称，现有模型、密钥、数据库和部署无需改名。
