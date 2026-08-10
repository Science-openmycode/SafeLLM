# Qwen2.5-0.5B 功能缺口补齐记录

日期：2026-08-10

仓库：`E:\AloePri`

目标工件：`data/packages/qwen05b-product-v31-blockperm8`

在线密钥：`data/keys/qwen05b-product-v31-blockperm8-online`

## 1. 本轮完成结论

本轮没有修改已经通过验证的 v31 模型变换权重。修改集中在此前能够实现但尚未闭环的四类功能：可信客户端、流式协议、论文攻击实验入口和产品发布包。

| 功能 | 修改前 | 本轮实现 | 验证 |
|---|---|---|---|
| 远程传输 | SDK接受远程HTTP | 非localhost地址强制HTTPS | `test_private_client_requires_https_for_remote_servers` |
| 远程服务 | 远程绑定只检查Bearer Token | 远程绑定同时要求TLS证书或显式反向代理TLS终止 | 全仓ruff、mypy |
| SSE完整性 | 收到`done`即退出，不检查缺包、乱序和换request | 检查连续序号、唯一request ID、唯一`done`、`done`后无token、禁止截断 | 4种损坏SSE单元测试 |
| 流式RmDP账本 | `_privacy`被丢弃 | 每个恢复后的流式chunk携带本地隐私账本 | SDK测试 |
| M1随机数边界 | 客户端M1和服务端采样共用`seed` | 增加只在客户端使用的`privacy_seed`；未指定时使用系统随机数 | 检查HTTP正文不含`privacy_seed` |
| M1发送确认 | 只有`/privacy`可查看预计改变率 | 每次发送前显示预计token改变率，用户可取消；`--yes`可关闭交互确认 | CLI实现与静态检查 |
| IMA | 通用Transformer随机token smoke | 2层、8 Q-head、8 KV-head Qwen2反演器；公共语料切窗；train/val/test隔离；全词表分块Top-k | 真实v31 GPU冒烟 |
| TFMA | 同一prompt mixture | 支持独立prior/observed语料和三种knowledge setting；Top-1/10/100曲线 | 真实v31词表/密钥冒烟 |
| SDA | 单文件固定8:2切分，依赖可选NLTK | 支持独立train/test语料、三种knowledge setting；仓库内BLEU-4 | 真实Qwen tokenizer冒烟 |
| known-plaintext | 只支持均匀随机流量 | 支持真实语料token流和观测量恢复曲线，仍保留均匀对照 | 真实v31词表/密钥冒烟 |
| 发布包 | 只有服务端包和拆钥命令 | 生成client/server/configs/key-template/manifests/reports/evidence目录；递归SHA-256；禁止已配置密钥进入release | 本地19文件发布包检查通过 |

模型本体的词表置换、P/Q扩维、Embedding/LM Head噪声、Attention、GQA、RoPE、BlockPerm、FFN、RMSNorm修正式、Residual、HF forward/generate、KV Cache、FastAPI、SSE、vLLM和SGLang仍使用 v31 已有实现及证据。本轮全仓测试没有发现这些模块的回归。

## 2. 修改的代码

```text
src/aloepri/
├── client/sdk.py              # HTTPS、M1随机数拆分、SSE完整性、流式隐私账本
├── serving/app.py             # 非法Content-Length安全失败
├── cli.py                     # RmDP发送确认、TLS服务配置、release命令
├── release.py                 # 产品发布包构建和复验
└── attacks/
    ├── corpus.py              # TXT/MD/JSON/JSONL语料读取和固定窗口
    ├── ima.py                 # 2层8头Qwen2反演器和分块Top-k
    └── metrics.py             # 可审计的corpus BLEU-4

scripts/
├── train_ima_paper_like.py    # 公共语料IMA训练/验证/测试入口
├── run_tfma_curve.py          # 三种知识设置和独立语料
├── train_sda_smoke.py         # 独立训练/测试语料和内部BLEU-4
└── run_known_plaintext_curve.py

tests/
├── unit/test_client_sdk.py
├── unit/test_release.py
└── attacks/
    ├── test_paper_like_ima.py
    ├── test_tfma.py
    └── test_attack_metrics.py
```

## 3. 产品请求的实际执行顺序

1. 客户端加载原始Qwen tokenizer和只包含`tau/inverse_tau`的在线密钥。
2. `apply_chat_template()`在客户端把对话转成明文token ID。
3. `privacy_mode=rmdp`时，客户端先显示预计改变率；用户确认后用本地`privacy_seed`执行逐token M1。
4. 客户端计算`private_ids=tau(local_ids)`。
5. 非localhost服务地址若不是HTTPS，SDK在发包前拒绝。
6. 服务端只接收`model_id`、`key_id`、混淆token和生成参数。
7. 服务端混淆checkpoint执行prefill和decode，SSE逐token返回混淆ID。
8. 客户端逐事件检查request ID和sequence number，再执行`inverse_tau`。
9. 客户端本地tokenizer增量解码；服务器不获得明文对话历史。
10. RmDP账本留在客户端chunk和`/stats`结果中，不进入HTTP正文。

## 4. 直接运行命令

### 4.1 全仓门禁

```powershell
cd E:\AloePri
uv sync --frozen
uv run ruff check src scripts tests
uv run mypy src
uv run pytest -q
```

本轮结果：51个mypy源文件无错误；134个测试通过，1个真实模型测试因默认环境变量未设置而跳过。随后单独启用该测试并强制CPU运行，结果为1个通过。

```powershell
$env:ALOEPRI_RUN_MODEL_TESTS='1'
$env:ALOEPRI_SOURCE_MODEL='data/models/qwen2.5-0.5b'
$env:ALOEPRI_PRIVATE_MODEL='data/packages/qwen05b-product-v31-blockperm8'
$env:ALOEPRI_KEY_DIR='data/keys/qwen05b-product-v31-blockperm8-online'
$env:CUDA_VISIBLE_DEVICES='-1'
uv run pytest -q tests/integration/test_real_private_api.py
```

### 4.2 paper-like IMA

```powershell
uv run python scripts/train_ima_paper_like.py `
  --original data/models/qwen2.5-0.5b `
  --private data/packages/qwen05b-product-v31-blockperm8 `
  --key-dir data/keys/qwen05b-product-v31-blockperm8-online `
  --corpus <public-corpus.json> `
  --sequence-length 32 `
  --train-sequences 128 --val-sequences 16 --test-sequences 16 `
  --batch-size 4 --epochs 2 --topk 10 `
  --gpu-memory-fraction 0.75 `
  --out artifacts/privacy/v31-ima-paper-like.json
```

该入口严格创建Qwen2骨干，并把`num_hidden_layers=2`、`num_attention_heads=8`、`num_key_value_heads=8`写入结果。训练对使用已知明文token及其`tau`对应的私有Embedding行；结果明确记录目标`tau`是否用于形成已知对，不再把它误写成未知密钥攻击。

### 4.3 TFMA三种先验

```powershell
uv run python scripts/run_tfma_curve.py `
  --tokenizer data/models/qwen2.5-0.5b `
  --key-dir data/keys/qwen05b-product-v31-blockperm8-online `
  --prior-corpus <attacker-prior.json> `
  --observed-corpus <private-traffic.json> `
  --knowledge-setting domain-aware `
  --exposures 100 1000 10000 100000 `
  --out artifacts/privacy/v31-tfma-domain-aware.json
```

`unrelated`、`domain-aware`和`distribution-aware`的差异由传入语料决定，程序不再用同一文件伪造三种设置。频率候选通过排序和二分搜索产生，避免对每个观测token构造151,936长度的完整距离向量。

### 4.4 SDA

```powershell
uv run python scripts/train_sda_smoke.py `
  --tokenizer data/models/qwen2.5-0.5b `
  --train-corpus <attacker-train.json> `
  --test-corpus <held-out-private.json> `
  --knowledge-setting distribution-aware `
  --steps 2000 --batch-size 32 --max-len 128 `
  --out artifacts/privacy/v31-sda-distribution-aware.json
```

训练和测试文件可物理分离。输入是对token置换不变的frequency-rank序列，输出是原token；因果2层8头Transformer恢复序列，并输出token accuracy和corpus BLEU-4。

### 4.5 产品发布包

```powershell
uv build
uv run aloepri build-server-package `
  --source-checkpoint data/packages/qwen05b-product-v31-blockperm8 `
  --output data/server-packages/qwen05b-product-v31-blockperm8

uv run aloepri build-release `
  --output release/qwen05b-product `
  --server-package data/server-packages/qwen05b-product-v31-blockperm8 `
  --config configs/product/qwen05b_v31_blockperm8.yaml `
  --wheel dist/aloepri-0.1.0-py3-none-any.whl `
  --report docs/QWEN05B_V31_BLOCKPERM8_FUNCTIONAL_COMPLETION.md `
  --evidence artifacts/verification/qwen05b-product-v31-blockperm8/functional_evidence_index.json

uv run aloepri inspect-release --release release/qwen05b-product
```

生成结果：`release/qwen05b-product`，19个受清单约束的文件，检查结果`pass=true`。最小服务端目录已去掉tokenizer等非推理必需文件，并再次用真实0.5B API测试确认可以加载。模型文件使用硬链接优先的本地物化方式，避免重复占用约3.5GB磁盘；复制到其他介质或归档时仍作为普通文件处理。

## 5. 本轮真实冒烟数据

下表只列本轮确实运行的协议冒烟，不把它当作正式安全结论。

| 实验 | 实际输入规模 | 实际结果 | 工件 |
|---|---:|---:|---|
| IMA | train 2×4 token；val 1×4；test 1×4；1 epoch | Top-1 0；Top-10 0；CosSim -0.015307；GPU峰值1,593,397,760 bytes | `artifacts/privacy/v31-ima-paper-like-protocol-smoke.json` |
| TFMA | MMLU语料836,170 token；观察100 token | Top-1 0.038462；Top-10 0.153846；Top-100 0.365385 | `artifacts/privacy/v31-tfma-protocol-smoke.json` |
| TFMA | 同上；观察1,000 token | Top-1 0.012987；Top-10 0.051948；Top-100 0.151515 | 同上 |
| SDA | train 4序列；test 2序列；长度16；1 step | token accuracy 0；BLEU-4 0.401614 | `artifacts/privacy/v31-sda-protocol-smoke.json` |
| known-plaintext | MMLU流量1,000 token；10个已知对 | 流量token恢复率0.051 | `artifacts/privacy/v31-known-plaintext-corpus-smoke.json` |
| known-plaintext | 同上；100个已知对 | 流量token恢复率0.465 | 同上 |

IMA本次只用于证明真实0.5B权重、扩维私有Embedding、Qwen2反演器、GPU训练、全词表Top-k和证据绑定能够完整运行。8个已知训练对不足以形成安全结论。

## 6. 为什么这些地方需要修改

| 修改 | 原因 |
|---|---|
| M1使用独立`privacy_seed` | 原实现把生成seed同时发给服务端，不满足“客户端扰动随机数不离开客户端” |
| SDK强制远程HTTPS | 仅有Bearer Token不能阻止链路窃听混淆token和认证信息 |
| SSE严格状态机 | 乱序、重复、截断或串request时继续解码会产生不可检测的错误回答 |
| 内部BLEU-4 | SDA在默认锁定环境中因缺少可选NLTK不能运行；指标现在由仓库代码直接计算和测试 |
| IMA使用Qwen2骨干 | 原通用Transformer只满足“2层8头”的表面参数，不满足论文写明的Qwen架构 |
| release运行配置脱敏 | 原产品YAML含源模型路径、离线密钥路径和转换seed，不能直接发给服务器 |

## 7. 仍不能写成“论文原式已完成”的项目

以下不是尚未编程的普通缺口。

| 项目 | 状态 | 原因 |
|---|---|---|
| 论文标量RMSNorm严格等价 | 使用精确metric修正式 | 一般矩形P下不存在对所有输入成立的单一标量kappa |
| 论文Attn-IA原式 | 保留维度正确proxy和独立错误推导 | 原式矩阵维度不成立 |
| ISA同维隐藏态原协议 | 保留明确标注的维度适配proxy | 论文比较同维隐藏态，而本实现按论文扩维后是896对1152 |
| 一般长序列精确M1 | 小词表精确oracle＋产品逐token组合 | 论文未给151,936词表一般序列的可扩展精确采样器，且序列距离定义存在退化 |
| CCI3/MedDialog/Huatuo26M-Lite正式数值 | 代码已可接入，数值未生成 | 本地没有这三份数据及论文未公开的精确split |
| MoE、MLA、MTP | 对Qwen2.5-0.5B不适用 | 该checkpoint没有专家路由、MLA或MTP层 |
| 多GPU tensor parallel | 单GPU未验收 | 当前机器只有一张RTX 3060；这不是0.5B单卡功能缺失 |

因此当前可以准确表述为：Qwen2.5-0.5B适用且数学定义充分的功能代码和产品闭环已经实现；正式精度、完整攻击规模和性能门禁仍必须用锁定数据运行后才能决定是否安全发布。
