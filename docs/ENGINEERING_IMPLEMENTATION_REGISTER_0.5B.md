# AloePri 0.5B 工程实现登记表

范围：`Qwen2.5-0.5B-Instruct`。路径均相对仓库根目录 `E:\AloePri`。

## 1. 运行环境

```powershell
cd E:\AloePri
.\.venv\Scripts\Activate.ps1
$env:HF_DATASETS_OFFLINE='1'
$env:HF_HUB_OFFLINE='1'
```

| 项目 | 路径或版本 |
|---|---|
| Python 环境 | `.venv/` |
| 明文模型 | `data/models/qwen2.5-0.5b/` |
| Python 包 | `src/aloepri/` |
| 配置 | `configs/` |
| 命令入口 | `scripts/` |
| 自动测试 | `tests/` |
| 运行证据 | `artifacts/` |
| 报告 | `docs/`、`output/pdf/` |

## 2. 模型变换代码

| 组件 | 实现文件 | 实现内容 | 测试 |
|---|---|---|---|
| 词表置换 | `src/aloepri/transforms/vocab.py` | `tau`、`inverse_tau`、Embedding 行和 Head 行同步置换 | `tests/unit/test_vocab.py`、`test_special_tokens.py` |
| Embedding/Head 噪声 | `src/aloepri/transforms/paper_noise.py` | 按原权重标准差生成高斯噪声，Embedding 与 Head 独立 seed | `tests/unit/test_paper_noise.py` |
| Algorithm 1 | `src/aloepri/transforms/paper_key_matrix.py` | `d -> d+2h`、P/Q、零空间构造、条件数和误差统计 | `tests/unit/test_paper_key_matrix.py` |
| 独立兼容右逆 | `src/aloepri/transforms/paper_key_matrix.py` | 固定公共 P，在 `null(P)` 中生成 Qq/Qk/Qv/Qffn-gate/Qffn-up/Qhead | `tests/unit/test_paper_key_matrix.py` |
| Attention | `src/aloepri/transforms/qwen_structural.py` | Q/K/V/O 坐标变换、GQA head permutation、RoPE block permutation、Q/K 缩放、Uvo | `tests/unit/test_qwen_structural.py` |
| FFN | `src/aloepri/conversion/paper_qwen2.py` | Gate/Up/Down 置换、缩放和 P/Q 变换 | `tests/unit/test_paper_qwen2_conversion.py` |
| RMSNorm | `src/aloepri/transforms/rms_calibration.py` | 逐层期望比例、标准差、标准误、分位数 | `tests/unit/test_rms_calibration.py` |
| Qwen checkpoint | `src/aloepri/conversion/paper_qwen2.py` | 24 层 Dense Qwen 权重转换和配置写入 | `tests/unit/test_paper_qwen2_conversion.py` |
| 自定义 HF 模型 | `src/aloepri/models/modeling_aloepri_qwen2.py` | 扩维 Qwen forward、generate、prefill、decode、KV Cache | `tests/unit/test_aloepri_qwen2_model.py` |

## 3. 转换、保存和恢复

| 功能 | 入口 | 产物 |
|---|---|---|
| 整模型转换 | `scripts/convert_paper_qwen2_checkpoint.py` | safetensors shards、config、tokenizer、`aloepri_manifest.json`、密钥 |
| 流式转换 | `scripts/convert_paper_qwen2_streaming.py` | 逐层处理、`.partial`、断点恢复、原子改名 |
| RMS 标定 | `scripts/calibrate_paper_rms.py` | `artifacts/calibration/*.json` |
| 模型清单验证 | `scripts/verify_manifest.py` | 文件数、大小和 SHA-256 检查结果 |
| checkpoint 验证 | `scripts/verify_paper_qwen2_checkpoint.py` | logits、top-1、greedy、prefill/decode/KV Cache 结果 |

正式配置：

| 配置 | 作用 |
|---|---|
| `configs/transform/paper_qwen05b_complete.yaml` | v15，论文数值超参数和 corrected-paper 修正公式，BF16 |
| `configs/transform/paper_qwen05b_engineering_complete.yaml` | v16，低噪声工程候选，FP32 |
| `configs/transform/paper_qwen05b_ablation_no_noise.yaml` | v17，仅关闭 Embedding/Head 噪声的因果消融，BF16 |

对应 checkpoint：

| ID | 模型目录 | 密钥目录 |
|---|---|---|
| v15 | `data/checkpoints/qwen2.5-0.5b-paper-v15-complete-bf16/` | `data/keys/dev-qwen05b-paper-v15-complete-bf16/` |
| v16 | `data/checkpoints/qwen2.5-0.5b-paper-v16-engineering-complete-fp32/` | `data/keys/dev-qwen05b-paper-v16-engineering-complete-fp32/` |
| v17 | `data/checkpoints/qwen2.5-0.5b-paper-v17-no-noise-bf16/` | `data/keys/dev-qwen05b-paper-v17-no-noise-bf16/` |

## 4. 客户端和服务端

| 组件 | 实现文件 | 职责 |
|---|---|---|
| 密钥加载 | `src/aloepri/client/sdk.py` | 校验 key manifest；只在客户端读取 `tau` 和 `inverse_tau` |
| 客户端 SDK | `src/aloepri/client/sdk.py` | prompt 分词、token ID 置换、HTTP/SSE 请求、逐 token 逆置换、解码 |
| 请求 schema | `src/aloepri/serving/protocol.py` | `model_id`、`key_id`、`input_ids`、生成参数 |
| 私有 API | `src/aloepri/serving/app.py` | 非流式和 SSE；校验模型/密钥；返回 output IDs、usage、TTFT、TPOT |
| 审计日志 | `src/aloepri/serving/audit_log.py` | 只记请求 ID、模型 ID、密钥 ID、计数和耗时；不记文本/密钥/token 流 |

真实模型验证入口：

```powershell
.\.venv\Scripts\python.exe scripts\verify_private_api_local.py `
  --source data\models\qwen2.5-0.5b `
  --private data\checkpoints\qwen2.5-0.5b-paper-v15-complete-bf16 `
  --key-dir data\keys\dev-qwen05b-paper-v15-complete-bf16 `
  --out artifacts\api\paper-qwen05b-v15-complete-bf16.json
```

## 5. 准确率与统计

| 功能 | 入口 | 统计口径 |
|---|---|---|
| MMLU/C-Eval/PIQA | `scripts/run_lm_eval.py` | 本地 task YAML、0-shot、逐样本 log-likelihood |
| 配对比较 | `scripts/compare_lm_eval.py` | 同 doc hash 配对；10,000 次非参数 bootstrap |
| IFEval 生成 | `scripts/run_hf_ifeval.py` | 可恢复逐 batch 保存；保留每条 response |
| IFEval 评分 | `scripts/score_ifeval.py` | prompt/instruction × strict/loose |
| IFEval 比较 | `scripts/compare_ifeval.py` | 逐 prompt 或 instruction 配对 bootstrap |
| HumanEval 生成 | `scripts/run_lm_eval.py` + `configs/eval/humaneval_local/` | 每题 1 个确定性生成 |
| HumanEval 执行 | `scripts/evaluate_humaneval_wsl.py` | WSL 隔离、CPU/内存/文件/网络限制 |
| HumanEval 比较 | `scripts/compare_humaneval.py` | 逐题通过/失败配对 bootstrap |

## 6. 隐私攻击

| 攻击 | 入口或实现 | 当前 0.5B 协议 |
|---|---|---|
| 直接权重匹配/NN | `scripts/run_direct_attack.py`、`src/aloepri/attacks/mapping.py` | 全矩阵哈希；NN 分批扫描候选词表 |
| VMA | `scripts/run_vma_pupa.py` | PUPA 全查询 token；16,384 候选；5 个 Dense 组合；24 层投票；CPU 流式权重；TTRSR/PIIRSR/BLEU-4/CosSim/Wilson 95% 区间 |
| Gate-IA | `scripts/run_gate_ia.py` | 24 层 Gate 均值不变量，top-1/top-10 |
| Attn-IA | `scripts/run_attn_ia.py` | 维度成立的 Q/K RoPE block 代理；标记 `paper_exact=false` |
| ISA | `scripts/run_isa_hidden_state.py` | 隐状态/Token Gram 诊断代理、最近 token 恢复；`paper_exact=false`，不计论文精确通过 |
| IMA | `scripts/train_ima_smoke.py` | 独立 surrogate key、2 层 8-head inversion transformer |
| TFMA | `scripts/run_tfma_curve.py` | 观测量曲线、top-1/top-10/top-100 |
| SDA | `scripts/train_sda_smoke.py` | recurrence encoding、因果 Transformer、held-out BLEU-4 |
| Known plaintext | `src/aloepri/attacks/mapping.py` | 观测量和映射恢复率曲线 |

## 7. 性能

| 功能 | 入口 | 输出 |
|---|---|---|
| HF 性能 | `scripts/benchmark_hf.py` | 每请求 TTFT、TPOT、吞吐、峰值显存、加载时间 |
| 平衡性能调度 | `scripts/run_balanced_hf_performance.ps1` | 8 个独立进程，ABBA+BAAB，模型/key/prompt/环境/hash 绑定 |
| 配对性能比较 | `scripts/compare_performance.py` | 80 对请求的 p50/p95/p99 相对劣化和 bootstrap 95% 上界 |
| vLLM | `src/aloepri/serving/vllm_plugin.py`、`vllm_qwen2.py`、`scripts/vllm_smoke.py` | 自定义模型注册和服务入口；本机环境需安装兼容 vLLM/CUDA 后运行 |

## 8. 质量门禁

```powershell
.\.venv\Scripts\ruff.exe check src scripts tests
.\.venv\Scripts\mypy.exe src
.\.venv\Scripts\pytest.exe -q
```

真实 0.5B 集成测试：

```powershell
$env:ALOEPRI_RUN_MODEL_TESTS='1'
.\.venv\Scripts\pytest.exe tests\integration\test_real_private_api.py -q
```

验收聚合：

```powershell
.\.venv\Scripts\python.exe scripts\build_qwen05b_final_acceptance.py `
  --config configs\transform\paper_qwen05b_complete.yaml `
  --out artifacts\acceptance\paper-qwen05b-v15-complete-bf16.json
```

## 9. 论文勘误与复现限制

| 内容 | 路径 |
|---|---|
| E01–E10 公式推导 | `docs/paper_errata/*.md` |
| E01–E10 独立 PDF | `output/pdf/paper_errata/*.pdf` |
| 论文未给出的复现参数 | `docs/PAPER_REPRODUCTION_MISSING_DETAILS.md` |
| 论文逐项实现审核 | `docs/PAPER_IMPLEMENTATION_TRACEABILITY_AUDIT.md` |

## 10. 当前 v15 运行登记

| 作业 | 参数 | 产物 | 结果或资源峰值 |
|---|---|---|---|
| MMLU | 14,042 题，BF16 | `artifacts/accuracy/mmlu-paper-v15-complete-bf16-comparison.json` | -7.214 pp，95% CI [-8.049, -6.386] |
| C-Eval | 1,346 题，BF16 | `artifacts/accuracy/ceval-paper-v15-complete-bf16-comparison.json` | -26.969 pp，95% CI [-30.314, -23.551] |
| PIQA | 1,838 题，BF16 | `artifacts/accuracy/piqa-paper-v15-complete-bf16-comparison.json` | -11.371 pp，95% CI [-13.765, -9.032] |
| PUPA VMA | 16,384 候选，24 层，5 组合，空缓存 97 文件 | `artifacts/privacy/vma-pupa-paper-v15-complete-bf16-c16384.json` | 最高 TTRSR 10.588%；1,346.05 秒；缓存未绑定历史文件 |
| Gate-IA | 512 token，24 层 | `artifacts/privacy/gate-ia-paper-v15-complete-bf16-full-vocab.json` | Top-1 17.383% |
| Attention-IA 代理 | 256 token，4 层 | `artifacts/privacy/attn-ia-paper-v15-complete-bf16-proxy.json` | Top-1 7.031%；`paper_exact=false` |
| IMA | 8192 token，2000 步 | `artifacts/privacy/ima-paper-v15-complete-bf16.json` | 恢复率 0%；协议等价未确认 |
| ISA | 20 prompt，125 token，100 步 | `artifacts/privacy/isa-paper-v15-complete-bf16.json` | TTRSR 0%；`paper_exact=false`；私有/优化阶段峰值 1.6336/1.5514 GB |
| TFMA | 100 至 100,000 观测 token | `artifacts/privacy/tfma-paper-v15-complete-bf16.json` | 最大 Top-10 13.462% |
| SDA 代理 | MMLU 10,000/2,000，2000 步 | `artifacts/privacy/sda-paper-v15-complete-bf16.json` | BLEU-4 13.120 |
| HF 性能 | ABBA+BAAB，8 进程，80 对请求，每请求 100 token，batch=1；运行时绑定 tokenizer/dtype/device/script | `artifacts/performance/hf-balanced-v15-v2/`；`artifacts/performance/hf-paper-v15-complete-bf16-comparison.json` | 私有峰值 1,649,150,976 bytes；TTFT p50 与 TPOT p50/p95/p99 通过，TTFT p95/p99 未通过联合 CI 门禁 |
| API | 真实 v15 checkpoint，HTTP 与 SSE | `artifacts/api/paper-qwen05b-v15-complete-bf16.json` | HTTP 200；流式一致；错误 key 400 |
| IFEval | 541 prompt / 834 instruction，BF16，batch=4，1280 token 上限，eager attention | `artifacts/accuracy/ifeval-paper-v15-complete-bf16-comparison.json` | prompt strict 22.181%→7.579%，-14.603 pp，95% CI [-18.299, -10.906]；四项均未通过；正式 provenance 已绑定 |
| HumanEval | 164 题，BF16，batch=4，greedy，512 token 上限；WSL 逐题受限执行 | `artifacts/accuracy/humaneval-paper-v15-complete-bf16-comparison.json` | 42/164→0/164，-25.610 pp，95% CI [-32.317, -18.902]；正式 provenance 已绑定 |

GPU 作业串行执行。VMA 的全词表矩阵留在 CPU；ISA 和 checkpoint 验证的私有模型/明文模型分阶段加载；HF 每轮独立进程退出后才启动下一轮。IFEval 私有阶段运行时 JSON 记录的批次间最小空闲显存为 3,504,340,992 bytes；外部监控的瞬时空闲显存在约 2.1-2.9 GiB。HumanEval 明文阶段空闲约 3.46-3.62 GiB，私有阶段空闲约 2.76-2.84 GiB，未用满 6 GiB GPU。

## 11. v18–v25 函数等价与产品化作业登记

下表只登记当前重建链路。旧 v15 的精度和攻击结果不作为这些 checkpoint 的证据。

| 作业 | 与上一版的唯一主要差异 | 产物 | 实测结果 |
|---|---|---|---|
| v18 | 修正 Algorithm 1 inverse family；论文标量 $\kappa$；$\beta=8$ | `data/checkpoints/qwen2.5-0.5b-paper-v18-alg1-faithful-fp32` | Prefill mean abs 2.631565；Top-1 22.22%；生成失败 |
| v19 | Algorithm 2 关闭；精确度量 RMSNorm | `data/checkpoints/qwen2.5-0.5b-paper-v19-exact-rms-core-fp32` | Prefill mean abs $1.039\times10^{-5}$；Top-1 100%；greedy 一致 |
| v20 | Algorithm 2 开启；论文 $\beta=8$；精确 RMS | `data/checkpoints/qwen2.5-0.5b-paper-v20-exact-rms-alg2-fp32` | 603 个 RoPE 块跨频率移动；mean abs 0.012042；Top-1 97.22% |
| v21 | $\beta=1$；$U_{vo}$ condition $\le100$；无噪声 | `data/checkpoints/qwen2.5-0.5b-product-v21-beta1-no-noise-fp32` | 453/460 逐层项通过；200/200 greedy；prefill/decode mean abs $1.842\times10^{-5}$/$9.855\times10^{-6}$ |
| v22 | v21 结构；$\alpha_e=1.0,\alpha_h=0.2$ | `data/checkpoints/qwen2.5-0.5b-product-v22-alpha1-head02-fp32` | 20 条问答 0/20 exact；token agreement 61.125%；单 key We/Wh VMA 数值达标 |
| v23 | 无噪声；高斯 $U_{vo}$ 拒绝采样 condition $\le50$ | `data/checkpoints/qwen2.5-0.5b-product-v23-uvo50-no-noise-fp32` | 455/460 逐层项通过；$U_{vo}$ condition 37.92–49.95 |
| v24-uvo40 | 尝试 condition $\le40$ | `.partial` 目录 | 某层 10,000 次高斯采样均无合格矩阵，按设计中止；未形成 checkpoint |
| v24 | 离线 Q/K/V 投影改为 FP64 后一次舍入；condition $\le50$ | `data/checkpoints/qwen2.5-0.5b-product-v24-fp64proj-uvo50-no-noise-fp32` | 456/460 逐层项通过 |
| v25 | v24；condition $\le45$ | `data/checkpoints/qwen2.5-0.5b-product-v25-fp64proj-uvo45-no-noise-fp32` | 457/460；200/200 greedy；prefill/decode mean abs $1.521\times10^{-5}$/$9.074\times10^{-6}$ |
| v26 | Q/K 逆矩阵按持久化值配对；结构 key FP64 | checkpoint 已清理；key 与证据保留 | Q/K 代数误差降至 $3.61\times10^{-16}$，但仅 456/460；拒绝作为基线 |
| v27 | 基础 Attention P/Q 投影也改为 FP64 | checkpoint 已清理；key 与证据保留 | 454/460；FP32 原模型的舍入轨迹反而偏离，拒绝 |
| v28 | seed 20260805；$U_{vo}\le42$ | checkpoint 已清理；key 与证据保留 | 456/460；seed 搜索不优于 v25，拒绝 |
| v29 | seed 20260803；$U_{vo}\le42$；仅 Attention Q/K/V/O 权重与计算 FP64；结构 key FP64 | `data/checkpoints/qwen2.5-0.5b-product-v29-attnfp64-uvo42-no-noise` | 457/460；最终 logits NRMSE $4.756\times10^{-6}$；200/200 greedy；HTTP/SSE 与产品边界通过 |

### 当前 v29 正确性优先工件

| 工件 | 路径 | 实测检查 |
|---|---|---|
| 完整 checkpoint | `data/checkpoints/qwen2.5-0.5b-product-v29-attnfp64-uvo42-no-noise` | 4 个 shard；Attention FP64，其余 FP32 |
| 完整转换 key | `data/keys/dev-qwen05b-product-v29-attnfp64-uvo42-no-noise` | 结构矩阵 FP64；manifest SHA-256 已更新 |
| 在线 key | `data/keys/qwen05b-product-v29-online` | 仅 `tau`、`inverse_tau` 与特殊 token 映射 |
| 离线 master key | `data/keys/qwen05b-product-v29-offline` | P/Q、Attention、FFN 与重建材料 |
| 净化服务器包 | `data/packages/qwen05b-functional-v29` | `pass=true`，`findings=[]` |
| 产品配置 | `configs/product/qwen05b_v29_functional.yaml` | HF `dtype=auto`，保留混合精度 |
| 逐层证据 | `artifacts/verification/qwen05b-product-v29-layerwise-fp64-key.json` | 457/460；logits NRMSE $4.756\times10^{-6}$ |
| 200 prompt 证据 | `artifacts/verification/qwen05b-product-v29-greedy200.json` | 200/200；3200/3200 新 token 一致 |
| Prefill/decode 证据 | `artifacts/verification/qwen05b-product-v29-checkpoint.json` | Top-1、next token、greedy、Cache 全部一致 |
| HTTP 隐私边界 | `artifacts/verification/qwen05b-functional-v29/privacy-boundary.json` | 全部检查为 true；实际恢复文本为“42” |
| HTTP/SSE | `artifacts/verification/qwen05b-functional-v29/http-sse-smoke.json` | 私有 token 顺序一致；正常中文回答 |

v29 私有阶段 CUDA peak allocated 为 3,692,543,488 bytes。测试过程始终先释放明文模型，
再加载私有模型；两个模型不同时驻留 GPU。

### 11.1 v25 模型加载与显存

| 阶段 | 同时加载模型数 | CUDA peak allocated | 结果 |
|---|---:|---:|---|
| 明文 forward/generate | 1 | 2,112,222,720 bytes | 完成后释放 |
| 私有 forward/generate | 1 | 3,451,209,216 bytes | 约占 6 GiB 的 53.6% |
| 双模型对照 | 0 | 不同时驻留 | 明文结果转 CPU 后再加载私有模型 |

### 11.2 v25 产品工件

| 工件 | 路径 | 检查 |
|---|---|---|
| 完整 checkpoint | `data/checkpoints/qwen2.5-0.5b-product-v25-fp64proj-uvo45-no-noise-fp32` | manifest 绑定 4 个 shard |
| 完整转换 key | `data/keys/dev-qwen05b-product-v25-fp64proj-uvo45-no-noise-fp32` | 离线转换与验证使用 |
| 在线 key | `data/keys/qwen05b-product-v25-online` | `tau`、`inverse_tau`；文件集合、大小、SHA-256 校验 |
| 离线 master key | `data/keys/qwen05b-product-v25-offline` | P/Q、Attention、FFN、噪声重建材料 |
| 净化服务器包 | `data/packages/qwen05b-functional-v25` | `pass=true`，`findings=[]` |
| 产品配置 | `configs/product/qwen05b_v25_functional.yaml` | HF FP32、context 2048、batch 1 |
| 逐层证据 | `artifacts/verification/qwen05b-product-v25-layerwise.json` | 457/460 |
| 200 prompt 证据 | `artifacts/verification/qwen05b-product-v25-greedy200.json` | 200/200 |
| 产品边界证据 | `artifacts/verification/qwen05b-functional-v25/privacy-boundary.json` | 全部检查为 true |
| 真实模型集成测试 | `tests/integration/test_real_private_api.py` | v25 package + online key：1 passed |

### 11.3 v25 实际产品加载命令

```powershell
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = '1'
uv run aloepri inspect-package `
  --server-package data/packages/qwen05b-functional-v25
uv run aloepri serve `
  --config configs/product/qwen05b_v25_functional.yaml
```

另一个终端：

```powershell
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = '1'
uv run aloepri chat `
  --server http://127.0.0.1:8000 `
  --key-dir data/keys/qwen05b-product-v25-online `
  --tokenizer-dir data/models/qwen2.5-0.5b
```

设置 PowerShell UTF-8 是终端显示要求；HTTP 响应 token 和 SDK 解码不依赖控制台代码页。
