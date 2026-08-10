# AloePri 在 Qwen2.5-0.5B 上的复现与工程实现

当前唯一正式目标是 `qwen05b-candidate-v47-best-single`。模型配置为 `h=128`、
`lambda=0.3`、`beta=8`、`alpha_e=0.65`、`alpha_h=0.6`、FP32 权重和 FP64 Attention
计算。现有 checkpoint 的 24 层公式重建、Algorithm 1、Algorithm 2、词表往返、
prefill/decode 输入坐标和 KV Cache 长度检查已经通过。

版本 0.2.0 将攻击实验拆为“可信观测生成 → 不读取目标密钥的攻击 → 独立真值评分”，
覆盖 Gate-IA、修正维度后的 Attention-IA、IMA、Attention/Hidden ISA、TFMA 和 SDA。
正式验收配置为 `configs/acceptance/qwen05b_v47.yaml`，当前报告由原始工件重新计算，
缺失的全量攻击、五项完整精度、产品 100 问、抓包和性能工件直接记为 `NOT_TESTED`，
不会再引用 v15/v29/v31 的结果替代 v47。

直接执行位置和每条命令见 `docs/QWEN05B_V47_EXECUTION.md`；0.2.0 的代码变更和兼容性
说明见 `docs/VERSION_0.2.0.md`。README 后文的 v15/v16/v17/v29/v30/v31 数值仅作为历史
实验记录，不进入 v47 的验收结论。

## 1. 论文要求实现的完整流程

### 1.1 协变混淆目标

设明文模型为 $f_\theta$，输入 token 序列为 $x$，输出 token 为 $y=f_\theta(x)$。论文同时变换输入、权重和输出，要求混淆模型满足：

$$
\widetilde f_{\phi_\Theta(\theta)}\bigl(\phi_X(x)\bigr)
\approx \phi_Y\bigl(f_\theta(x)\bigr).
$$

客户端保存词表置换 $\tau$ 及其逆置换 $\tau^{-1}$。在完整模型上：

$$
\phi_X(x)=\tau(x),\qquad
\phi_Y(y)=\tau(y),\qquad
\psi_Y(\widetilde y)=\tau^{-1}(\widetilde y).
$$

因此服务器接收和返回的都是混淆 token ID；客户端通过 $\tau^{-1}$ 恢复输出。

### 1.2 P/Q 密钥矩阵

论文把隐藏维从 $d$ 扩到 $d+2h$。每个前向边界使用一个宽矩阵 $P\in\mathbb R^{d\times(d+2h)}$ 和右逆 $Q\in\mathbb R^{(d+2h)\times d}$：

$$
PQ=I_d.
$$

相邻模块的权重分别乘 $P$ 和 $Q$，使线性路径中的密钥矩阵相消。实现允许同一 $P$ 对应多个独立右逆 $Q_j$，但每一个都必须满足 $PQ_j=I_d$。`paper_key.safetensors` 保存词表置换、各层矩阵、注意力置换和 FFN 置换；manifest 保存形状、参数和 SHA-256。

### 1.3 Embedding 与 LM Head

论文先按权重自身标准差加入高斯噪声：

$$
W_e^\star=W_e+\alpha_e\mathcal E_e,
\qquad
W_h^\star=W_h+\alpha_h\mathcal E_h.
$$

然后同步应用词表置换和 P/Q 变换：

$$
\widetilde W_e=\Pi W_e^\star P_e,
\qquad
\widetilde W_h=Q_h W_h^\star\Pi^T.
$$

$\Pi$ 是 $\tau$ 的置换矩阵。Embedding 的行和 LM Head 的输出列必须使用同一个 $\Pi$，否则自回归生成不能恢复到原词表。

### 1.4 Attention、GQA 与 RoPE

Qwen2.5-0.5B 使用 GQA。每层 Q/K/V/O 权重按 head 切分；Q head 与其对应的 KV head 使用一致的 head 置换。论文对单组权重打印为：

$$
\widetilde W_q=Q_qW_qR_{qk}H_{qk}Z_{block},
$$

$$
\widetilde W_k=Q_kW_kR_{qk}H_{qk}^{-1}Z_{block}^{T},
$$

$$
\widetilde W_v=Q_vW_vU_{vo},
\qquad
\widetilde W_o=U_{vo}^{-1}W_oP_o.
$$

这里的 Q/K block 置换打印式一般不保持内积。若行向量输出分别为
$q'=qZ$、$k'=kZ^T$，则 $q'k'^T=qZ^2k^T$；只有 $Z^2=I$ 时才等于
$qk^T$。v15 使用勘误后的同侧置换 $q'=qZ$、$k'=kZ$，因此
$q'k'^T=qZZ^Tk^T=qk^T$。完整反例和推导见
`output/pdf/paper_errata/E06_attention_block_permutation_cancellation.pdf`。

实现同时处理：

- Q/K 共享的二维 RoPE 旋转；
- Qwen `rotate_half` 的实际坐标布局；
- GQA 中 Q head 到 KV head 的映射；
- head permutation；
- 2×2 RoPE block permutation；
- prefill、单 token decode 和 KV Cache。

### 1.5 FFN

Qwen 的 SwiGLU FFN 为：

$$
\operatorname{FFN}(x)=
\bigl(\operatorname{SiLU}(xW_{gate})\odot xW_{up}\bigr)W_{down}.
$$

论文的混淆权重为：

$$
\widetilde W_{gate}=Q_{gate}W_{gate}Z_{ffn},
$$

$$
\widetilde W_{up}=Q_{up}W_{up}H_{ffn}^{-1}Z_{ffn},
$$

$$
\widetilde W_{down}=Z_{ffn}^{-1}H_{ffn}W_{down}P_{down}.
$$

同一中间维置换 $Z_{ffn}$ 必须同步用于 gate、up 和 down；$H_{ffn}$ 与其逆矩阵必须成对出现。

### 1.6 RMSNorm 与 Residual

论文打印的 L2 范数比为：

$$
\kappa=\mathbb E\left[\frac{\lVert xP\rVert_2}{\lVert x\rVert_2}\right].
$$

RMSNorm 实际比较的是每维均方根。令原维度为 $d$、扩维后为 $D=d+2h$，则实现使用

$$
\kappa_{\mathrm{RMS}}=
\mathbb E\left[\frac{\operatorname{RMS}(xP)}{\operatorname{RMS}(x)}\right]
=\sqrt{\frac dD}\,
\mathbb E\left[\frac{\lVert xP\rVert_2}{\lVert x\rVert_2}\right].
$$

v29 不再使用单个标量近似。若私有 residual 为 $z=xP$ 且 $PQ=I$，则

$$
\operatorname{RMS}(x)^2=\frac{1}{d}zQQ^Tz^T.
$$

因此 v29 在 checkpoint 内保存派生度量 $G=QQ^T$，精确计算原坐标 RMS。该做法是
论文标量 $\kappa$ 的修正式，不是原文字面实现。v15 的数值由固定 20 条样本和固定种子校准。历史 checkpoint manifest 把该模式记为
`paper-expectation`；这是旧字段名，实际 estimator 是上式的经验 RMS 比，不等同于论文打印的
L2 比。完整推导见 `output/pdf/paper_errata/E07_rmsnorm_kappa_scale.pdf`。实现还保证
attention residual 与 FFN residual 两个分支处于同一坐标系后再相加。

### 1.7 客户端到服务器的真实数据流

论文正文描述的是“混淆 token 先 detokenize 成文本、服务端再 tokenize”的接口。任意 token ID 序列并不保证经过 `decode→encode` 后保持不变，因此本仓库把可信主接口改为 token ID；模型变换公式和 $\tau/\tau^{-1}$ 不变。

1. 客户端加载原始 tokenizer、`tau`、`inverse_tau` 和 key manifest。
2. 客户端把 prompt 编码成明文 token ID：$x=\operatorname{Tokenizer}(prompt)$。
3. 客户端逐 token 计算 $\widetilde x=\tau(x)$。
4. 客户端调用 `POST /v1/private/generate`，发送 `model_id`、`key_id`、`input_ids=\widetilde x` 和生成参数。
5. 服务端校验 `model_id/key_id`，把混淆 token ID 直接送入混淆模型，不重新处理原文。
6. 混淆模型执行 prefill，随后使用 KV Cache 自回归 decode，生成 $\widetilde y$。
7. 非流式接口返回完整 `output_ids`；SSE 接口逐 token 返回混淆 ID。
8. 客户端逐 token 计算 $y=\tau^{-1}(\widetilde y)$。
9. 客户端使用原始 tokenizer 解码 $y$，得到明文响应。

服务端日志不写 prompt 原文、密钥张量或完整 token 流。

## 2. 仓库位置和代码框架

仓库绝对路径：`E:\AloePri`

```text
E:\AloePri
├─ configs/
│  └─ transform/
│     ├─ paper_qwen05b_complete.yaml
│     ├─ paper_qwen05b_ablation_no_noise.yaml
│     └─ paper_qwen05b_engineering_complete.yaml
├─ src/aloepri/
│  ├─ transforms/          # tau、噪声、P/Q、Attention、FFN、RMSNorm
│  ├─ models/              # 自定义 Qwen2 forward、RoPE、KV Cache
│  ├─ conversion/          # 分片转换、partial、断点续跑、manifest
│  ├─ client/              # tokenizer→tau→HTTP/SSE→inverse_tau→decode
│  ├─ serving/             # FastAPI 私有推理接口
│  ├─ attacks/             # VMA 映射与递推基础算子；正式攻击入口在 scripts/
│  └─ eval/                # lm-eval 私有模型接入
├─ scripts/                # 可直接运行的转换、验证和评测入口
├─ tests/                  # 单元与集成测试
├─ data/
│  ├─ models/qwen2.5-0.5b/
│  ├─ checkpoints/
│  └─ keys/
├─ artifacts/              # JSON 结果、攻击缓存、验收证据
├─ docs/
│  ├─ ENGINEERING_IMPLEMENTATION_REGISTER_0.5B.md
│  ├─ PAPER_REPRODUCTION_MISSING_DETAILS.md
│  ├─ AloePri_0.5B_Research_Progress_Report.html
│  └─ paper_errata/
└─ output/pdf/             # 报告 PDF 与逐项论文勘误 PDF
```

论文步骤与实现入口：

| 论文步骤 | 实现文件 | 验证入口 |
|---|---|---|
| 词表置换、逆置换 | `src/aloepri/transforms/vocab.py` | `tests/unit/test_vocab.py` |
| P/Q 与独立右逆族 | `src/aloepri/transforms/paper_key_matrix.py` | `scripts/verify_paper_key_matrix.py` |
| Embedding/Head 噪声 | `src/aloepri/transforms/paper_noise.py` | `tests/unit/test_paper_noise.py` |
| GQA、RoPE、head/block permutation | `src/aloepri/transforms/qwen_structural.py` | `tests/unit/test_qwen_structural.py` |
| Attention/FFN/RMSNorm/Residual | `src/aloepri/conversion/paper_qwen2.py` | `tests/unit/test_paper_qwen2_conversion.py` |
| checkpoint 转换与恢复 | `src/aloepri/conversion/` | `scripts/verify_paper_qwen2_checkpoint.py` |
| HF forward/generate/KV Cache | `src/aloepri/models/` | `scripts/verify_paper_qwen2_checkpoint.py` |
| 客户端 SDK | `src/aloepri/client/sdk.py` | `tests/unit/test_client_sdk.py` |
| FastAPI 与 SSE | `src/aloepri/serving/app.py` | `scripts/verify_private_api_local.py` |
| VMA/PUPA | `src/aloepri/attacks/mapping.py` | `scripts/run_vma_pupa.py` |
| MMLU/C-Eval/PIQA | `scripts/run_lm_eval.py` | `scripts/compare_lm_eval.py` |
| TTFT/TPOT/显存 | `scripts/benchmark_hf.py` | `scripts/compare_performance.py` |

完整逐张量工程登记见 `docs/ENGINEERING_IMPLEMENTATION_REGISTER_0.5B.md`。

## 3. 环境配置

### 3.1 Windows PowerShell

```powershell
cd E:\AloePri
uv sync --frozen
$env:PYTHONPATH = "E:\AloePri\src"
$env:HF_DATASETS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"
.\.venv\Scripts\python.exe scripts\audit_env.py
```

主要环境：Python 3.11、PyTorch CUDA、Transformers、Accelerate、lm-eval、FastAPI、SacreBLEU。锁定版本在 `uv.lock`。

### 3.2 低显存运行规则

本机为 6 GiB GPU。正式攻击使用以下策略：

- 原始和混淆的全词表矩阵常驻 CPU；
- 只把当前 query/candidate 分块搬到 GPU；
- `query_batch_size=64`；
- `candidate_batch_size=256`；
- 每个 layer/weight-combination 单独缓存，异常后从已有 `.pt` 继续；
- VMA cache manifest 绑定 checkpoint、key、query/candidate ID、算法版本及每个 `.pt` 的 SHA-256；
- ISA 先运行私有模型并把目标 hidden state 搬回 CPU，释放后才加载明文模型；
- HF 性能基准每个进程只加载一个模型，明文进程退出后才启动私有模型；
- 不并行启动两个 GPU 评测。
- 本机已测最高 PyTorch 分配量为 1,748,945,408 bytes；6 GiB GPU 实测最低仍有约 2.37 GiB 空闲。
- 后续正确性实验优先在命令前设置 `$env:CUDA_VISIBLE_DEVICES = "-1"` 强制 CPU，执行后用 `Remove-Item Env:CUDA_VISIBLE_DEVICES` 恢复；CPU 模式更慢，但不占用模型侧显存。
- 启动 CUDA 实验前要求 `nvidia-smi` 显示至少 4 GiB 空闲；不满足时等待其他进程退出或改用 CPU，禁止并行抢占。

## 4. 从原始模型生成三个 0.5B checkpoint

### 4.1 论文数值超参数的 corrected-paper v15

参数：$\alpha_e=1.0$、$\alpha_h=0.2$、$h=128$、$\lambda=0.3$、$\beta=8$、$\gamma=1000$、BF16。

先校准 RMSNorm。该命令的 20 条 prompt 产生 822 个 token hidden-state 样本；输出记录 prompt、原模型、key、脚本和 `uv.lock` 的 SHA-256。

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_paper_rms.py `
  --source data\models\qwen2.5-0.5b `
  --key-seed 20260803 --expansion-h 128 --lambda 0.3 `
  --prompts configs\eval\benchmark_prompts_20.json `
  --dtype bfloat16 `
  --out artifacts\calibration\qwen05b-paper-seed20260803-rms-expectation-bf16-20.json
```

校准脚本按转换器的相同参数确定性生成 P，因此不依赖最终 key 目录。现有 v15 的 49 个 κ 值已用上面命令重新计算，和转换时使用的值逐项完全相同（最大绝对差 0）。

```powershell
.\.venv\Scripts\python.exe scripts\convert_paper_qwen2_checkpoint.py `
  --source data\models\qwen2.5-0.5b `
  --output data\checkpoints\qwen2.5-0.5b-paper-v15-complete-bf16 `
  --key-dir data\keys\dev-qwen05b-paper-v15-complete-bf16 `
  --h 128 --lambda 0.3 --seed 20260803 `
  --embedding-noise-seed 31005 --head-noise-seed 41005 `
  --alpha-e 1.0 --alpha-h 0.2 --dtype bfloat16 --max-shard-size 1GB `
  --model-id qwen2.5-0.5b-paper-v15-complete-bf16 `
  --key-id dev-qwen05b-paper-v15-complete-bf16 `
  --rms-calibration artifacts\calibration\qwen05b-paper-seed20260803-rms-expectation-bf16-20.json `
  --kappa-mode paper-expectation --algorithm2 `
  --ffn-scale-min 0.5 --ffn-scale-max 2.0 `
  --block-beta 8 --sampling-gamma 1000 `
  --blockperm-mode gamma-corrected --rope-frequency-mode qwen-actual `
  --qk-scale-min 0.5 --qk-scale-max 2.0
```

该命令必须在输出目录不存在时执行；转换器先写 `.partial`，成功后再原子重命名。这里的
`paper-expectation` 是为复现现有 v15 manifest 保留的历史 CLI 名称，数值含义按 1.6 节解释。

输出：

```text
data/checkpoints/qwen2.5-0.5b-paper-v15-complete-bf16
data/keys/dev-qwen05b-paper-v15-complete-bf16
```

### 4.2 无噪声消融 v17

除 $\alpha_e=\alpha_h=0$ 外，其余结构与 v15 相同。

```powershell
.\.venv\Scripts\python.exe scripts\convert_paper_qwen2_checkpoint.py `
  --source data\models\qwen2.5-0.5b `
  --output data\checkpoints\qwen2.5-0.5b-paper-v17-no-noise-bf16 `
  --key-dir data\keys\dev-qwen05b-paper-v17-no-noise-bf16 `
  --h 128 --lambda 0.3 --seed 20260803 `
  --embedding-noise-seed 31005 --head-noise-seed 41005 `
  --alpha-e 0 --alpha-h 0 --dtype bfloat16 --max-shard-size 1GB `
  --model-id qwen2.5-0.5b-paper-v17-no-noise-bf16 `
  --key-id dev-qwen05b-paper-v17-no-noise-bf16 `
  --rms-calibration artifacts\calibration\qwen05b-paper-seed20260803-rms-expectation-bf16-20.json `
  --kappa-mode paper-expectation --algorithm2 `
  --ffn-scale-min 0.5 --ffn-scale-max 2.0 `
  --block-beta 8 --sampling-gamma 1000 `
  --blockperm-mode gamma-corrected --rope-frequency-mode qwen-actual `
  --qk-scale-min 0.5 --qk-scale-max 2.0
```

### 4.3 低噪声工程参数 v16

参数：$\alpha_e=0.1$、$\alpha_h=0.01$、$\beta=1$、FP32；用于测量 0.5B 上的精度与隐私取舍，不替代 v15 corrected-paper 分支。

```powershell
.\.venv\Scripts\python.exe scripts\convert_paper_qwen2_checkpoint.py `
  --source data\models\qwen2.5-0.5b `
  --output data\checkpoints\qwen2.5-0.5b-paper-v16-engineering-complete-fp32 `
  --key-dir data\keys\dev-qwen05b-paper-v16-engineering-complete-fp32 `
  --h 128 --lambda 0.3 --seed 20260803 `
  --embedding-noise-seed 31005 --head-noise-seed 41005 `
  --alpha-e 0.1 --alpha-h 0.01 --dtype float32 --max-shard-size 1GB `
  --model-id qwen2.5-0.5b-paper-v16-engineering-complete-fp32 `
  --key-id dev-qwen05b-paper-v16-engineering-complete-fp32 `
  --kappa-mode covariant-rms --algorithm2 `
  --ffn-scale-min 0.5 --ffn-scale-max 2.0 `
  --block-beta 1 --sampling-gamma 1000 `
  --blockperm-mode gamma-corrected --rope-frequency-mode qwen-actual `
  --qk-scale-min 1.0 --qk-scale-max 1.0 --uvo-condition-max 100
```

## 5. 可复现验证命令

### 5.1 P/Q、manifest、forward、generate 与 KV Cache

```powershell
.\.venv\Scripts\python.exe scripts\verify_paper_key_matrix.py `
  --key-dir data\keys\dev-qwen05b-paper-v15-complete-bf16

.\.venv\Scripts\python.exe scripts\verify_manifest.py `
  data\checkpoints\qwen2.5-0.5b-paper-v15-complete-bf16

.\.venv\Scripts\python.exe scripts\verify_paper_qwen2_checkpoint.py `
  --original data\models\qwen2.5-0.5b `
  --private data\checkpoints\qwen2.5-0.5b-paper-v15-complete-bf16 `
  --key-dir data\keys\dev-qwen05b-paper-v15-complete-bf16 `
  --dtype bfloat16 `
  --out artifacts\verification\paper-qwen05b-v15-complete-bf16.json
```

### 5.2 私有 API 完整流程

```powershell
.\.venv\Scripts\python.exe scripts\verify_private_api_local.py `
  --source data\models\qwen2.5-0.5b `
  --private data\checkpoints\qwen2.5-0.5b-paper-v15-complete-bf16 `
  --key-dir data\keys\dev-qwen05b-paper-v15-complete-bf16 `
  --out artifacts\api\paper-qwen05b-v15-complete-bf16.json
```

该命令同时检查 HTTP 200、SSE 与非流式结果一致、错误 key 返回 400，以及客户端逆置换后可解码。

### 5.3 PIQA、MMLU、C-Eval 与配对置信区间

PIQA：

```powershell
.\.venv\Scripts\python.exe scripts\run_lm_eval.py `
  --model data\models\qwen2.5-0.5b `
  --tokenizer data\models\qwen2.5-0.5b `
  --tasks piqa --batch-size 4 --dtype bfloat16 `
  --out artifacts\accuracy\piqa-plaintext-bf16-full.json

.\.venv\Scripts\python.exe scripts\run_lm_eval.py `
  --model data\checkpoints\qwen2.5-0.5b-paper-v15-complete-bf16 `
  --tokenizer data\models\qwen2.5-0.5b `
  --key data\keys\dev-qwen05b-paper-v15-complete-bf16\paper_key.safetensors `
  --tasks piqa --batch-size 4 --dtype bfloat16 `
  --out artifacts\accuracy\piqa-paper-v15-complete-bf16-full.json

.\.venv\Scripts\python.exe scripts\compare_lm_eval.py `
  --baseline artifacts\accuracy\piqa-plaintext-bf16-full.json `
  --candidate artifacts\accuracy\piqa-paper-v15-complete-bf16-full.json `
  --metric acc_norm,none --bootstrap-resamples 10000 `
  --out artifacts\accuracy\piqa-paper-v15-complete-bf16-comparison.json
```

MMLU 使用 6 个任务 shard；每个 shard 完成后保存，最后合并：

```powershell
.\.venv\Scripts\python.exe scripts\merge_lm_eval_shards.py `
  --inputs artifacts\eval\mmlu-v15-shard-00.json artifacts\eval\mmlu-v15-shard-01.json `
           artifacts\eval\mmlu-v15-shard-02.json artifacts\eval\mmlu-v15-shard-03.json `
           artifacts\eval\mmlu-v15-shard-04.json artifacts\eval\mmlu-v15-shard-05.json `
  --out artifacts\eval\mmlu-local-paper-v15-complete-bf16-full.json
```

C-Eval 和 MMLU 的比较同样调用 `compare_lm_eval.py`，统一使用逐题配对、10,000 次非参数 bootstrap 的 95% 置信区间。

### 5.4 PUPA VMA 正式实验

```powershell
$layers = 0..23
.\.venv\Scripts\python.exe scripts\run_vma_pupa.py `
  --original data\models\qwen2.5-0.5b `
  --private data\checkpoints\qwen2.5-0.5b-paper-v15-complete-bf16 `
  --key-dir data\keys\dev-qwen05b-paper-v15-complete-bf16 `
  --candidate-sizes 16384 `
  --layers $layers `
  --query-batch-size 64 `
  --candidate-batch-size 256 `
  --stream-known-from-cpu `
  --prediction-cache-dir artifacts\privacy\cache\v15-c16384 `
  --out artifacts\privacy\vma-pupa-paper-v15-complete-bf16-c16384.json
```

数据口径：237 条 PUPA `user_query`，62,634 个 token occurrence，664 个 PII 单元；候选集含全部 10,749 个查询/PII token ID 和确定性抽样的词表干扰项，总计 16,384 个候选。Dense Qwen 测试论文 Table 9 的 `We/Wh`、`We/Wq·We/Wk^T`、`We/Wgate`、`We/Wup`、`Wdown/Wh` 五种组合；MoE Router 不适用于该模型。

指标定义：

$$
\operatorname{TTRSR}=
\frac{\text{正确恢复的 token occurrence 数}}
{\text{PUPA user\_query token occurrence 总数}},
$$

$$
\operatorname{PIIRSR}=
\frac{\text{完全恢复正确的 PII 单元数}}
{\text{PII 单元总数}}.
$$

BLEU-4 使用 SacreBLEU 对恢复文本和原文做 corpus BLEU；CosSim 使用 Qwen 原始输入 embedding 对每条文本做 mean pooling 后计算平均余弦相似度。已有缓存只能在 cache manifest 的输入指纹和文件 SHA-256 全部一致时复用；旧缓存必须显式使用 `--bind-existing-unversioned-cache` 绑定一次，正常新运行不得使用该选项。

### 5.5 Gate/Attention IA、IMA、ISA 与 SDA

```powershell
$layers = 0..23
.\.venv\Scripts\python.exe scripts\run_gate_ia.py `
  --original data\models\qwen2.5-0.5b `
  --private data\checkpoints\qwen2.5-0.5b-paper-v15-complete-bf16 `
  --key-dir data\keys\dev-qwen05b-paper-v15-complete-bf16 `
  --sample 512 --layers $layers `
  --out artifacts\privacy\gate-ia-paper-v15-complete-bf16-full-vocab.json

.\.venv\Scripts\python.exe scripts\run_attn_ia.py `
  --original data\models\qwen2.5-0.5b `
  --private data\checkpoints\qwen2.5-0.5b-paper-v15-complete-bf16 `
  --key-dir data\keys\dev-qwen05b-paper-v15-complete-bf16 `
  --sample 256 --layers 0 8 16 23 `
  --out artifacts\privacy\attn-ia-paper-v15-complete-bf16-proxy.json

.\.venv\Scripts\python.exe scripts\train_ima_smoke.py `
  --original data\models\qwen2.5-0.5b `
  --private data\checkpoints\qwen2.5-0.5b-paper-v15-complete-bf16 `
  --key-dir data\keys\dev-qwen05b-paper-v15-complete-bf16 `
  --steps 2000 --sample 8192 `
  --out artifacts\privacy\ima-paper-v15-complete-bf16.json

.\.venv\Scripts\python.exe scripts\run_isa_hidden_state.py `
  --original data\models\qwen2.5-0.5b `
  --private data\checkpoints\qwen2.5-0.5b-paper-v15-complete-bf16 `
  --key-dir data\keys\dev-qwen05b-paper-v15-complete-bf16 `
  --prompts configs\eval\prompts.json --layer 12 `
  --sequence-length 16 --steps 100 --max-prompts 20 `
  --learning-rate 0.05 `
  --out artifacts\privacy\isa-paper-v15-complete-bf16.json

.\.venv\Scripts\python.exe scripts\run_tfma_curve.py `
  --tokenizer data\models\qwen2.5-0.5b `
  --key-dir data\keys\dev-qwen05b-paper-v15-complete-bf16 `
  --prompts data\eval\mmlu-test-prompts.json `
  --exposures 100 1000 10000 100000 `
  --out artifacts\privacy\tfma-paper-v15-complete-bf16.json

.\.venv\Scripts\python.exe scripts\train_sda_smoke.py `
  --tokenizer data\models\qwen2.5-0.5b `
  --prompts data\eval\mmlu-test-prompts.json `
  --steps 2000 --batch-size 4 --max-len 64 `
  --train-limit 10000 --test-limit 2000 `
  --out artifacts\privacy\sda-paper-v15-complete-bf16.json
```

Attention-IA 和 ISA 固定写入 `paper_exact=false`。ISA 的 token-Gram 目标只对正交右变换不变，而本实现的 P 是一般矩形矩阵，因此它不能证明与论文同维 ISA 等价。SDA 使用 MMLU 文本，不等同于论文未公开的医疗语料。这三个 JSON 只作为工程诊断，不通过论文精确协议门禁。

### 5.6 HF 20×100 性能

```powershell
powershell -ExecutionPolicy Bypass `
  -File scripts\run_balanced_hf_performance.ps1 `
  -BaselineModel data\models\qwen2.5-0.5b `
  -CandidateModel data\checkpoints\qwen2.5-0.5b-paper-v15-complete-bf16 `
  -Tokenizer data\models\qwen2.5-0.5b `
  -Prompts configs\eval\benchmark_prompts_20.json `
  -CandidateKey data\keys\dev-qwen05b-paper-v15-complete-bf16\paper_key.safetensors `
  -OutputDir artifacts\performance\hf-balanced-v15-v2 `
  -MaxNewTokens 100 -Warmup 2
```

脚本顺序运行 8 个独立进程，角色次序固定为 ABBA+BAAB。四对同一 `run_id`、同一 prompt 的请求汇总为 80 对观测，再做 10,000 次配对 bootstrap。每个运行 artifact 在执行时写入模型、key、prompt、tokenizer 文件、benchmark 脚本、BF16、CUDA 环境和运行序号的 SHA-256/身份信息；缺少运行时 tokenizer 指纹的旧 artifact 会被比较器拒绝。这是本机 HF 单并发口径，不是 vLLM 服务口径。

### 5.7 代码质量与真实模型集成

```powershell
uv run ruff check src scripts tests
uv run mypy src
uv run pytest -q
$env:ALOEPRI_RUN_MODEL_TESTS = "1"
uv run pytest -q tests\integration\test_real_private_api.py
```

## 6. 已完成的工程功能

| 功能 | 当前状态 | 直接证据 |
|---|---|---|
| tau/inverse_tau 全词表双射 | 已实现 | key manifest、单元测试 |
| Embedding 与 LM Head 同步置换 | 已实现 | checkpoint 张量、API round-trip |
| 论文标准差高斯噪声 | 已实现 | v15/v16/v17 独立 checkpoint |
| $d\rightarrow d+2h\rightarrow d$ 与独立右逆 | 已实现 | `PQ_j` 数值误差验证 |
| Attention Q/K/V/O、GQA、RoPE | 已实现 | 逐层测试、prefill/decode |
| head/block permutation | 已实现 | key 文件和结构测试 |
| FFN、RMSNorm、Residual | 已实现 | 模块测试和整模 forward |
| checkpoint 分片、partial、续跑、SHA-256 | 已实现 | converter 与 manifest |
| HF forward/generate/KV Cache | 已实现 | v15 验证 JSON |
| token-ID API、SSE、客户端逐 token 恢复 | 已实现 | API 验证 JSON |
| MMLU/C-Eval/PIQA 与配对区间 | 已运行 | accuracy JSON |
| IFEval 541 prompt / 834 instruction 全量评分 | 已运行 | 四个 strict/loose 指标及配对区间 |
| HumanEval 164 题生成与隔离执行 | 已运行 | 逐题 pass/fail、pass@1 及配对区间 |
| PUPA VMA 五种 Dense 组合 | 已运行 | privacy JSON 与逐层缓存 |
| Gate-IA、Attention-IA 代理 | 已运行 | Gate 512/24 层；Attention 256/4 层 |
| IMA、ISA、TFMA、SDA | 已运行 | 正式规模或明确代理口径 JSON |
| HF 20×100 TTFT/TPOT | 已运行 | ABBA+BAAB，8 个独立进程，80 个配对请求、100 token、BF16 |

## 7. 实验结果

### 7.1 corrected-paper v15 的精度

下表把论文 Table 3 的 Qwen3-14B 原始数据、甲方 PPT 第 7 页目标和本机 0.5B 实测放在同一行。论文没有给 Qwen2.5-0.5B 数据，因此论文列只作为原论文结果参考，不和 0.5B 绝对分数混算。

| 数据集 | 论文 Qwen3-14B 明文→AloePri | 论文变化 | PPT 目标 | 0.5B 明文 | 0.5B v15 | 0.5B 变化与 95% CI | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| MMLU `acc_norm`，n=14,042 | 87.64%→84.57% | -3.07 pp | 下降≤3.5 pp | 34.475% | 27.261% | -7.214 pp，[-8.049, -6.386] | 未达到 |
| C-Eval `acc`，n=1,346 | 87.35%→87.12% | -0.23 pp | 下降≤3.5 pp | 52.972% | 26.003% | -26.969 pp，[-30.314, -23.551] | 未达到 |
| PIQA `acc_norm`，n=1,838 | 89.72%→90.26% | +0.54 pp | 下降≤3.5 pp | 70.239% | 58.868% | -11.371 pp，[-13.765, -9.032] | 未达到 |

生成任务使用完整测试集、BF16、batch=4 和确定性解码。论文 Table 3 只给一个 IFEval 总指标，没有给 strict/loose 的具体聚合口径；因此仅在 IFEval prompt strict 行并列论文原值，其余三个本机分项不强行映射到论文单列。

| 数据集/指标 | 论文 Qwen3-14B 明文→AloePri | 论文变化 | PPT 目标 | 0.5B 明文 | 0.5B v15 | 0.5B 变化与 95% CI | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| IFEval prompt strict，n=541 | 86.14%→85.40% | -0.74 pp | 下降≤3.5 pp | 22.181% | 7.579% | -14.603 pp，[-18.299, -10.906] | 未达到 |
| IFEval instruction strict，n=834 | 论文未分项 | - | 下降≤3.5 pp | 35.731% | 17.266% | -18.465 pp，[-22.062, -14.868] | 未达到 |
| IFEval prompt loose，n=541 | 论文未分项 | - | 下降≤3.5 pp | 25.878% | 9.797% | -16.081 pp，[-20.148, -12.015] | 未达到 |
| IFEval instruction loose，n=834 | 论文未分项 | - | 下降≤3.5 pp | 39.448% | 21.103% | -18.345 pp，[-22.302, -14.508] | 未达到 |
| HumanEval pass@1，n=164 | 94.51%→95.12% | +0.61 pp | 下降≤3.5 pp | 25.610%（42/164） | 0%（0/164） | -25.610 pp，[-32.317, -18.902] | 未达到 |

置信区间来自逐题明文/混淆正确性差值的 10,000 次配对非参数 bootstrap。全部基准的明文和 v15 都使用 BF16。MMLU/C-Eval/PIQA 的历史原始 JSON 未在运行时写入模型和 key 文件指纹，因此其 comparison 为 `formal_run_binding=false`。本轮 IFEval 与 HumanEval 从生成开始重新运行，comparison 已逐层绑定模型、tokenizer、key、数据集或文档 hash、dtype、batch、脚本、运行时和上下游 JSON 的 SHA-256，`formal_run_binding=true`。

正式生成命令：

```powershell
.\scripts\run_ifeval_v15_formal_pipeline.ps1
.\scripts\run_humaneval_v15_formal_pipeline.ps1
```

### 7.2 0.5B 噪声消融

PIQA 在结构不变时的结果：

| 配置 | $\alpha_e$ | $\alpha_h$ | dtype | PIQA 明文 | 候选 | 变化与 95% CI |
|---|---:|---:|---|---:|---:|---:|
| v15 corrected-paper | 1.0 | 0.2 | BF16 | 70.239% | 58.868% | -11.371 pp，[-13.765, -9.032] |
| v17 无噪声 | 0 | 0 | BF16 | 70.239% | 69.042% | -1.197 pp，[-2.448, +0.054] |
| v16 低噪声工程参数 | 0.1 | 0.01 | FP32 | 70.131% | 70.239% | +0.109 pp，[-0.490, +0.707] |

这组实验在同一个 0.5B 模型内控制了结构和 dtype：v15 对 v17 的主要变量是 Embedding/Head 噪声。它能证明论文数值噪声是本次精度下降的主要来源；仅凭单一参数规模不能证明“参数量小”是原因。要检验参数量因果关系，必须固定转换算法、噪声定义、数据集和 dtype，增加至少一个更大模型作为对照。

### 7.3 corrected-paper v15 的生成与 API

| 项目 | 实测 |
|---|---:|
| prefill cache length | 明文 36；混淆 36 |
| 单步 decode 后 cache length | 37 |
| prefill top-1 agreement | 19.44% |
| greedy token 序列完全一致 | 否 |
| 非流式 API | HTTP 200 |
| SSE 与非流式 output ID | 完全一致 |
| 错误 key | HTTP 400 |
| 客户端逆置换后输出 | 可解码、非空 |

生成不一致与上面的论文数值噪声精度下降一致；API 请求与恢复流程已通过。

### 7.4 0.5B 的精度与隐私取舍

v16 低噪声使 PIQA 达到精度目标，但同一套 16,384 候选、24 层 VMA 显示三个主要组合可恢复几乎全部 token。

| 权重组合 | v15 论文数值噪声 TTRSR | v16 低噪声 TTRSR（Wilson 95% CI） | v16 PIIRSR | v16 BLEU-4 | v16 CosSim |
|---|---:|---:|---:|---:|---:|
| `We/Wh` | 10.118% | 98.584% [98.488, 98.673] | 88.102% | 91.065 | 0.9889 |
| `We/(Wq·Wk^T)` | 0.065% | 99.441% [99.380, 99.497] | 98.042% | 96.757 | 0.9978 |
| `We/Wgate` | 9.204% | 100.000% [99.994, 100] | 99.699% | 99.981 | 1.0000 |
| `We/Wup` | 2.708% | 0.187% [0.156, 0.224] | 0% | ≈0 | -0.3705 |
| `Wdown/Wh` | 10.588% | 8.427% [8.212, 8.647] | 0% | 0.000635 | -0.2915 |

因此 v16 只能作为精度工程消融，不能作为隐私候选。0.5B 当前两个端点是：v15 隐私达标但精度未达标；v16 精度达标但主要 VMA 组合未达标。

### 7.5 corrected-paper v15 的 VMA

论文 Table 3 的 Qwen3-14B VMA 为 TTRSR 25.05%、PIIRSR 1.62%、BLEU-4 1.72；PPT 目标为 TTRSR≤15%（≤5% 为优秀）、PIIRSR≤3%、BLEU-4≤2.5、CosSim≤0.5。0.5B v15 使用一个变换 key、16,384 个候选和全部 24 层。

| 权重组合 | 论文 Qwen3-14B VMA | PPT 目标 | 0.5B v15 TTRSR（Wilson 95% CI） | PIIRSR（Wilson 95% CI） | BLEU-4 | CosSim | 判定 |
|---|---|---|---:|---:|---:|---:|---|
| `We/Wh` | TTRSR 25.05%；PIIRSR 1.62%；BLEU-4 1.72 | TTRSR≤15%；PIIRSR≤3%；BLEU≤2.5；CosSim≤0.5 | 10.118% [9.884, 10.356] | 0.151% [0.027, 0.848] | 0.000057 | -0.3340 | 达到目标 |
| `We/(Wq·Wk^T)` | 同上 | 同上 | 0.065% [0.048, 0.089] | 0% [0, 0.575] | 0.000000001 | -0.3665 | 达到目标 |
| `We/Wgate` | 同上 | 同上 | 9.204% [8.980, 9.433] | 0.151% [0.027, 0.848] | 0.000996 | -0.3305 | 达到目标 |
| `We/Wup` | 同上 | 同上 | 2.708% [2.584, 2.838] | 0% [0, 0.575] | ≈0 | -0.3696 | 达到目标 |
| `Wdown/Wh` | 同上 | 同上 | 10.588% [10.350, 10.832] | 0% [0, 0.575] | 0.005385 | -0.2679 | 达到目标 |

Wilson 区间是 occurrence/PII 单元层面的描述性区间。token occurrence 来自同一批文本且共享一个 key，不满足完全独立同分布；该区间不表示文本聚类波动或不同随机 key 之间的波动。v15 已在全新空缓存上重算 97 个预测文件，耗时 1,346.05 秒，GPU 峰值 allocated 为 1,261,535,744 bytes，reserved 为 1,858,076,672 bytes。cache manifest 的 `bound_existing_unversioned_cache=false`，并绑定模型、key、数据、候选集、24 层、五个组合、入口脚本和每个预测文件的 SHA-256。

### 7.6 同分布 TFMA 曲线与直接权重匹配

TFMA 使用 MMLU 14,042 条 test prompt 同时构造先验频率和观测流，测量同分布频率攻击；这不是论文未公开的医疗语料协议。

| 观测 token | 观测 unique token | Top-1 | Top-10 | Top-100 |
|---:|---:|---:|---:|---:|
| 100 | 52 | 3.846% | 13.462% | 34.615% |
| 1,000 | 231 | 1.299% | 4.762% | 14.719% |
| 10,000 | 2,256 | 0.133% | 1.020% | 5.142% |
| 100,000 | 9,840 | 0.071% | 0.407% | 2.063% |

原始 Embedding 的直接行匹配不适用于论文扩维 checkpoint：明文矩阵形状为 `151936×896`，v15 为 `151936×1152`。脚本现在输出 `applicable=false`，不再把维度错误解释为恢复率 0。需要先通过论文 Table 9 的权重组合消去隐藏维变换，再执行 VMA。

### 7.7 其余攻击

| 攻击 | 论文原始数据 | PPT/本项目门槛 | 0.5B v15 实测 | 判定 |
|---|---|---|---|---|
| Gate-IA | 未给 0.5B 数值 | Top-1≤15% | 512 token、24 层；Top-1 17.383%，Top-10 46.094% | 未达到 |
| Attention-IA | 未给 0.5B 数值 | Top-1≤15% | 256 token、4 层；代理 Top-1 7.031%，Top-10 27.344% | 仅诊断；非论文精确协议 |
| IMA | 未给 0.5B 数值 | TTRSR≤15% | 8192 token、2000 步；恢复率 0% | 数值达到；协议等价性未确认 |
| ISA | 未给 0.5B 数值 | TTRSR≤15% | 20 prompt、125 token、100 步；TTRSR 0% | 仅诊断；token-Gram 对一般矩形 P 不构成论文等价不变量 |
| TFMA | 未给 0.5B 数值 | Top-10≤20% | 最大 Top-10 13.462%；100,000 token 时 0.407% | 当前同分布口径达到 |
| SDA | 论文报告 BLEU-4，但医疗语料未公开 | BLEU-4≤2.5 | MMLU 10,000/2,000；token accuracy 17.664%；BLEU-4 13.120 | 未达到；语料非论文协议 |

IMA 的 `paper_exact_scale_claimed=false` 表示论文没有公开足以证明 surrogate-key 训练分布等价的配置；它不会因为恢复率为 0 就自动通过论文精确验收。ISA 固定输出 `paper_exact=false`，不进入论文精确通过项。ISA 的两个模型不同时加载：私有阶段峰值 1,633,619,456 bytes，优化阶段峰值 1,551,377,408 bytes。

### 7.8 HF 性能与显存

| 指标 | 明文 0.5B | v15 | 相对变化与 95% CI | PPT 目标 | 判定 |
|---|---:|---:|---:|---:|---|
| TTFT p50 | 38.466 ms | 38.545 ms | +0.204% [-1.858%, +3.173%] | 劣化≤15%且 CI 上界≤15% | 达到 |
| TTFT p95 | 42.675 ms | 45.486 ms | +6.588% [-4.782%, +22.790%] | 劣化≤15%且 CI 上界≤15% | 未达到 |
| TTFT p99 | 46.646 ms | 67.359 ms | +44.404% [-2.515%, +186.199%] | 劣化≤15%且 CI 上界≤15% | 未达到 |
| TPOT p50 | 36.120 ms | 36.579 ms | +1.270% [+0.709%, +1.980%] | 劣化≤15%且 CI 上界≤15% | 达到 |
| TPOT p95 | 37.676 ms | 38.646 ms | +2.574% [-0.331%, +5.653%] | 劣化≤15%且 CI 上界≤15% | 达到 |
| TPOT p99 | 38.747 ms | 39.399 ms | +1.684% [-1.316%, +5.162%] | 劣化≤15%且 CI 上界≤15% | 达到 |
| 每进程吞吐中位数 | 8.640 token/s | 10.392 token/s | +20.28% | 未规定 | 记录值 |
| 加载时间中位数 | 2.648 s | 5.001 s | +88.86% | 未规定 | 记录值 |
| 推理峰值 allocated 最大值 | 1,023,297,536 B | 1,649,150,976 B | +625,853,440 B | 6 GiB 本机容量 | 模型侧约留 4.46 GiB |

结果来自重新执行的 ABBA+BAAB 8 个独立进程和 80 对请求。TPOT 三个分位和 TTFT p50 通过；TTFT p95 因置信区间上界为 22.790% 未通过，TTFT p99 点估计为 44.404% 且区间很宽。验收按点估计和 95% CI 上界同时不超过 15% 判定。本轮监控到整卡最少仍约有 2.8 GiB 空闲；表中的 4.46 GiB 是相对 6 GiB 容量扣除 PyTorch 模型峰值后的模型侧余量，不包含桌面进程占用。结果不外推到并发、vLLM 或其他 GPU。

## 8. 论文未提供、无法从原文唯一确定的实验细节

这些缺项的逐条位置、影响和本仓库固定口径记录在 `docs/PAPER_REPRODUCTION_MISSING_DETAILS.md`：

- 论文实验 checkpoint、转换脚本和服务端部署代码；
- 每个实验使用的随机种子和 key 数量；
- Q/K/V/O 是否共享同一 P/Q 实例；
- RMSNorm 的 $\kappa$ 采样数据、样本数和随机种子；
- VMA 候选集合、距离、层投票冲突处理；
- TTRSR、PIIRSR、BLEU-4、CosSim 的完整计算口径；
- IMA、ISA、TFMA、SDA 的训练配置；
- 性能测试的 GPU、节点、并发调度和分位数定义。

论文公式中确认的每个独立问题均有单独推导文档和 PDF，见 `docs/paper_errata/README.md` 与 `output/pdf/paper_errata/`。

## 9. 不能按论文精确口径确认或尚未运行的项目

以下项目不出现在“已测试结果”表：

| 项目 | 后续运行入口 |
|---|---|
| 论文医疗语料 TFMA/SDA | 已运行 MMLU 代理；论文医疗语料和切分未公开，不能声称论文精确协议 |
| 论文精确 Attention-IA/IMA/SDA | 工程代理已运行；论文缺少维度适配和训练分布配置 |
| vLLM 正式性能 | WSL2 环境中的 `scripts/benchmark_vllm_streaming.py` |

## 10. 结果

0.5B 上已经完成 corrected-paper 模型变换、checkpoint、HF 推理、KV Cache、客户端/服务端 token-ID 请求与恢复、MMLU、C-Eval、PIQA、IFEval、HumanEval、空缓存 PUPA VMA、Gate-IA、Attention-IA 代理、IMA、ISA 诊断、TFMA、SDA 代理和 ABBA+BAAB HF 性能测试。最终验收共有 57 项检查，41 项通过、16 项失败、缺失证据 0 项，结论为 `NO-GO`。IFEval 四个指标下降 14.603-18.465 个百分点；HumanEval 从 42/164 降到 0/164。样本输出显示乱码、重复 token 和非 Python 文本；这与 checkpoint 验证得到的 19.44% prefill top-1 agreement 一致，不是 WSL 执行器或 `inverse_tau` 漏恢复造成。无噪声 v17 和低噪声 v16 的 PIQA 消融表明，论文数值噪声是本次 0.5B 精度下降的主要来源。是否由参数量导致，现有单模型实验不能给出因果结论。
