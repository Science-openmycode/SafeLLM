# Qwen2.5-0.5B AloePri 复现与产品化实施状态

更新日期：2026-08-09  
模型：`Qwen2.5-0.5B-Instruct`  
设备：NVIDIA GeForce RTX 3060 Laptop GPU，6 GiB  
当前发布判定：`NO-GO`  
当前可用工件：无噪声功能演示包 `qwen05b-functional-v25`

## 1. 论文方法要执行的完整数据流

### 1.1 客户端输入

客户端保留原始 tokenizer、对话历史、词表置换 $\tau$ 和逆置换
$\tau^{-1}$。用户输入消息 $m$ 后，客户端先执行 Qwen Chat Template：

$$
x=\operatorname{Tokenizer}\bigl(\operatorname{ChatTemplate}(m,\text{history})\bigr).
$$

默认 `privacy_mode=permutation` 时直接置换 token ID：

$$
\widetilde{x}_i=\tau(x_i).
$$

可选 `privacy_mode=rmdp` 时，先使用 M1 指数机制得到 $x'_i$，再置换：

$$
\Pr[x'_i=j\mid x_i]
=\frac{\exp\!\left(-\epsilon_1 d(x_i,j)/2\right)}
{\sum_{k\in\mathcal V}\exp\!\left(-\epsilon_1 d(x_i,k)/2\right)},
\qquad
\widetilde{x}_i=\tau(x'_i).
$$

本实现按论文单 token 置换距离取 $d(i,i)=0$、$d(i,j)=2$。因此原 token
权重为 1，其余每个词表项权重为 $e^{-\epsilon_1}$。长序列采用逐 token
组合，并在客户端保存本地隐私账本。

### 1.2 词表、Embedding 与 LM Head

设置换矩阵为 $\Pi$，明文 Embedding 和 Head 为 $W_e,W_h$。加入论文噪声后：

$$
W_e^\star=W_e+\alpha_e\mathcal E_e,
\qquad
W_h^\star=W_h+\alpha_h\mathcal E_h.
$$

模型转换使用：

$$
\widetilde W_e=\Pi W_e^\star P_e,
\qquad
\widetilde W_h=Q_hW_h^\star\Pi^\top.
$$

客户端发送 $\tau(x)$，服务器模型的 Embedding 行和 LM Head 输出列使用同一
$\Pi$。生成的私有 token 会作为下一步 decode 输入继续回灌，不在服务器上逆置换。

### 1.3 P/Q 扩维

明文残差维度为 $d$，私有维度为 $D=d+2h$：

$$
P\in\mathbb R^{d\times D},
\qquad
Q\in\mathbb R^{D\times d},
\qquad
PQ=I_d.
$$

若明文行向量为 $x$，私有坐标为 $z=xP$。相邻线性层分别乘 $P$ 和 $Q$，
使中间坐标抵消。实现保留论文 Algorithm 1 的共享 `Init` 和独立 $D_j$，没有
换成自行设计的可逆方阵。

### 1.4 RMSNorm 修正式

论文标量 $\kappa$ 对一般矩形 $P$ 不能保证逐输入等价。精确实现由 $PQ=I_d$
构造公开派生度量：

$$
G=QQ^\top,
\qquad
zGz^\top=xPQ Q^\top P^\top x^\top=xx^\top.
$$

私有 RMS 使用：

$$
\operatorname{PrivRMS}(z)
=\frac{z}{\sqrt{zGz^\top/d+\varepsilon}}.
$$

该修改保留 P/Q 扩维和所有论文结构变换，只替换无法满足函数等价的标量归一化。
完整推导见 `docs/paper_errata/E15_RMSNorm_Scalar_Kappa_Is_Not_Exact.md`。

### 1.5 Qwen Attention

Qwen2.5-0.5B 每层有 14 个 Q head 和 2 个 KV head，即每个 KV head 对应 7 个
Q head。实现对这 7 个 Q head 与对应 KV head 同步置换。Q/K、V/O 的变换为：

$$
\widetilde W_q=Q_qW_qR_{qk}H_{qk}Z,
\qquad
\widetilde W_k=Q_kW_kR_{qk}H_{qk}^{-1}Z,
$$

$$
\widetilde W_v=Q_vW_vU_{vo},
\qquad
\widetilde W_o=U_{vo}^{-1}W_oP_o.
$$

其中 $R_{qk}$ 是 Q/K head 同步置换，$H_{qk}$ 是逐 head 缩放，$Z$ 是二维
RoPE block 置换。Qwen 不同二维块使用不同频率。只有 $ZR(t)=R(t)Z$ 时置换
才保持函数，而跨频率块一般不满足该式。因此：

- `paper-default` 保留论文 $\beta=8$，作为有损论文参数实验；
- 无噪声函数等价配置使用 $\beta=1$；
- GQA、head permutation、Q/K 缩放、V/O 变换仍全部启用。

推导见 `docs/paper_errata/E16_RoPE_Block_Permutation_Is_Not_Function_Preserving.md`。

### 1.6 FFN、Residual 与 KV Cache

Qwen SwiGLU 的 Gate 和 Up 使用同一个中间维置换 $S$ 与缩放 $D_f$：

$$
\widetilde W_{gate}=Q_gW_{gate}SD_f,
\qquad
\widetilde W_{up}=Q_uW_{up}SD_f,
$$

$$
\widetilde W_{down}=D_f^{-1}S^\top W_{down}P_{out}.
$$

Attention 和 FFN 的主支路、Residual 支路在相加前使用同一个私有残差坐标。
Prefill 产生的私有 K/V Cache 直接用于 decode，服务器不恢复明文坐标。

### 1.7 服务端输出与客户端恢复

服务端返回私有 token：

$$
\widetilde y_t=\widetilde f(\widetilde x,\widetilde y_{<t}).
$$

客户端逐 token 执行：

$$
y_t=\tau^{-1}(\widetilde y_t),
\qquad
\text{text}_t=\operatorname{Tokenizer.decode}(y_{\le t}).
$$

SDK 对累计文本做前缀差分，SSE 每次只打印新增文本。

## 2. 已实现的代码

### 2.1 仓库入口

| 功能 | 文件 | 实现内容 |
|---|---|---|
| 统一 CLI | `src/aloepri/cli.py` | `convert`、`verify`、`serve`、`chat`、打包、拆钥、RmDP 计算 |
| 客户端 SDK | `src/aloepri/client/sdk.py` | Chat Template、本地历史、token 置换、SSE 增量恢复、M1、Bearer Token |
| HF 服务 | `src/aloepri/serving/app.py` | token-ID-only FastAPI、认证、请求限制、安全错误响应 |
| HF 运行时 | `src/aloepri/serving/hf_runtime.py` | 私有 checkpoint、prefill/decode、KV Cache、TTFT/TPOT |
| 协议 | `src/aloepri/serving/protocol.py` | 请求、响应、SSE 数据结构和边界校验 |
| 审计日志 | `src/aloepri/serving/audit_log.py` | 只记录 ID、token 数、状态和耗时 |
| 模型转换 | `src/aloepri/conversion/paper_qwen2.py` | Algorithm 1、Algorithm 2、噪声、逐层权重转换和分片保存 |
| 私有 Qwen | `src/aloepri/models/modeling_aloepri_qwen2.py` | 精确度量 RMSNorm、自定义 Qwen 加载和 forward |
| 结构变换 | `src/aloepri/transforms/qwen_structural.py` | GQA、RoPE、Q/K/V/O、FFN 置换和缩放 |
| P/Q | `src/aloepri/transforms/paper_key_matrix.py` | 论文 Algorithm 1 共享初始化和独立右逆 |
| 论文噪声 | `src/aloepri/transforms/paper_noise.py` | Embedding、Head 高斯噪声 |
| RmDP/M1 | `src/aloepri/privacy/rmdp.py` | 论文预算计算器和客户端逐 token 指数机制 |
| 密钥打包 | `src/aloepri/packaging.py` | 在线/离线拆钥、服务器包、SHA-256、秘密扫描 |
| 证据绑定 | `src/aloepri/evidence.py` | 模型、key、数据、脚本和运行时指纹 |

### 2.2 诊断与验收脚本

| 脚本 | 检查内容 |
|---|---|
| `scripts/verify_layerwise_equivalence.py` | 460 个逐层边界，首个失败算子、误差、Cache |
| `scripts/verify_greedy_corpus.py` | 固定 200 条中英文 prompt 的 greedy 逐 token 对齐 |
| `scripts/verify_paper_qwen2_checkpoint.py` | prefill、decode、Cache、greedy 与显存 |
| `scripts/inspect_structural_key.py` | V/O 条件数、RoPE 移动块、Q/K 代数误差 |
| `scripts/verify_product_privacy_boundary.py` | 抓包、唯一标记、错误 key/model、越界 token、包扫描 |
| `scripts/screen_qwen05b_multiseed_vma.py` | PUPA VMA、TTRSR、PIIRSR、BLEU-4、CosSim |
| `scripts/run_prompt_regression.py` | 噪声 checkpoint 的真实问答和 token 分叉 |
| `scripts/product_chat_smoke.py` | 真实 HTTP 问答闭环 |

## 3. 交付工件和用途

| 工件 | 路径 | 用途 | 状态 |
|---|---|---|---|
| 原模型 | `data/models/qwen2.5-0.5b` | 明文基准 | 已使用 |
| v20 | `data/checkpoints/qwen2.5-0.5b-paper-v20-exact-rms-alg2-fp32` | $\beta=8$ 结构消融 | 有损，不发布 |
| v21 | `data/checkpoints/qwen2.5-0.5b-product-v21-beta1-no-noise-fp32` | $U_{vo}\le100$ 的无噪声函数等价 | 历史功能基线 |
| v22 | `data/checkpoints/qwen2.5-0.5b-product-v22-alpha1-head02-fp32` | $\alpha_e=1,\alpha_h=0.2$ 隐私候选 | 效用失败 |
| v25 | `data/checkpoints/qwen2.5-0.5b-product-v25-fp64proj-uvo45-no-noise-fp32` | FP64 离线投影、$U_{vo}\le45$ | 当前最佳功能基线 |
| 在线 key | `data/keys/qwen05b-product-v25-online` | 客户端 `tau`、`inverse_tau`、特殊 token | manifest 通过 |
| 离线 key | `data/keys/qwen05b-product-v25-offline` | P/Q、结构、噪声和重建材料 | 仅离线保存 |
| 服务器包 | `data/packages/qwen05b-functional-v25` | 不含 tokenizer 和密钥的 HF 服务包 | 扫描通过 |

服务器包只包含模型配置、私有权重和 manifest。扫描拒绝 `tau`、`inverse_tau`、
P/Q、随机种子、原始权重和未登记文件。在线 key 加载前校验文件集合、大小和
SHA-256。

## 4. 实际运行命令

### 4.1 安装和代码检查

```powershell
cd E:\AloePri
uv sync --frozen --all-groups --extra eval
uv run ruff check src scripts tests
uv run mypy src
uv run pytest -q
```

### 4.2 从配置转换

```powershell
uv run aloepri convert --config configs/product/qwen05b.yaml
```

该命令读取原模型，转换 checkpoint，生成完整 key，再拆分 online/offline key。
`configs/product/qwen05b.yaml` 当前是无噪声功能候选，不是安全发布参数。

### 4.3 验证 v25

```powershell
uv run python scripts/verify_layerwise_equivalence.py `
  --original data/models/qwen2.5-0.5b `
  --private data/checkpoints/qwen2.5-0.5b-product-v25-fp64proj-uvo45-no-noise-fp32 `
  --key-dir data/keys/dev-qwen05b-product-v25-fp64proj-uvo45-no-noise-fp32 `
  --nrmse-limit 1e-5 `
  --out artifacts/verification/qwen05b-product-v25-layerwise.json

uv run python scripts/verify_greedy_corpus.py `
  --original data/models/qwen2.5-0.5b `
  --private data/checkpoints/qwen2.5-0.5b-product-v25-fp64proj-uvo45-no-noise-fp32 `
  --key-dir data/keys/dev-qwen05b-product-v25-fp64proj-uvo45-no-noise-fp32 `
  --prompts configs/eval/gate1_prompts_200.json `
  --max-new-tokens 16 `
  --batch-size 4 `
  --dtype float32 `
  --out artifacts/verification/qwen05b-product-v25-greedy200.json
```

### 4.4 启动服务和真实问答

终端 A：

```powershell
cd E:\AloePri
uv run aloepri inspect-package `
  --server-package data/packages/qwen05b-functional-v25
uv run aloepri serve `
  --config configs/product/qwen05b_v25_functional.yaml
```

终端 B：

```powershell
cd E:\AloePri
uv run aloepri chat `
  --server http://127.0.0.1:8000 `
  --key-dir data/keys/qwen05b-product-v25-online `
  --tokenizer-dir data/models/qwen2.5-0.5b
```

交互命令：`/clear` 清除本地历史，`/stats` 查看 TTFT/TPOT，`/privacy` 查看本地
模式，`/quit` 退出。非 localhost 部署时设置 `ALOEPRI_BEARER_TOKEN`，并在 TLS
反向代理后启动服务。

### 4.5 RmDP 预算

```powershell
uv run aloepri rmdp-budget `
  --epsilon1 2.0 `
  --noise-variance 1.0 `
  --embedding-sigma1 1.0 `
  --embedding-sigma2 0.8 `
  --head-sigma1 1.0 `
  --head-sigma2 0.8 `
  --vocab-size 151936
```

`rmdp` 问答必须显式传入 `epsilon1`。客户端显示预计 token 改变率，允许取消，
且不向服务端发送 `epsilon1` 或扰动前 token。

## 5. 当前实测结果

### 5.1 无噪声函数等价

| 检查 | 计划门槛 | v25 FP32 实测 | 判定 |
|---|---:|---:|---|
| 逐层算子 NRMSE | 每项 $\le10^{-5}$ | 457/460 项通过；3 项在 $1.343\times10^{-5}$ 至 $1.540\times10^{-5}$ | 未严格通过 |
| 最终 logits NRMSE | $\le10^{-5}$ | $6.155\times10^{-6}$ | 通过 |
| Prefill Top-1 | 100% | 当前样本 36/36 token 位置一致 | 通过 |
| Prefill 平均绝对误差 | 记录值 | $1.521\times10^{-5}$ | 记录 |
| Decode 平均绝对误差 | 记录值 | $9.074\times10^{-6}$ | 记录 |
| 固定 prompt greedy | 200/200 | 200/200；Wilson 95% CI 为 [98.115%, 100%] | 通过 |
| Cache/无 Cache | token 完全一致 | 当前验证样本一致 | 通过 |

3 个超阈值点为 layer 0 Attention softmax、layer 22 value aggregate 和 O projection；
Attention Residual、FFN Residual、Final RMSNorm、Logits 和 KV Cache 均在阈值内。因为计划
规定每个边界都要通过，Gate 1 仍不能写成完成。

### 5.2 论文 RMSNorm 与 RoPE 错误的影响

| 配置 | 变化 | Prefill 平均绝对误差 | Top-1 | 结果 |
|---|---|---:|---:|---|
| v18 | 论文标量 $\kappa$ | 2.631565 | 22.22% | 函数失败 |
| v20 | 精确 RMS，$\beta=8$ | 0.012042 | 97.22% | 跨频率置换有损 |
| v21 | 精确 RMS，$\beta=1$ | 0.00001842 | 100% | 200/200 greedy 一致 |
| v25 | v21 + FP64 离线投影，$U_{vo}\le45$ | 0.00001521 | 100% | 200/200 greedy 一致 |

### 5.3 当前 checkpoint 的效用—隐私冲突

| checkpoint | $\alpha_e/\alpha_h$ | 功能结果 | VMA 结果 | 产品判定 |
|---|---:|---|---|---|
| v25 | 0 / 0 | 200/200 greedy 一致 | 无噪声不改变权重匹配安全结论 | 不满足隐私 |
| v22 | 1.0 / 0.2 | 20 条问答 0/20 完全一致；token agreement 61.125% | 当前单 key 的 We/Wh 与 We/Wgate 达到数值门槛 | 不满足效用 |

v22 的 0/20 exact 比例 Wilson 95% CI 为 [0%, 16.113%]。该 20 条检查是候选筛选，
不是 MMLU、C-Eval、PIQA、IFEval 或 HumanEval 的替代结果。

### 5.4 与论文和甲方目标同表比较：仅列本轮已测项目

论文列是论文 Qwen3-14B 的 VMA，不是 Qwen2.5-0.5B 基准。当前列是 v22、
seed `20260803`、PUPA 62,634 个 token occurrence、664 个 PII 单元、16,384
候选的筛选结果。

| 权重组合/指标 | 论文 Qwen3-14B | 甲方目标 | 0.5B v22 当前实测 | 数值判定 |
|---|---:|---:|---:|---|
| We/Wh TTRSR | 25.05% | $\le15\%$ | 13.426%，Wilson 95% CI [13.161%, 13.695%] | 达到 |
| We/Wh PIIRSR | 1.62% | $\le3\%$ | 0.151%，Wilson 95% CI [0.027%, 0.848%] | 达到 |
| We/Wh BLEU-4 | 1.72 | $\le2.5$ | 0.00536 | 达到 |
| We/Wh CosSim | 论文未给 | $\le0.5$ | -0.2942 | 达到 |
| We/Wgate TTRSR | 论文主表未分组合 | $\le15\%$ | 1.049%，Wilson 95% CI [0.972%, 1.132%] | 达到 |
| We/Wgate PIIRSR | 论文主表未分组合 | $\le3\%$ | 0.151%，Wilson 95% CI [0.027%, 0.848%] | 达到 |
| We/Wgate BLEU-4 | 论文主表未分组合 | $\le2.5$ | $8.88\times10^{-7}$ | 达到 |
| We/Wgate CosSim | 论文未给 | $\le0.5$ | -0.3653 | 达到 |

上表的 Wilson 区间描述 occurrence 或 PII 单元抽样波动，不代表不同 key 之间的
不确定性。计划要求 5 个独立 key；目前只完成 1 个 key，因此不能据此通过攻击总门禁。

### 5.5 产品边界和资源

| 项目 | 实测 |
|---|---:|
| 唯一明文标记出现在请求/响应 | 否 |
| 完整明文 token 序列出现在请求/响应 | 否 |
| 请求 ID 是否等于 $\tau(x)$ | 是 |
| 错误 key/model/越界 token | HTTP 400 安全失败 |
| v25 服务器包秘密扫描 | 通过，0 findings |
| 恢复回答 | `42` |
| v25 真实问答样例 | “矩阵乘法是一种将一个矩阵与另一个矩阵相乘的运算……” |
| v25 预热后 16-token HTTP TTFT | 242.08 ms |
| v25 预热后 16-token HTTP TPOT | 45.05 ms |
| 明文 FP32 模型峰值 allocated | 2,112,222,720 bytes |
| v25 FP32 私有模型峰值 allocated | 3,451,209,216 bytes |
| Ruff | `All checks passed` |
| Mypy | 44 个源文件，0 issues |
| 单元/普通集成测试 | 96 passed，1 个真实模型测试默认跳过 |
| v25 真实模型集成测试 | 1 passed；HTTP、SSE、逆置换、错误 key |

TTFT/TPOT 只有私有模型单次产品冒烟值，没有同一请求分布下的明文 ABBA 对照，
因此不用于 15% 性能门禁。

## 6. 门禁状态

| Gate | 必要条件 | 当前状态 | 原因 |
|---|---|---|---|
| Gate 1 | 逐层每项 NRMSE $\le10^{-5}$，200/200 greedy | 未通过 | greedy 通过；460 项中 7 项略超阈值 |
| Gate 2 | 至少一个非零噪声配置同时通过功能、精度、隐私 | 未通过 | v25 无噪声隐私失败；v22 强噪声问答效用失败 |
| Gate 3 | 产品闭环和密钥拆分 | 功能演示已实现 | HF、CLI、SDK、SSE、拆钥和包扫描可用；不是安全发布包 |
| Gate 4 | 全基准、全攻击、5 key、HF/vLLM/SGLang | 未执行正式验收 | Gate 2 未通过，不允许用旧 v15 代替 |
| Gate 5 | 当前工件全部绑定且前四门通过 | `NO-GO` | 停止条件“非零噪声未同时满足效用和隐私”已发生 |

## 7. 尚未执行的正式验收

下列项目没有写入当前结果表：

| 项目 | 必须执行的下一步 |
|---|---|
| Gate 1 剩余数值点 | 对 3 个 Attention 边界做 FP64 在线参考与舍入误差定位；修复后重跑 460 项和 200 prompts |
| 参数校准 | 按配置网格运行独立 calibration/candidate/test manifest；不能在最终集调参 |
| 完整精度 | 当前产品候选运行 MMLU 14,042、C-Eval 1,346、PIQA 1,838、IFEval 541、HumanEval 164 |
| 多 key 攻击 | 20260803–20260807 五个 key；VMA 五组 Dense 组合、24 层、16,384 候选 |
| 其他攻击 | Gate-IA、Attention-IA、IMA、ISA、TFMA、SDA、known-plaintext 全部绑定当前候选 |
| 性能 | 同硬件、同请求分布明文/私有 ABBA；TTFT/TPOT/吞吐/显存 p50/p95/p99 |
| vLLM | WSL2、上下文 2048、batch 1、显存上限 75%，与 HF 做 32-token greedy 一致性 |
| SGLang | 同一 checkpoint 的注册、KV Cache、EOS 和 SSE 一致性 |
| TLS | 用 `deploy/` 下反向代理配置做远程部署和 Bearer Token 实测 |

## 8. 人工检查路径

1. 打开 `configs/product/qwen05b_functional_package.yaml`，确认服务只加载
   `data/packages/qwen05b-functional-v25`。
2. 运行 `aloepri inspect-package`，确认 `pass=true` 且 `findings=[]`。
3. 打开 `data/keys/qwen05b-product-v25-online/manifest.json`，核对只列
   `key.json` 和 `online_key.safetensors`。
4. 打开 `artifacts/verification/qwen05b-functional-v25/privacy-boundary.json`，
   核对 `checks` 全部为 `true`。
5. 打开 `artifacts/verification/qwen05b-product-v25-layerwise.json`，从
   `first_failure` 和 `records` 查看每个算子的误差。
6. 打开 `artifacts/verification/qwen05b-product-v25-greedy200.json`，核对
   `exact_prompt_count=200`、`prompt_count=200`、`all_exact=true`。
7. 打开 `artifacts/verification/qwen05b-product-v22-prompt20.json`，直接查看
   20 条明文回答与恢复回答，确认强噪声候选不能发布。
8. 打开 `artifacts/privacy/qwen05b-product-grid-screen-default-seed20260803.json`，
   核对 VMA 分子、分母、Wilson 区间和指标定义。

## 9. 结论

当前 0.5B 已经具备真实客户端输入、Chat Template、本地分词、token ID 置换、
私有 HF checkpoint、SSE 生成、客户端逆置换和正常回答的完整产品闭环。服务器包
不含在线 key、P/Q、转换种子、原 tokenizer 和原权重，抓包中不出现明文或明文
token 序列。

当前不能发布为“同时满足论文效用和甲方隐私门槛”的产品：无噪声 v25 保持问答
但不能抵抗权重匹配；论文强噪声 v22 的当前 VMA 数值合格，但 20 条问答 0/20
完全一致。按照既定停止条件，正式结论保持 `NO-GO`，旧 v15、Toy 结果和代理攻击
均未用于当前 checkpoint 验收。
