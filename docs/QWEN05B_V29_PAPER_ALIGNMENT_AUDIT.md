# Qwen2.5-0.5B v29 论文对齐审核

## 1. 审核结论

本次审核对象是：

- 原文：`01_client_technical_route.pdf`，arXiv:2603.01499v2；
- 模型：`Qwen2.5-0.5B-Instruct`；
- checkpoint：`data/checkpoints/qwen2.5-0.5b-product-v29-attnfp64-uvo42-no-noise`；
- full key：`data/keys/dev-qwen05b-product-v29-attnfp64-uvo42-no-noise`；
- server package：`data/packages/qwen05b-functional-v29`。

审核结果分为三层：

| 判定对象 | 结果 | 直接证据 |
|---|---|---|
| Qwen2.5-0.5B 适用的模型变换是否已写入 checkpoint | 通过 | 292 个保存张量按转换公式重新计算，292/292 完全相等 |
| 客户端置换、服务端生成、客户端恢复的功能闭环是否成立 | 通过 | 200/200 个 prompt、3200/3200 个生成 token 完全一致；HTTP 与 SSE 序列一致 |
| 是否逐字采用原文所有公式、默认参数和文本协议 | 不通过 | 当前有 8 项明确差异，均在第 5 节列出 |
| 是否可作为非零噪声安全产品发布 | 不通过 | v29 的 `alpha_e=0`、`alpha_h=0`，没有绑定 v29 的非零噪声精度与攻击验收 |

因此，当前可以确认的是：**论文中适用于稠密 GQA Qwen 的坐标变换已经形成可运行、可重建、可问答的功能版本；不能把 v29 写成论文默认安全参数的实验复现。**

## 2. 原文要求的完整数据流

### 2.1 客户端输入

客户端持有原始 tokenizer、词表置换 $\tau$ 和逆置换 $\tau^{-1}$。对明文 prompt 应用 Chat Template 后得到原始 token：

$$
s=(t_1,t_2,\ldots,t_n).
$$

默认隐私模式直接置换每个 token ID：

$$
\widetilde{s}=\tau(s)
=(\tau(t_1),\tau(t_2),\ldots,\tau(t_n)).
$$

原文第 5.3 节要求先把 $\widetilde{s}$ 解码成“混淆文本”，服务端再分词。该等式要求

$$
\operatorname{Encode}(\operatorname{Decode}(\widetilde{s}))=\widetilde{s},
$$

但 Qwen tokenizer 对任意置换 ID 不满足这个条件。因此产品接口直接发送 $\widetilde{s}$ 的 token ID，避免混淆文本重分词破坏序列。

### 2.2 Embedding 和词表

原始 Embedding 为 $W_e\in\mathbb{R}^{|V|\times d}$。原文先加入高斯噪声：

$$
W_e^*=W_e+\alpha_e E_e,
\qquad
(E_e)_{ij}\sim\mathcal{N}(0,\sigma_e^2),
$$

其中 $\sigma_e$ 是原始 Embedding 权重的标准差。再使用词表置换矩阵 $\Pi$ 和扩维矩阵 $P$：

$$
\widetilde{W}_e=\Pi W_e^*P.
$$

代码采用 HF 的行存储方向，等价操作是先计算 `weight @ P`，再把原始词表行移动到 `tau[plain_id]` 对应的私有行。

### 2.3 Algorithm 1：从 $d$ 扩展到 $d+2h$

代码按原文构造

$$
B=U+\lambda V,
$$

并生成共享的 $B,E,F,Z$。修正后的矩阵为

$$
P=[B\;C\;E]Z,
$$

$$
Q=Z^\top
\begin{bmatrix}
B^{-1}\\
F\\
D
\end{bmatrix},
$$

其中必须满足

$$
CF=0,
\qquad
ED=0.
$$

于是

$$
PQ=BB^{-1}+CF+ED=I_d.
$$

原文把 $C$ 和 $D$ 的“行/列属于哪个零空间”写反，按原句无法满足矩阵维度。实现使用可计算且能证明 $PQ=I_d$ 的解释：$C$ 的每一行位于 $\operatorname{null}(F^\top)$，$D$ 的每一列位于 $\operatorname{null}(E)$。

### 2.4 Attention

进入 Attention 前，私有 residual 为

$$
z=xP.
$$

每个输入投影使用与同一 $P$ 兼容的右逆：

$$
PQ_q=PQ_k=PQ_v=I_d.
$$

在 HF 权重方向下，基础投影变换为

$$
\widetilde{W}_q=W_q\Gamma Q_q^\top,
\quad
\widetilde{W}_k=W_k\Gamma Q_k^\top,
\quad
\widetilde{W}_v=W_v\Gamma Q_v^\top,
$$

其中 $\Gamma$ 是融合进投影的原始 RMSNorm 权重。

Algorithm 2 对每个 KV 组构造 Q/K 变换：

$$
T_q=R_{qk}H_{qk}Z_{block},
\qquad
T_k=R_{qk}H_{qk}^{-1}Z_{block},
$$

因而在不发生错误 RoPE 频率置换时：

$$
T_qT_k^\top=I.
$$

V/O 使用高斯矩阵 $U_{vo}$ 和逆矩阵：

$$
v'=vU_{vo},
\qquad
W_o'=U_{vo}^{-1}W_oP.
$$

Qwen2.5-0.5B 有 14 个 Q head、2 个 KV head，每个 KV head 对应 7 个 Q head。实现同步置换 KV 组、对应的 7 个 Q/O head，并在每组内部置换 Q/O head。全量公式审核检查了 24 层的组对应关系。

### 2.5 FFN

对 SwiGLU 的 Gate、Up、Down 三个矩阵使用同一个中间维置换 $Z_{ffn}$ 和同一组非零缩放 $H_{ffn}$：

$$
\widetilde{W}_{gate}=Q_{gate}W_{gate}Z_{ffn},
$$

$$
\widetilde{W}_{up}=Q_{up}W_{up}H_{ffn}^{-1}Z_{ffn},
$$

$$
\widetilde{W}_{down}=Z_{ffn}^{-1}H_{ffn}W_{down}P.
$$

代码按 HF 转置存储执行同一组行置换、列置换和互逆缩放。原文没有给出随机缩放的概率分布；配置记录实际使用的对数均匀范围 `[0.5, 2.0]`。

### 2.6 RMSNorm 和 Residual

原文用标量

$$
\kappa=\mathbb{E}\left[\frac{\lVert xP\rVert_2}{\lVert x\rVert_2}\right]
$$

近似补偿 RMSNorm。但一般矩形 $P$ 会各向异性地改变向量，单个标量不能对每个输入精确恢复原 RMS。

v29 使用可验证的精确修正式。若 $z=xP$ 且 $PQ=I$，则

$$
x=zQ,
$$

$$
\operatorname{RMS}(x)^2
=\frac{1}{d}zQQ^\top z^\top.
$$

服务端 checkpoint 因此保存派生度量

$$
G=QQ^\top.
$$

两条 residual 支路始终保留在同一个 $P$ 坐标中；只有进入 Q/K/V、Gate/Up、LM Head 等投影时，才通过相应的兼容右逆恢复原坐标。

### 2.7 LM Head 和客户端输出

Head 噪声和变换为

$$
W_h^*=W_h+\alpha_hE_h,
$$

$$
\widetilde{W}_h=\Pi W_h^*\Gamma_fQ_h^\top.
$$

服务端输出私有 token ID $\widetilde{y}_i$。客户端逐 token 计算

$$
y_i=\tau^{-1}(\widetilde{y}_i)
$$

并使用原始 tokenizer 解码。对话历史、明文 prompt、$\tau$ 和 $\tau^{-1}$ 均留在客户端。

### 2.8 M1 和 Theorem 4

原文 Definition 2 在长度为 $n$ 的 token 序列空间上定义换位距离 $d(x,x')$，M1 的分布为

$$
p_{M_1(x)}(y)\propto \exp\{-\epsilon_1 d(x,y)\}.
$$

Theorem 4 的分段预算使用的是**序列长度 $n$**，不是词表大小：

$$
\epsilon_2=\pi^2(\epsilon_e+\epsilon_h),
$$

$$
\epsilon=
\begin{cases}
\epsilon_1-\dfrac{\epsilon_1^2}{4(n-1)\epsilon_2},
& \epsilon_1\le 2(n-1)\epsilon_2,\\
(n-1)\epsilon_2,
& \text{otherwise}.
\end{cases}
$$

本轮审核发现旧实现误把 `vocab_size` 代入了 $n$。现已把正式接口改为
`sequence_length` / `--sequence-length`；旧 `vocab_size` 仅保留为会触发弃用警告的兼容别名。
M1 产品路径仍是长度 1 的精确机制逐 token 组合，不把它写成原文完整长度-$n$ 联合采样器。

## 3. 原文步骤与当前实现逐项审核

| ID | 原文位置 | 功能 | 当前状态 | 代码位置 | 审核说明 |
|---|---|---|---|---|---|
| M01 | PDF p.8，5.2.2 | 词表置换与逆置换 | 原文公式已启用 | `src/aloepri/transforms/vocab.py` | 词表双向置换正确，Embedding/Head 行同步变换 |
| M02 | PDF pp.8-9，5.2.2 | Embedding/Head 高斯噪声 | 已实现，本工件关闭 | `src/aloepri/transforms/paper_noise.py` | v29 为无噪声功能工件，`alpha_e=alpha_h=0` |
| M03 | PDF p.8，Algorithm 1 | $d\to d+2h\to d$ | 修正式已启用 | `src/aloepri/transforms/paper_key_matrix.py` | 修正原文 C/D 零空间维度错误，所有角色满足 $PQ\approx I$ |
| M04 | PDF pp.8-9，5.2.2 | Embedding、LM Head、P/Q | 原文公式已启用 | `src/aloepri/conversion/paper_qwen2.py` | 保存权重逐张量重建完全一致 |
| M05 | PDF p.9，Algorithm 2 | Q/K、V/O、RoPE 映射 | 数值稳定版已启用 | `src/aloepri/transforms/qwen_structural.py` | 高斯 $U_{vo}$ 增加条件数拒绝，Attention 用 FP64 |
| M06 | PDF p.9，BlockPerm | RoPE 频率块置换 | 已实现，本工件因原文错误关闭 | 同上 | `beta=1`；非平凡置换不与 Qwen RoPE 对易 |
| M07 | PDF p.9，5.2.3 | GQA 组间与组内置换 | 原文公式已启用 | `make_attention_key()` | 14 Q / 2 KV / 每组 7 Q，24 层全部通过 |
| M08 | PDF p.10，5.2.4 | FFN 置换和缩放 | 已启用；分布未公开 | `transform_mlp_scaled()` | Gate/Up/Down 使用同一个置换及互逆缩放 |
| M09 | PDF p.10，5.2.5 | RMSNorm | 精确修正式已启用 | `AloePriMetricRMSNorm` | 使用 $G=QQ^\top$，不是原文标量 $\kappa$ |
| M10 | PDF pp.10-11 | Residual 坐标组合 | 修正式已启用 | conversion + custom model | 292/292 张量重建通过；最终 logits NRMSE 通过 |
| M11 | PDF p.10，5.3 | 客户端/服务端在线流程 | 工程修正式已启用 | client + serving | 使用私有 token ID 协议，修复文本 round-trip 不闭合 |
| M12 | PDF p.12，5.4 | M1 与 RmDP | 可选逐 token 版本 | `src/aloepri/privacy/rmdp.py` | 长序列精确采样协议未公开；产品明确标记 `rmdp-tokenwise` |
| M13 | PDF pp.26-27，D.1 | VMA/IA/ISA/IMA/TFMA/SDA | 部分协议复现 | `src/aloepri/attacks`、`scripts/run_*` | 未公开数据/训练协议和错误 Attn-IA 不计为原文精确结果 |
| M14 | PDF pp.9-10 | MLA、MoE | 对该模型不适用 | 无 0.5B 路径 | Qwen2.5-0.5B 是稠密 GQA，不含 MLA、MoE |

## 4. 实测结果

### 4.1 保存权重公式重建

审核器从原模型和 full key 重新计算转换后的每个保存张量，再与 checkpoint 比较。

| 检查 | 实测值 |
|---|---:|
| Algorithm 1 | 通过 |
| Algorithm 2，24 层 | 通过 |
| 保存张量总数 | 292 |
| 完全相等 | 292 |
| 失败张量 | 0 |
| 未覆盖张量 | 0 |
| $\max\lVert BB^{-1}-I\rVert_\infty$ | $6.247\times10^{-15}$ |
| $\max\lVert CF\rVert_\infty$ | $2.686\times10^{-17}$ |
| 六个角色最大相对 $PQ-I$ 误差 | $2.2735\times10^{-14}$ |

这里的“完全相等”是重建结果在写入 dtype 后与 safetensors 中张量逐元素相等，不是根据最终输出反推的结论。

### 4.2 运行时函数结果

| 指标 | 结果 | 判定 |
|---|---:|---|
| 逐层/逐算子检查 | 460 | — |
| NRMSE 不超过 $10^{-5}$ | 457/460 | 3 个局部边界略超阈值 |
| 最终 logits NRMSE | $4.756\times10^{-6}$ | 通过 |
| 200 个固定 prompt | 200/200 完全一致 | 通过 |
| 生成 token | 3200/3200 完全一致 | 通过 |
| Prefill Top-1 | 1.0 | 通过 |
| Decode 下一 token | 完全一致 | 通过 |
| KV Cache 长度 | 36 → 37 | 通过 |
| 私有模型峰值显存 | 3,692,543,488 bytes | 约为 6 GiB 的 57.3% |
| HTTP 隐私边界 | 全部检查为 true | 通过 |
| HTTP/SSE 私有 ID 序列 | 完全一致 | 通过 |
| 实际恢复回答 | 正常中文 | 通过 |

三个局部阈值超限点为：

| 算子 | NRMSE |
|---|---:|
| layer 0 Attention softmax | $1.0881\times10^{-5}$ |
| layer 22 Attention value aggregate | $1.5805\times10^{-5}$ |
| layer 22 O projection | $1.2784\times10^{-5}$ |

它们没有导致 residual、最终 logits 或生成 token 分叉。因而“模型函数可用于真实问答”通过；“460/460 个内部边界均小于 $10^{-5}$”没有通过。

## 5. 与原文字面实现的差异

| 差异 | 原文 | v29 | 原因与处理 |
|---|---|---|---|
| Algorithm 1 的 C/D 零空间 | 原文行列描述维度不成立 | 使用可证明 $CF=0,ED=0$ 的修正式 | E01、E02 |
| 在线传输 | 私有 ID 解码成文本，服务端重分词 | 直接传私有 token ID | Qwen tokenizer round-trip 不闭合，E12 |
| 噪声 | Table 10：$\alpha_e=1,\alpha_h=0.2$ | 0、0 | 本工件只验证无噪声函数闭环 |
| BlockPerm | Table 10：$\beta=8$ | $\beta=1$ | 非平凡频率置换不与标准 RoPE 对易，E03-E06、E10、E16 |
| RMSNorm | 标量 $\kappa$ 近似 | 精确度量 $QQ^\top$ | 标量不能精确补偿一般矩形 P，E07、E15 |
| $U_{vo}$ | 无条件高斯 | 高斯采样后限制条件数不超过 42 | 避免多层逆矩阵误差放大 |
| 数值精度 | 实验设置 bf16 | Attention FP64，其余 FP32 | 保证 0.5B 功能等价 |
| 攻击实验 | 给出名称和汇总结果 | 部分为代理或协议不完整 | 原文缺少若干数据集、训练和候选构造细节；Attn-IA 公式维度错误 |

服务器包扫描结果仍为通过：没有 `P`、`Q`、`tau`、`inverse_tau` 或随机种子。扫描器现在额外给出一条非阻断披露：精确 RMSNorm 的 $QQ^\top$ 是服务端权重中的派生度量。它不是完整 P/Q，但也不能写成原文标量 $\kappa$ 方案。

## 6. 本轮代码修改

| 文件 | 修改内容 |
|---|---|
| `scripts/verify_paper_formula_checkpoint.py` | 按每个目标张量的真实 dtype 重建；支持 FP64 Attention；支持精确 RMSNorm 及 $G=QQ^\top$ 检查 |
| `scripts/audit_qwen05b_paper_alignment.py` | 新增工件绑定的逐步骤审核器；自动区分原文公式、修正式、关闭项、不适用项和未完整复现实验 |
| `src/aloepri/packaging.py` | server package 扫描新增精确 RMS 派生度量披露，不把它误报成 P/Q 泄露 |
| `scripts/convert_paper_qwen2_checkpoint.py` | 新转换工件自动写入 `paper_alignment` 配置，说明 RMS、BlockPerm、RoPE、Uvo、精度和噪声状态 |
| `src/aloepri/privacy/rmdp.py` | 修正 Theorem 4：分段边界和预算使用序列长度 $n$，不再使用词表大小 |
| `src/aloepri/cli.py` | `rmdp-budget` 正式参数改为 `--sequence-length` |
| `tests/unit/test_server_package_inspection.py` | 新增派生度量披露和直接秘密张量拒绝测试 |
| `tests/unit/test_converter_config.py` | 新增论文对齐配置元数据测试 |
| `tests/unit/test_rmdp.py` | 新增序列长度敏感性和旧参数弃用测试 |
| `configs/audit/paper_line_audit_0.5b.json` | 更新 exact-metric、FP64 和新审核器的状态定义 |

## 7. 人工复核命令

### 7.1 重新计算全部 292 个保存张量

```powershell
cd E:\AloePri
uv run python scripts/verify_paper_formula_checkpoint.py `
  --source data/models/qwen2.5-0.5b `
  --private data/checkpoints/qwen2.5-0.5b-product-v29-attnfp64-uvo42-no-noise `
  --key-dir data/keys/dev-qwen05b-product-v29-attnfp64-uvo42-no-noise `
  --out artifacts/verification/qwen05b-functional-v29/formula-reconstruction-full.json
```

预期输出：

```text
overall_pass: true
algorithm1_pass: true
algorithm2_pass: true
tensor_check_count: 292
failed_tensors: []
missing_formula_coverage: []
```

### 7.2 重新生成逐步骤论文对齐结论

```powershell
uv run python scripts/audit_qwen05b_paper_alignment.py `
  --checkpoint data/checkpoints/qwen2.5-0.5b-product-v29-attnfp64-uvo42-no-noise `
  --key-dir data/keys/dev-qwen05b-product-v29-attnfp64-uvo42-no-noise `
  --server-package data/packages/qwen05b-functional-v29 `
  --formula-evidence artifacts/verification/qwen05b-functional-v29/formula-reconstruction-full.json `
  --layerwise-evidence artifacts/verification/qwen05b-product-v29-layerwise-fp64-key.json `
  --greedy-evidence artifacts/verification/qwen05b-product-v29-greedy200.json `
  --runtime-evidence artifacts/verification/qwen05b-product-v29-checkpoint.json `
  --privacy-evidence artifacts/verification/qwen05b-functional-v29/privacy-boundary.json `
  --http-evidence artifacts/verification/qwen05b-functional-v29/http-sse-smoke.json `
  --out artifacts/verification/qwen05b-functional-v29/paper-alignment-audit.json
```

### 7.3 检查服务器包

```powershell
uv run aloepri inspect-package `
  --server-package data/packages/qwen05b-functional-v29
```

预期为 `pass=true`、`findings=[]`，并披露 `aloepri_rms_metric`。

### 7.4 运行代码回归

```powershell
uv run ruff check src scripts tests
uv run mypy src
uv run pytest -q
```

本轮常规结果：`103 passed, 1 skipped`。随后显式指定 v29 checkpoint 和 key，单独运行真实 GPU 集成测试，结果为 `1 passed`。该测试加载真实 0.5B 私有模型，覆盖 FastAPI 非流式生成、SSE 逐 token 生成、客户端逆置换和错误 key 安全失败。

## 8. 继续实现的边界

要把“当前功能工件”推进为“论文默认安全参数工件”，必须另建 checkpoint，不得覆盖 v29：

1. 启用非零 $\alpha_e,\alpha_h$，重新跑 200 prompt、完整精度任务和全部攻击；
2. 将 `paper-default` 与 `product` 分开，不能用无噪声结果证明有噪声模型；
3. BlockPerm 若坚持 $\beta=8$，只能如实测量其精度损失；标准 Qwen RoPE 下不能同时声称它严格函数保持；
4. 标量 $\kappa$ 分支可以复现实验近似，但不能取代 exact-metric 产品分支的功能门禁；
5. VMA、Gate-IA、TFMA 等有足够定义的攻击应绑定新 checkpoint 重跑；缺少公开协议的 IMA/SDA 和错误 Attn-IA 必须继续单列。

本报告对应的机器证据是 `artifacts/verification/qwen05b-functional-v29/paper-alignment-audit.json`。该文件绑定当前 checkpoint config、offline key、公式证据、逐层证据、200 prompt、运行时、隐私边界和 HTTP/SSE 结果的 SHA-256。
