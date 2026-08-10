# AloePri 0.5B 失败分析与论文忠实实现方案 v1.0

## 1. 结论

当前 `qwen2.5-0.5b-full-exact` 是保持原始 Qwen2 形状的可逆置换版本，只覆盖论文 AloePri 的一部分。它完成了 token 置换、GQA 头置换、RoPE 可交换坐标变换、V/O 变换和 FFN 神经元置换，但没有把论文 Algorithm 1 的扩维 Key Matrix 注入模型，也没有按论文尺度对 Embedding 和 Model Head 独立加噪。

因此当前结果只能证明：

1. token-ID 私有推理、checkpoint、HF、vLLM、API 和客户端链路可运行；
2. 保持原模型形状的精确置换不能提供论文声称的权重隐私；
3. 固定 `1e-4` 噪声不足以隐藏公开原模型与混淆模型之间的词表对应关系；
4. 当前攻击工具不等价于论文的完整 VMA、IA、TFMA、SDA、ISA 和 IMA；
5. 不能用当前 99.22% 恢复率直接否定论文，也不能用现有精度结果宣称论文复现成功。

后续主分支必须从“结构不变的工程近似”切换到“论文忠实扩维模型”，并重新执行完整攻击和精度实验。旧分支保留为对照组，不再作为隐私产品候选。

## 2. 输入基线

### 2.1 论文基线

文件：`01_client_technical_route.pdf`

论文版本：arXiv:2603.01499v2，2026-03-30。

固定论文参数：

| 参数 | 论文默认值 | 含义 |
|---|---:|---|
| `alpha_e` | 1.0 | Embedding 噪声系数 |
| `alpha_h` | 0.2 | Model Head 噪声系数 |
| `lambda` | 0.3 | Key Matrix 范数调节系数 |
| `h` | 128 | 扩维参数，残差维度从 `d` 变为 `d+2h` |
| `beta` | 8 | RoPE block permutation 最大窗口 |
| `gamma` | 1000 | block window 采样参数 |
| dtype | BF16 | 论文实验推理精度 |
| generation | temperature 0.65, top-k 20, top-p 0.95 | 论文生成配置 |

论文的核心权重变换采用扩维左右变换：

```text
P_hat: [d, d+2h]
Q_hat: [d+2h, d]
P_hat @ Q_hat = I_d

Embedding:  W_embed_tilde = Pi @ (W_embed + alpha_e E_embed) @ P_hat
LM Head:    W_head_tilde  = Q_hat @ (W_head + alpha_h E_head) @ Pi^T
Linear:     W_tilde       = Q_in @ W @ P_out
```

对 Qwen2.5-0.5B：

```text
d = 896
h = 128
D = d + 2h = 1152
num_attention_heads = 14
num_key_value_heads = 2
head_dim = 64
intermediate_size = 4864
```

残差流维度应变为1152，但Q/K/V内部投影仍保持14×64、2×64和2×64。不能用 `D / num_heads` 推导 `head_dim`。

### 2.2 甲方要求

文件：`02_client_project_brief.pptx`

甲方目标是落地论文并加以改进，交付离线模型混淆和在线隐私推理原型。验收包括：

- 精度绝对下降不超过3.5个百分点；
- TTFT、TPOT劣化不超过15%；
- 抵抗VMA；
- 兼容PyTorch、CUDA和vLLM/SGLang推理链路；
- 保护prompt和response。

PPT中“整个过程不改变模型结构”和论文扩维存在冲突。工程口径改为：Transformer层数、Attention语义、FFN语义、KV Cache协议和生成流程不变；权重形状及模型注册需要改变。不得继续解释为“原生Qwen类无需修改”。

### 2.3 当前实现与证据

当前关键证据：

| 证据 | 结果 |
|---|---:|
| FP32 20条greedy生成一致率 | 100% |
| MMLU/PIQA/C-Eval/IFEval小样本下降 | 0 |
| TTFT p50劣化 | 0.16% |
| TPOT p50劣化 | 4.68% |
| 无噪声直接权重匹配恢复率 | 99.9901% |
| `1e-4`噪声后cosine映射恢复率 | 99.21875% |
| HumanEval 3题 | 原模型0.667，私有模型0.333 |
| vLLM | 0.11.1、WSL2、真实生成通过 |

这些数据仅作为开发期 smoke evidence，不能作为论文级统计结论。

## 3. 不成功原因

### 3.1 主因：论文的 P/Q 扩维没有进入真实Qwen checkpoint

`src/aloepri/transforms/pq.py` 中虽然存在 `make_expanded()`，但它只生成一个半正交投影：

```text
expand = Q^T
contract = Q
expand @ contract = I
```

该实现没有复现 Algorithm 1 的 `B、E、F、C、D、Z` 构造，也没有被 `convert_structural_checkpoint.py` 调用。实际结构转换只调用了 `transform_qwen_layers()`，对原形状权重做置换和正交变换。

直接后果：

- 混淆 Embedding 仍与原 Embedding 处于同一896维坐标系；
- 除极小噪声外，原词向量与混淆词向量的cosine关系基本保留；
- 攻击者拥有公开原模型时可以逐行最近邻恢复 `tau`；
- 当前几乎100%的映射恢复率是预期结果。

### 3.2 噪声定义错误

论文定义：

```text
E_embed ~ N(0, sigma_e^2)
W_embed_star = W_embed + alpha_e * E_embed

E_head ~ N(0, sigma_h^2)
W_head_star = W_head + alpha_h * E_head
```

其中 `sigma_e`、`sigma_h` 来自相应权重的标准差。当前实现把命令行 `--std 1e-4` 直接当作噪声标准差，没有乘权重标准差，也没有分别接受 `alpha_e`、`alpha_h`。

当前实现还对 tied Embedding/Head 只加一次相同噪声。论文的 `alpha_e=1.0` 和 `alpha_h=0.2` 不相同，完整实现必须先解除权重绑定，再分别生成噪声。

### 3.3 Attention实现只覆盖了精确子集

当前实现包含：

- GQA一致的head permutation；
- RoPE可交换的旋转或signed permutation；
- V/O可逆坐标变换；
- FFN neuron permutation。

当前没有实现：

- Attention输入侧 `Q_hat_q/Q_hat_k/Q_hat_v`；
- Attention输出侧 `P_hat_o`；
- 论文Algorithm 2的成对 `R_hat_qk` 和 `H_hat_qk`；
- `BlockPerm(beta, gamma, zeta, mblocks)`动态窗口算法；
- 与论文一致的随机 `U_hat_vo` 及数值筛选；
- 层间扩维残差坐标。

当前 `signed_permutation` 分支是为保证数值精确而设计的工程近似。它几乎不改变权重统计量，不能承担隐藏 `tau` 的任务。

### 3.4 FFN实现缺少论文缩放和输入/输出Key Matrix

当前FFN只同步置换 `gate_proj/up_proj` 的行和 `down_proj` 的列。论文要求：

```text
W_gate_tilde = Q_gate @ W_gate @ Z_ffn
W_up_tilde   = Q_up @ W_up @ H_ffn @ Z_ffn
W_down_tilde = Z_ffn^-1 @ H_ffn^-1 @ W_down @ P_down
```

必须实现正缩放、逆缩放、输入Q和输出P，并验证SiLU与Hadamard乘法下的代数约束。仅置换仍允许跨层结构匹配。

### 3.5 RMSNorm没有进入真实模型

论文RMSNorm是近似项，也是多层误差主要来源。当前真实checkpoint仍使用原始Qwen RMSNorm，没有：

- 1152维RMSNorm；
- 每层 `kappa` 校准；
- 原 `W_norm` 向下游线性层融合；
- 对真实hidden-state分布的误差测量。

因此当前“无噪声等价”只证明了置换分支，不证明论文扩维分支可以稳定运行。

### 3.6 当前攻击名称与实际算法不一致

`run_privacy_suite.py` 的问题：

| 输出标签 | 当前实际操作 | 论文要求 |
|---|---|---|
| VMA/NN/IMA | noisy embedding cosine nearest neighbor | VMA应实现RowSort消元、邻居匹配和跨层投票；IMA需训练反演模型 |
| TFMA | cosine分数加少量token频率罚项 | 只使用客户端流量token频率与先验语料频率，报告Top-k恢复曲线 |
| SDA | known-plaintext coverage | 训练/调用substitution deciphering模型，报告BLEU-4与语义相似度 |
| IA | 对频次排序 | Gate-IA和Attn-IA应使用论文给出的权重不变量 |
| known-plaintext | 20条prompt逐位置匹配 | 应扫描0/10/100/1000/10000已知pair及泛化恢复率 |

现有99.22%结果可以作为“embedding nearest-neighbor attack”保留，但必须重命名，不能与论文攻击结果直接比较。

### 3.7 数据量不足，统计结论无效

- MMLU只跑abstract algebra 20题；
- PIQA只跑20题；
- C-Eval只跑18题；
- IFEval只跑5题；
- HumanEval只跑3题；
- privacy只抽128个词；
- known-plaintext只有20条prompt、104个unique tokens；
- latency只有20个请求。

这些数据只适合发现程序错误。最终门禁必须跑完整任务或预先冻结的、具备置信区间和统计功效的子集。

### 3.8 HumanEval下降尚未归因

无噪声FP32的普通prompt生成一致率为100%，但HumanEval 3题不一致。需要先排除：

- Windows与WSL的Transformers版本不同；
-原模型和私有模型加载dtype或attention backend不同；
- `PrivateTokenHFLM.generate_until()`对stop sequence、padding、EOS映射处理错误；
- logits近似误差在接近并列token处改变greedy选择；
- 3题样本造成的高方差。

在完成逐请求token/logit对照前，不调整模型参数来“修HumanEval”。

### 3.9 固定tau的流量泄漏是方案本身的边界

即使完整权重混淆阻止VMA，确定性token置换仍保留：

- 相同token相等关系；
- unigram和高阶n-gram频率；
- 已知系统prompt、chat template和协议固定片段；
- 多请求累积统计；
- 已知明文token对的永久映射。

论文明确把目标限定为受限攻击者，而非最坏情况安全。甲方PPT中“服务器无法从输出反推原始内容”必须改为有攻击预算、观测量和密钥生命周期条件的经验安全声明。

## 4. 新实现的代码边界

创建新分支或目录，不覆盖当前可运行对照实现：

```text
src/aloepri/
  keys/
    paper_keygen.py
    coordinate_graph.py
  transforms/
    paper_key_matrix.py
    paper_noise.py
    paper_embedding.py
    paper_attention.py
    paper_ffn.py
    paper_norm.py
  models/
    configuration_aloepri_qwen2.py
    modeling_aloepri_qwen2.py
    cache_aloepri.py
  conversion/
    paper_converter.py
    shard_reader.py
    shard_writer.py
    conversion_state.py
  attacks/
    vma_rowsort.py
    gate_ia.py
    attention_ia.py
    tfma_traffic.py
    sda_model.py
    known_plaintext_curve.py
    isa.py
    ima.py
  eval/
    paired_generation.py
    full_accuracy.py
    privacy_protocol.py
    latency_protocol.py
tests/
  paper_equivalence/
  paper_attacks/
  paper_integration/
configs/
  transform/paper_qwen05b.yaml
  eval/paper_accuracy.yaml
  eval/paper_privacy.yaml
  eval/paper_performance.yaml
```

当前 `qwen2.5-0.5b-full-exact` 更名为逻辑名称 `square_control`。新checkpoint逻辑名称使用 `paper_expand_h128`。

## 5. 论文忠实实现

### 5.1 Algorithm 1 Key Matrix

在 `paper_key_matrix.py` 实现：

```python
@dataclass(frozen=True)
class PaperKeyPair:
    p: Tensor       # [d, d + 2h]
    q: Tensor       # [d + 2h, d]
    b: Tensor       # [d, d]
    condition_b: float
    pq_relative_error: float
    spectral_norm_p: float
    spectral_norm_q: float
```

严格步骤：

1. CPU float64采样 `U in O_d`；
2. `V ~ N(0, 1/d)`，计算 `B=U+lambda*V`；
3. 若 `cond(B)>1e4` 则丢弃重采样；
4. 按论文低秩分解采样 `E1/E2/F1/F2`；
5. 使用SVD计算 `null(F^T)` 和 `null(E)`；
6. 构造 `C:[d,h]`、`D:[h,d]`；
7. 采样 `Z in O_(d+2h)`；
8. 构造论文 `P_hat/Q_hat`；
9. 检查 `||P_hat Q_hat-I||F/||I||F < 1e-10`；
10. 转float32后再次检查 `<1e-5`；
11. 转BF16后用真实hidden states检查相对误差。

注意：论文Algorithm 1排版中的拼接和转置必须用shape test消歧。每个中间量必须assert shape，禁止依赖矩阵乘法报错来发现问题。

### 5.2 残差坐标图

在 `coordinate_graph.py` 为每个残差边界定义坐标状态：

```text
S0: embedding output
S1: layer0 attention residual output
S2: layer0 FFN residual output
...
S48: layer23 FFN residual output
S49: final norm output
```

第一实现采用全局残差Key Matrix：所有残差状态使用同一个 `P_res`。原因是identity skip没有可变换权重；若子层输入输出使用不同P，残差两支不能直接相加。

```text
x_private = x_plain @ P_res
attention_private(x_private) ≈ attention_plain(x_plain) @ P_res
ffn_private(x_private)       = ffn_plain(x_plain) @ P_res
residual_private             = residual_plain @ P_res
```

只有全局P版本通过后，才实验“每层P+显式bridge”版本：

```text
bridge_i = Q_i @ P_(i+1)    # [D,D]
x_(i+1) = x_i @ bridge_i
```

bridge会增加新的矩阵乘法，不满足原论文近零在线开销主张，必须作为改进分支单独计时。

### 5.3 自定义Qwen2配置

`AloePriQwen2Config` 必须包含：

```json
{
  "model_type": "aloepri_qwen2",
  "hidden_size": 1152,
  "plain_hidden_size": 896,
  "head_dim": 64,
  "num_attention_heads": 14,
  "num_key_value_heads": 2,
  "intermediate_size": 4864,
  "expansion_h": 128,
  "tie_word_embeddings": false,
  "aloepri_transform_version": 2
}
```

不得通过 `hidden_size/num_attention_heads` 计算head_dim，因为1152不能被14整除。所有Q/K/V输出shape按显式head_dim构造：

```text
q_proj: [14*64, 1152]
k_proj: [2*64, 1152]
v_proj: [2*64, 1152]
o_proj: [1152, 14*64]
gate/up: [4864, 1152]
down: [1152, 4864]
embedding: [151936, 1152]
lm_head: [151936, 1152]
```

### 5.4 Embedding和Head

执行顺序固定为：

```text
读取FP32原权重
→ 解除tied weights
→ 分别计算sigma_e、sigma_h
→ 分别采样E_embed、E_head
→ 加alpha_e/alpha_h噪声
→ tau行/列置换
→ 右乘P_res或左乘Q_res
→ 写BF16权重
```

必须保存公开统计但不保存噪声本身：

```text
sigma_before, sigma_noise, alpha, mean_after, std_after,
max_abs_after, finite_ratio, seed_commitment_hash
```

噪声参数扫描：

```text
alpha_e = [0, 0.1, 0.25, 0.5, 1.0]
alpha_h = [0, 0.05, 0.1, 0.2]
```

先固定 `h=128, lambda=0.3`，不要同时搜索所有参数。

### 5.5 Attention

实现顺序：

1. 输入侧 `Q_res @ Wq/Wk/Wv`；
2. V/O的 `U_vo/U_vo^-1`；
3. GQA的 `tau_kv/tau_group`；
4. Q/K成对rotation；
5. Q/K成对scaling；
6. RoPE block permutation；
7. 输出侧 `Wo @ P_res`；
8. prefill/decode/KV Cache。

每个步骤都产生独立消融checkpoint。无噪声测试指标：

```text
q/k/v tensor relative_l2
attention score relative_l2
softmax probability max_abs
attention output relative_l2
cached key/value relative_l2
full layer output relative_l2
```

RoPE block permutation本身是近似变换，必须与精确rotation分开。先令 `beta=1` 验证精确链路，再使用论文 `beta=8, gamma=1000` 测量新增误差。

### 5.6 FFN

`H_ffn` 只允许正数对角缩放，避免改变SiLU符号区域。初始采样：

```text
log_scale ~ Uniform(-log(2), log(2))
scale = exp(log_scale)
```

逐步扩大范围，禁止一开始使用高条件数缩放。验证：

```text
SiLU(xW_gate) elementwise error
xW_up error
Hadamard product error
down projection error
full FFN relative_l2
```

### 5.7 RMSNorm

实现 `AloePriRMSNorm`，输入维度为1152。原始 `w_norm` 不直接扩展到1152，而是融合到相邻投影：

- input layernorm的 `w_norm` 融入Q/K/V输入侧权重；
- post-attention layernorm的 `w_norm` 融入gate/up输入侧权重；
- final norm的 `w_norm` 融入LM Head。

使用1024条校准样本采集每层真实输入，估计：

```text
kappa_layer = median(||x P_res||2 / ||x||2)
```

同时网格搜索补偿系数，直接最小化：

```text
relative_l2(
  private_norm(x @ P_res),
  plain_norm(x) @ P_res
)
```

校准集与精度评测集必须分离。每层保存kappa、样本数、分位数、训练误差和留出误差。

### 5.8 转换器

`paper_converter.py` 必须按层执行并保存状态：

```json
{
  "source_revision": "7ae557...",
  "phase": "layers",
  "completed_layers": [0, 1, 2],
  "current_shard": 1,
  "config_hash": "...",
  "key_commitment": "..."
}
```

恢复规则：

- config hash或source revision不同：拒绝恢复；
- 已完成shard先校验SHA-256；
- 单层写入临时safetensors后原子rename；
- 最终合并index，不重新加载完整原模型和输出模型；
- key bundle与checkpoint分目录；
- 服务端目录不得出现tau、inverse_tau、P、Q和随机种子。

## 6. 正确的攻击实现

### 6.1 攻击者视图

每个攻击进程只能读取：

```text
公开原模型
完整混淆模型
公开tokenizer和vocab
指定数量的混淆输入/输出流
指定数量的已知明文对
可选中间状态
公开先验语料
固定计算预算
```

攻击进程不得读取 `data/keys`。当前评测直接加载tau只允许用于最后计算ground-truth指标，必须由独立scorer执行。

### 6.2 VMA

实现论文Table 9的每一种权重组合，使用RowSort消元和跨层投票：

1. 构造已知原模型组合 `X`；
2. 构造混淆模型组合 `Y`；
3. 对行执行 `RowSort` 消除右侧置换；
4. 使用kNN或assignment恢复左侧置换；
5. 对24层结果投票；
6. 分别报告Embedding/Head、QK、Gate、Up、Down组合；
7. 报告Top-1、Top-10、全局key recovery和TTRSR。

全词表精确assignment成本过高时，先对固定4096 token子集验证，再使用FAISS分块近邻。最终VMA必须至少覆盖全部非special高频token。

### 6.3 Gate-IA和Attn-IA

Gate-IA按论文构造多层 `Avg(e W_gate)` 不变量；Attn-IA按RoPE 2×2 block构造二次型不变量。分别执行：

```text
tau only
tau + noise
tau + noise + P/Q
tau + noise + P/Q + head permutation
full AloePri
```

这样才能验证每个防御组件的边际贡献。

### 6.4 TFMA

TFMA不得访问权重。输入只能是：

- public prior corpus token counts；
- observed private ciphertext token counts。

实验量级：

```text
100, 1k, 10k, 100k, 1M observed tokens
zero-knowledge, domain-aware, distribution-aware priors
Top-1, Top-10, Top-100 token recovery
TTRSR随观测量曲线
```

### 6.5 SDA

复现论文引用的recurrence encoding路线：

1. 从公开语料生成随机substitution cipher训练对；
2. 转为symbol recurrence序列；
3. 训练小型causal Transformer；
4. 在未见tau和未见文本上解码；
5. 报告BLEU-4、token accuracy、实体恢复率和CosSim。

SDA模型和被测模型严格分离，不能使用真实tau生成测试训练对。

### 6.6 Known-plaintext、ISA和IMA

Known-plaintext扫描已知token pair数量：`0/10/100/1k/10k`，报告直接覆盖率和利用结构推断后的全局恢复率。

ISA至少分为attention score和hidden state两条路径，输出TTRSR。IMA需要独立训练反演模型；NN最近邻只能命名为NN attack，不能代替IMA。

## 7. 精度与性能复验

### 7.1 先修HumanEval一致性

对同一题、同一Linux环境、同一Transformers版本执行逐步对照：

```text
tokenized input IDs
private input IDs
每步public/private-recovered top-20 logits
argmax margin
EOS/stop sequence
generated token IDs
decoded source code
```

先比较原模型与无噪声扩维模型，再比较有噪声模型。若无噪声模型不同，修实现；若只有有噪声模型不同，计入精度权衡。

### 7.2 精度矩阵

阶段门禁使用两级数据：

| 等级 | 数据量 | 用途 |
|---|---:|---|
| smoke | 每任务20至100题 | 快速发现错误，不作验收 |
| acceptance | 任务完整官方split | 0.5B最终门禁 |

完整任务：MMLU、C-Eval、HumanEval、PIQA、IFEval。每个任务报告原始分、混淆分、绝对下降、相对下降、bootstrap 95% CI。HumanEval必须在断网、只读根目录、独立临时目录中执行生成代码。

### 7.3 性能

HF用于正确性，vLLM用于性能。固定：

```text
vLLM 0.11.1
VLLM_USE_V1=0
BF16
max_model_len=2048
prompt lengths=[32,128,512,1024]
output lengths=[32,128]
concurrency=[1,4,8]
warmup=20
measured requests>=200 per cell
```

分别比较原模型、square_control、paper_expand_h128。报告TTFT、TPOT、throughput、load time、peak VRAM的p50/p95/p99和置信区间。

## 8. 实施阶段和退出条件

### Phase R0：冻结证据与修正命名，2天

任务：

- 冻结现有checkpoint和artifact SHA-256；
- 将现有攻击结果重命名为实际算法名称；
- 生成实现覆盖矩阵；
- 固定Linux core和vLLM环境；
- 禁止覆盖现有结果。

退出条件：所有旧证据可复验，报告不再把NN攻击称为VMA/IMA/TFMA。

### Phase R1：Algorithm 1与线性夹具，4天

任务：

- 实现B/E/F/C/D/Z；
- shape和null-space测试；
- float64/float32/BF16误差测试；
- 随机线性层及真实Qwen权重测试。

退出条件：100个seed全部满足float64误差 `<1e-10`、float32 `<1e-5`，无条件数越界。

### Phase R2：自定义扩维Qwen骨架，5天

任务：

- config、Embedding、LM Head、显式head_dim；
- 1152维残差流；
- 自定义RMSNorm接口；
- HF保存重载；
- 不加噪、不做block permutation。

退出条件：单层和两层模型forward/generate/KV Cache可运行；checkpoint重载一致。

### Phase R3：Attention、FFN、Norm，8天

任务：

- 按5.5顺序逐机制接入；
- FFN缩放；
- RMSNorm校准；
- 24层完整checkpoint；
- 每层误差曲线。

退出条件：无噪声、`beta=1`版本100条greedy token完全一致；BF16误差不随层数指数增长。

### Phase R4：论文噪声和参数搜索，5天

任务：

- 解除tied weights；
- 实现alpha×weight-std噪声；
- 扫描alpha_e/alpha_h；
- 加入beta/gamma；
- 生成Pareto前沿。

退出条件：至少一个配置在smoke精度下降不超过3.5pp，同时VMA开发集恢复率明显低于square_control。

### Phase R5：攻击复现，8天

任务：

- VMA Table 9；
- Gate-IA、Attn-IA；
- TFMA exposure curve；
- SDA；
- known-plaintext；
- NN、ISA、IMA分开实现。

退出条件：每个指标可从独立attacker环境生成；攻击代码无法读取key目录；结果包含计算预算和观测量。

### Phase R6：HF/vLLM与完整验收，6天

任务：

- HF API回归；
- vLLM自定义模型注册；
- 完整五任务精度；
- 完整性能矩阵；
- 200请求流式API测试；
- 故障恢复。

退出条件：生成完整验收包并自动给出GO/NO-GO。

总工期：38个工程日。最低人员：模型系统1人、安全攻击1人；若仍由1人执行，按60至75个工作日安排。

## 9. 门禁

### G1：论文忠实度

- P/Q来自Algorithm 1，不得用半正交投影替代；
- `D=1152`真实进入checkpoint；
- Embedding/Head解除绑定并独立加噪；
- Attention、FFN、RMSNorm变换全部进入真实24层模型；
- manifest记录所有非秘密超参数。

任一不满足，结果只能标记为engineering approximation。

### G2：正确性

- `P@Q`达到dtype阈值；
- `beta=1、alpha=0`时100条greedy完全一致；
- prefill/decode/KV Cache一致；
- 保存重载后结果一致；
- 无NaN/Inf；
- 层误差无指数增长。

### G3：精度

五项完整任务分别绝对下降不超过3.5个百分点，不允许平均值掩盖单项失败。

### G4：隐私

先由甲方补齐PPT缺失阈值。建议最低门禁：

- VMA key recovery `<5%`；
- VMA/IA/ISA TTRSR `<5%`；
- PII实体恢复率 `<5%`；
- SDA BLEU-4 `<3`；
- known-plaintext和TFMA必须给出随观测量增长曲线；
- 任一攻击在约定预算内超过20%，整体NO-GO。

这些阈值只用于项目验收，不能视为论文形式安全证明。

### G5：性能

- vLLM TTFT和TPOT p50/p95分别劣化不超过15%；
- 峰值显存必须记录。扩维后参数量和KV Cache变化不得省略；
- 自定义模型必须真实生成，不能只通过registry import。

## 10. 若论文忠实分支仍失败

### 10.1 精度失败

按以下顺序定位，不同时调整多个变量：

1. `alpha=0, beta=1`：验证纯P/Q代数；
2. 开RMSNorm近似：验证kappa；
3. 开RoPE block permutation：定位attention误差；
4. 只开Embedding噪声；
5. 只开Head噪声；
6. 联合噪声。

若纯P/Q就失败，停止参数搜索，修模型实现。若只有噪声失败，输出accuracy/privacy Pareto，不伪造单一最优点。

### 10.2 VMA失败

若论文默认参数下VMA恢复率仍高：

- 检查是否存在tied/shared权重泄漏；
- 检查server checkpoint是否包含原始权重、转换seed或key metadata；
- 对每个Table 9组合定位泄漏源；
- 增加按层独立噪声或不同P，但重新验证残差和性能；
- 引入session级token perturbation只能作为新机制，不能继续称为原论文复现。

### 10.3 长期流量攻击失败

固定tau无法从根本上消除频率。可行改进按成本排序：

1. 按租户独立tau，隔离跨租户统计；
2. 按模型版本轮换tau，限制累计观测窗口；
3. 对高风险token使用homophonic mapping；
4. 在客户端加入受控token perturbation；
5. 对极高敏感场景使用TEE/FHE保护映射或前几层。

第2项会要求重新转换至少Embedding/Head及相关权重，不能按请求轮换。第3和第4项会改变生成分布，必须重新跑精度门禁。

### 10.4 形式安全无法成立

产品声明降级为：

```text
在给定公开模型、混淆模型、观测token数量、已知明文数量、
攻击算法和计算预算下，实测token/PII恢复率低于阈值。
```

不得声明“服务器无法恢复”“达到DP”或“提供密码学安全”。若甲方必须要最坏情况保证，技术路线应切换到TEE、MPC/FHE或混合架构。

## 11. 下一步立即执行顺序

```text
1. 创建 paper-faithful 分支和新目录，不改 square_control。
2. 修正攻击结果命名，生成 current_implementation_coverage.json。
3. 实现 Algorithm 1 原始P/Q及100-seed测试。
4. 实现 d=896 → D=1152 → d 的单线性层和两层残差夹具。
5. 编写 AloePriQwen2Config，固定 head_dim=64。
6. 转换 Embedding/Head，解除tie，alpha先设0。
7. 接入一层Attention、FFN、RMSNorm，输出逐张量误差。
8. 扩展到24层，完成无噪声100 prompt门禁。
9. 按论文标准差加入alpha_e/alpha_h噪声并扫描。
10. 按论文实现VMA后再决定是否继续SDA/IMA和完整benchmark。
```

第10步的停止条件：如果论文默认参数在正确VMA下仍恢复超过20%，暂停完整精度评测，先完成泄漏源消融和甲方技术评审。
