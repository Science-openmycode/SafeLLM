# E16：不同 RoPE 频率块之间的置换不是函数保持变换

## 结论

Algorithm 2 的 RoPE block permutation 只有在被交换的二维块使用相同旋转频率时才与原始 RoPE 对易。Qwen2.5 的各二维块频率不同，因此 $\beta>1$ 的跨频率置换配合未修改的 $R(t)$ 会改变 Attention score。若论文意图保持函数，运行时必须同步使用 $B^{\mathsf T}R(t)B$；论文没有写出这一步。

## RoPE 与块置换

第 $i$ 个二维块在位置 $t$ 的旋转为

$$
R_i(t)=
\begin{bmatrix}
\cos(t\omega_i)&-\sin(t\omega_i)\\
\sin(t\omega_i)&\cos(t\omega_i)
\end{bmatrix}.
$$

整体 RoPE 为

$$
R(t)=\operatorname{BlockDiag}(R_1(t),\ldots,R_m(t)).
$$

设置换矩阵 $Z$ 交换块 $i$ 与 $j$。要保持旋转后的坐标变换，必须满足

$$
ZR(t)=R(t)Z.
$$

观察交换后的两个块可知，上式等价于

$$
R_i(t)=R_j(t), \qquad \forall t,
$$

进而要求

$$
\omega_i=\omega_j.
$$

Qwen 的频率为

$$
\omega_i=\theta^{-2i/d_{\mathrm{head}}},
$$

不同 $i$ 的频率严格不同，所以一般的非平凡 block permutation 不与 RoPE 对易。

## 对 Attention score 的影响

若 Query 和 Key 在投影后使用同一块置换，未施加 RoPE 时内积可以保持；施加 RoPE 后，私有内积包含

$$
qZ R(t_q)R(t_k)^\top Z^\top k^\top,
$$

而明文内积为

$$
q R(t_q)R(t_k)^\top k^\top.
$$

只有当 $Z$ 与两个位置的 RoPE 都对易时两者才相等。跨频率块置换不满足该条件。

## 工程处理

- 字面论文分支保留未修改 $R(t)$，用于证明 $\beta>1$ 时不等价。
- 修正分支把每层、每 KV 组的运行时旋转改为 $R_B(t)=B^{\mathsf T}R(t)B$。
- 生产 checkpoint `qwen05b-product-v31-blockperm8` 使用论文默认 $\beta=8$ 和修正运行时，不再通过把 $\beta$ 降为 1 回避问题。
- checkpoint、config 和离线 key 分别保存或推导同一 BlockPerm 顺序；验收器逐层三方核对。
- 报告必须把该运行时共轭标为论文缺失步骤，不能写成原文字面公式。

## 代码位置

- `src/aloepri/transforms/qwen_structural.py`：生成置换和频率表。
- `src/aloepri/models/modeling_aloepri_qwen2.py`：HF 执行 $B^{\mathsf T}R(t)B$。
- `src/aloepri/serving/vllm_qwen2.py`：vLLM 同步执行并严格加载审计张量。
- `src/aloepri/serving/sglang_models/aloepri_qwen2.py`：SGLang 同步执行并严格加载审计张量。
- 相关函数：`make_dynamic_rope_block_order`、`rope_block_frequencies`。
- `scripts/inspect_structural_key.py`：统计被移动的 RoPE 块及 Q/K 代数误差。
- `scripts/verify_layerwise_equivalence.py`：核对 RoPE 后 Q/K、score 和 softmax。

## 0.5B 实测

| 配置 | 24 层移动块数 | Prefill 平均绝对误差 | Prefill Top-1 | 16-token greedy |
|---|---:|---:|---:|---:|
| $\beta=8$，未同步 RoPE（历史反例） | 603 | 0.01204236 | 97.22% | 当前样本一致 |
| $\beta=1$，v21 | 0 | 0.00001842 | 100% | 一致 |
| $\beta=1$，v25 | 0 | 0.00001521 | 100% | 一致 |
| $\beta=8$，v31 同步 RoPE | 非平凡，24 层 | 有噪声，逐层仅诊断 | 94.44% 全位置 | 16-token 一致 |

$\beta=8$ 的 v31 已完成 316/316 checkpoint 张量公式重建，并在 HF、vLLM 和 SGLang 上得到相同的 32-token 私有 greedy 序列。v31 含 Embedding/Head 噪声，因此不把逐层张量相等作为门禁；同步 RoPE 的无噪声等价由 `tests/unit/test_qwen_structural.py` 的独立反例和修正测试覆盖。
