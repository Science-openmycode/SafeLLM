---
title: "AloePri 论文勘误 E10：BlockPerm 频率指数与 Qwen RoPE 不一致"
date: "2026-08-06"
---

# 结论

Algorithm 2 的 BlockPerm 使用了比 Qwen 实际 RoPE 大两倍的指数。若
`m_blocks=d_head/2`，论文频率不是 Qwen 模型中被置换的真实 RoPE block
频率。因此，按论文频率计算的窗口采样概率不能忠实反映 Qwen 各 block 的频率
距离。

# 论文公式

论文 Algorithm 2 定义

$$
\zeta_i=\zeta^{-2(i-1)/m_{\mathrm{blocks}}},
\qquad 1\le i\le m_{\mathrm{blocks}}.
$$

RoPE 的每个二维 block 占两个 head 坐标，因此

$$
m_{\mathrm{blocks}}=\frac{d_{\mathrm{head}}}{2}.
$$

代入后，论文公式成为

$$
\zeta_i=\zeta^{-4(i-1)/d_{\mathrm{head}}}.
$$

# Qwen 实际频率

Qwen/Transformers 对偶数坐标索引 $0,2,\ldots,d_{\mathrm{head}}-2$ 计算

$$
\operatorname{inv\_freq}_j
=\theta^{-2j/d_{\mathrm{head}}},
\qquad 0\le j<m_{\mathrm{blocks}}.
$$

由于 $d_{\mathrm{head}}=2m_{\mathrm{blocks}}$，等价于

$$
\operatorname{inv\_freq}_j=\theta^{-j/m_{\mathrm{blocks}}}.
$$

令 $j=i-1$，论文指数为 $-2j/m_{\mathrm{blocks}}$，Qwen 指数为
$-j/m_{\mathrm{blocks}}$，两者相差两倍。

# 最小修正

在 Qwen 上，BlockPerm 应使用

$$
\zeta_i=\zeta^{-(i-1)/m_{\mathrm{blocks}}}.
$$

仓库同时保留两个显式模式：

- `qwen-actual`：使用模型真实 RoPE 频率，作为 Qwen 工程分支；
- `paper-literal`：使用论文原式，仅用于验证论文公式本身。

两种模式必须写入 checkpoint 元数据，实验结果不能混用。

# 影响范围

频率只参与 BlockPerm 窗口长度的采样，不直接替换模型的 RoPE 算子。当
$\beta=1$ 时，block 顺序恒等，两个公式没有输出差异；当 $\beta>1$ 时，频率
指数会改变窗口分布和最终 block permutation，进而改变精度与攻击结果。

# 来源定位

论文版本：arXiv:2603.01499v2，Algorithm 2 `BlockPerm`；本地源文件
`data/papers/arxiv-2603.01499v2-source/chap/4.method.tex`。Qwen 对照实现位于本地
Transformers 的 `models/qwen2/modeling_qwen2.py`。
