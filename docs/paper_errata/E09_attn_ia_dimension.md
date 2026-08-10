---
title: "AloePri 论文勘误 E09：Appendix Attn-IA 不变量的矩阵乘法维度不成立"
date: "2026-08-06"
---

# 结论

Appendix D.1 给出的 Attn-IA 等式按其自身变量定义无法进行矩阵乘法，因此不能
直接实现。这是一个确定的维度错误，不是缺少数值超参数。

# 按论文定义检查维度

设词表大小为 $V$，隐藏维度为 $d$，单个 Attention head 的输出维度为 $r$。
论文定义

$$
W_e\in\mathbb R^{V\times d},\qquad
W_{\mathrm{query}}^{(i)}\in\mathbb R^{d\times r},
$$

从而

$$
Q=W_eW_{\mathrm{query}}^{(i)}\in\mathbb R^{V\times r}.
$$

对列索引 $j$，有

$$
Q[:,j]\in\mathbb R^{V},
\qquad
Q^\mathsf{T}[:,j]\in\mathbb R^{r}.
$$

论文写出的中间项

$$
Q^\mathsf{T}[:,j]Q[:,j]
$$

试图把一个 $r$ 维列向量与一个 $V$ 维列向量相乘；既不是合法内积，也不是
维度明确的外积。通常 $V\ne r$，Qwen2.5-0.5B 中更是
$V=151936$、$r=64$。

即使把 $j$ 理解为二维 RoPE block，令 $Q_J\in\mathbb R^{V\times2}$，Gram
矩阵 $Q_J^\mathsf{T}Q_J\in\mathbb R^{2\times2}$ 可以求逆，但论文等式外侧使用
的是原始 embedding $e\in\mathbb R^d$，表达式

$$
e(Q_J^\mathsf{T}Q_J)^{-1}e^\mathsf{T}
$$

仍因 $d\ne2$ 而没有定义。

# 一个维度成立的候选不变量

若目标是利用任意可逆的 block 内右变换，可以先取 token $t$ 在该 block 的
投影行向量

$$
q_{t,J}=Q[t,J]\in\mathbb R^{1\times2},
$$

再定义 leverage score

$$
\ell_{t,J}=q_{t,J}(Q_J^\mathsf{T}Q_J)^{-1}q_{t,J}^\mathsf{T}.
$$

若 $Q_J$ 右乘可逆矩阵 $A$，则

$$
\begin{aligned}
\widetilde\ell_{t,J}
&=q_{t,J}A\left(A^\mathsf{T}Q_J^\mathsf{T}Q_JA\right)^{-1}
A^\mathsf{T}q_{t,J}^\mathsf{T}\\
&=q_{t,J}(Q_J^\mathsf{T}Q_J)^{-1}q_{t,J}^\mathsf{T}
=\ell_{t,J}.
\end{aligned}
$$

该式维度成立，但它是对论文公式的候选修正，不能在作者未确认前当作原论文攻击。
因此仓库 `run_attn_ia.py` 将现有实现明确标记为
`paper_exact: false` 和 `dimensionally-valid_qk_pair_proxy`。

# 来源定位

论文版本：arXiv:2603.01499v2，Appendix D.1，Attn-IA 段落；本地源文件
`data/papers/arxiv-2603.01499v2-source/chap/appx/exp.tex`。
