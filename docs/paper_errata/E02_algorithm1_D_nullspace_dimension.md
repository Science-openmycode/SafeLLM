---
title: "AloePri 论文勘误 E02：矩阵 D 的零空间对象与行维度不一致"
date: "2026-08-06"
---

# 结论

这是一个确定的维度错误。论文 Algorithm 1 第 14 行写明

$$
D\in\mathbb{R}^{h\times d},\qquad
\text{rows}(D)\subseteq\operatorname{null}(E),
$$

而第 4 行给出 $E\in\mathbb{R}^{d\times h}$。所以

$$
\operatorname{null}(E)
=\{v\in\mathbb{R}^{h}:Ev=0\}
\subseteq\mathbb{R}^{h}.
$$

$D$ 的每一行却属于 $\mathbb{R}^{d}$。当 $d\ne h$ 时，论文声明的
“行属于 $\operatorname{null}(E)$”没有定义。

# 由恒等式反推正确对象

Algorithm 1 的未旋转矩阵满足

$$
[B\;C\;E]
\begin{bmatrix}B^{-1}\\F\\D\end{bmatrix}
=I_d+CF+ED.
$$

要消去最后一项，需要 $ED=0$。把 $D$ 写成列向量拼接

$$
D=[d_1\;d_2\;\cdots\;d_d],
\qquad d_j\in\mathbb{R}^{h},
$$

则

$$
ED=0
\iff Ed_j=0\quad(1\le j\le d)
\iff d_j\in\operatorname{null}(E).
$$

因此正确表述应为

$$
\boxed{\text{columns}(D)\subseteq\operatorname{null}(E)}.
$$

# 最小修正与实现

把第 14 行的 “rows” 改为 “columns”。仓库
`src/aloepri/transforms/paper_key_matrix.py` 按列构造 $D$，从而使
$ED=0$。与 E01 的修正合并后，正交矩阵 $Z$ 只做坐标旋转：

$$
P=[B\;C\;E]Z,\qquad
Q=Z^\mathsf{T}\begin{bmatrix}B^{-1}\\F\\D\end{bmatrix},
$$

所以 $PQ=I_d$ 仍然成立。

# 来源定位

论文版本：arXiv:2603.01499v2，Algorithm 1，第 4、14、15 行；本地源文件
`data/papers/arxiv-2603.01499v2-source/chap/4.method.tex`。
