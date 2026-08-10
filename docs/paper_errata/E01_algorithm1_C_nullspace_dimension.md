---
title: "AloePri 论文勘误 E01：矩阵 C 的零空间对象与列维度不一致"
date: "2026-08-06"
---

# 结论

这是一个确定的维度错误。论文 Algorithm 1 第 10 行写明

$$
C\in\mathbb{R}^{d\times h},\qquad
\text{columns}(C)\subseteq\operatorname{null}(F^\mathsf{T}),
$$

而同一算法第 5 行给出 $F\in\mathbb{R}^{h\times d}$。因此
$F^\mathsf{T}\in\mathbb{R}^{d\times h}$，其零空间是

$$
\operatorname{null}(F^\mathsf{T})
=\{v\in\mathbb{R}^{h}:F^\mathsf{T}v=0\}
\subseteq\mathbb{R}^{h}.
$$

但是 $C$ 的每一列属于 $\mathbb{R}^{d}$。当 $d\ne h$ 时，一个
$d$ 维列向量不可能被声明为 $F^\mathsf{T}$ 的 $h$ 维零空间元素。
论文实验使用的隐藏维度和扩维参数也不是 $d=h$，所以这不是可忽略的特殊情形。

# 由 $PQ=I$ 反推正确条件

Algorithm 1 构造

$$
P_0=[B\;C\;E]\in\mathbb{R}^{d\times(d+2h)},
\qquad
Q_0=\begin{bmatrix}B^{-1}\\F\\D\end{bmatrix}
\in\mathbb{R}^{(d+2h)\times d}.
$$

两者相乘得到

$$
P_0Q_0=BB^{-1}+CF+ED=I_d+CF+ED.
$$

要使 $P_0Q_0=I_d$，需要 $CF=0$ 和 $ED=0$。对于
$C\in\mathbb{R}^{d\times h}$，应当要求 $C$ 的每一行 $c_i^{\mathsf{T}}$
满足

$$
c_i^\mathsf{T}F=0
\iff F^\mathsf{T}c_i=0,
\qquad c_i\in\mathbb{R}^{h}.
$$

因此形状一致、且能推出 $CF=0$ 的表述应为：

$$
\boxed{\text{rows}(C)\subseteq\operatorname{null}(F^\mathsf{T})}.
$$

# 最小修正与实现

把第 10 行的 “columns” 改为 “rows” 即可。仓库
`src/aloepri/transforms/paper_key_matrix.py` 采用这一修正：先求
$\operatorname{null}(F^\mathsf{T})$ 的基，再按行构造 $C$，并直接测试
$P Q\approx I_d$。该修正不改变论文的矩阵尺寸、扩维规模或安全参数。

# 来源定位

论文版本：arXiv:2603.01499v2，Algorithm 1，第 5、10、11、15 行；本地源文件
`data/papers/arxiv-2603.01499v2-source/chap/4.method.tex`。
