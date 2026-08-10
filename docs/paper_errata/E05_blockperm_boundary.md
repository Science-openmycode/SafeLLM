---
title: "AloePri 论文勘误 E05：BlockPerm 边界少一个 RoPE block"
date: "2026-08-06"
---

# 结论

在对 E04 作最自然修复 $t\leftarrow t+w$ 后，论文给出的边界仍然只会生成
$m_{\mathrm{blocks}}-1$ 阶矩阵，而不是所需的 $m_{\mathrm{blocks}}$ 阶矩阵。
因此本项结论是有条件的：在采用这一自然循环修复时，off-by-one 可以被严格推出；
原文因循环不终止，本身不能产生任何有限阶结果。

# 推导

论文采用一基索引，初始化 $t=1$，循环条件 $t<m$，并设置

$$
c=\min(\beta,m-t),\qquad 1\le w\le c,
$$

其中 $m=m_{\mathrm{blocks}}$。每轮追加一个 $w\times w$ 置换，并令
$t\leftarrow t+w$。设已经追加的总阶数为 $S$。初始 $S=0,t=1$，且每轮

$$
S' = S+w,\qquad t'=t+w.
$$

因此不变量为

$$
S=t-1.
$$

循环在 $t=m$ 时结束，于是

$$
S=m-1.
$$

最终 $\operatorname{BlockDiag}(\mathcal U)$ 是 $(m-1)\times(m-1)$，不能作用于
含 $m$ 个二维 RoPE block 的坐标。

# 一致的修正版

一种完整的一基索引写法是

$$
\begin{aligned}
&\textbf{while }t\le m,\\
&c=\min(\beta,m-t+1),\\
&a_i=\zeta_{t+i-1}-\zeta_t,\quad 1\le i\le c,\\
&w\sim\operatorname{Categorical}(\operatorname{softmax}(a)),\\
&t\leftarrow t+w.
\end{aligned}
$$

此时终止时 $t=m+1$，由同一不变量得到 $S=m$。仓库内部改用零基索引
`start=0`、条件 `start < blocks` 和
`available=min(beta, blocks-start)`，从结构上避免该边界错误。

# 来源定位

论文版本：arXiv:2603.01499v2，Algorithm 2，BlockPerm 第 10--18 行；本地源文件
`data/papers/arxiv-2603.01499v2-source/chap/4.method.tex`。
