---
title: "AloePri 实现歧义 E08：Algorithm 2 是否共享同一次 Init 上下文"
date: "2026-08-06"
---

# 结论

本项不是可以直接判定的论文代数错误，而是影响实现的上下文歧义。Algorithm 2
第 5 行要求通过 Algorithm 1 采样 $Q_q,Q_k,Q_v,P_o$，但没有明确说明是共享
一次 `Init` 返回的 $B,B^{-1},E,F,Z$ 后反复调用 `KeyMatGen/InvKeyMatGen`，
还是每个矩阵都重新执行一次 `Init`。前一种解释可以生成相互兼容的矩阵；后一种
解释通常不兼容进入 Attention 的公共残差坐标。

# 输入侧约束

设明文残差为 $x\in\mathbb R^d$，服务器实际持有

$$
\widetilde x=xP,\qquad P\in\mathbb R^{d\times D}.
$$

对任一输入投影 $W_j$，论文形式的混淆权重为

$$
\widetilde W_j=Q_jW_jT_j,
\qquad Q_j\in\mathbb R^{D\times d}.
$$

服务器计算

$$
\widetilde x\widetilde W_j=x(PQ_j)W_jT_j.
$$

要等于期望的 $xW_jT_j$，每一条分支都必须满足

$$
\boxed{PQ_q=PQ_k=PQ_v=I_d}.
$$

若每个 $Q_j$ 都来自一次全新的 `Init`，Algorithm 1 只保证对应的
$P_jQ_j=I_d$，不能推出当前公共 $P$ 与它相消；在连续随机采样下，新的 $Q_j$
恰好满足固定约束 $PQ_j=I_d$ 的概率为零。

但若所有矩阵共享同一组 $B,B^{-1},E,F,Z$，只重新采样合法的 $C$ 和 $D$，则

$$
P(C)Q(D)
=[B\;C\;E]\begin{bmatrix}B^{-1}\\F\\D\end{bmatrix}
=I_d+CF+ED=I_d.
$$

因此同一个 `Init` 家族中的任意合法 $P(C)$ 与 $Q(D)$ 可以相消。这也与论文
正文中 “each key matrix can be canceled out by any of its inverses” 的文字一致。
问题在于 Algorithm 2 没有把这一共享上下文写进伪代码接口。

# 输出与残差约束

Attention 输出若落在 $P_o$ 坐标中，为

$$
\widetilde y=yP_o.
$$

残差连接要计算

$$
xP+ yP_o.
$$

若希望它表示 $(x+y)P$，必须满足

$$
\boxed{P_o=P}.
$$

否则两个加数属于不同的混淆坐标，逐元素相加没有协变含义。

# 可执行解释

实现必须固定解释。一种做法是让各分支共享 Algorithm 1 的同一个 `Init`
上下文；更保守的做法是固定每个残差流的公共 $P$，输出侧使用同一个 $P$，
输入侧只从该 $P$ 的右逆族中采样：

$$
Q_j=Q_0+N_j,\qquad PQ_0=I_d,\qquad PN_j=0.
$$

这样每个 $Q_j$ 可以不同，但都满足 $PQ_j=I_d$。当前 0.5B 实现采用更保守的
共享 $Q$ 和共享 $P$，并把注意力头内部的旋转、缩放、置换作为独立可逆变换。
该实现是一个明确的工程选择，不是对论文错误的修复。

# 来源定位

论文版本：arXiv:2603.01499v2，Algorithm 2 第 5--7 行，Section 5.2.1 的
$PQ=I$，以及残差组合语义；本地源文件
`data/papers/arxiv-2603.01499v2-source/chap/4.method.tex`。
