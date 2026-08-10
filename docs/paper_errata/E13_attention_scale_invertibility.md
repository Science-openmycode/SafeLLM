---
title: "AloePri 论文勘误 E13：Attention scaling 的采样域包含不可逆值"
date: "2026-08-08"
---

# 结论

Algorithm 2 第 2 行写“Sample $s_i\in\mathbb R$”，随后第 6 行使用
$\hat H_{qk}^{-1}$。若任一 $s_i=0$，则 $\hat H_{qk}$ 不可逆，算法没有定义。
因此采样域至少必须排除零。

# 推导

论文定义

$$
\hat H_{qk}=\operatorname{Diag}
\left(s_1I_2,\ldots,s_{d_{head}/2}I_2\right).
$$

其行列式为

$$
\det(\hat H_{qk})=\prod_{i=1}^{d_{head}/2}s_i^2.
$$

所以

$$
\hat H_{qk}^{-1}\text{ 存在}
\iff
\forall i,\ s_i\neq0.
$$

“$s_i\in\mathbb R$”没有表达这个必要条件，也没有给出实际采样分布。

# 最小修正与实现

最小修正是写为 $s_i\in\mathbb R\setminus\{0\}$。为避免极小绝对值造成 BF16
数值放大，本仓库从有界正区间采样：

$$
s_i\sim\operatorname{LogUniform}(0.5,2.0).
$$

正区间是论文约束的一个有效实例；具体分布属于论文未公开细节，不宣称为作者
原始分布。

# 来源定位

论文版本：arXiv:2603.01499v2，第 9 页 Algorithm 2 第 2、6 行；LaTeX 源文件
`chap/4.method.tex` 第 118--120、133 行。

