---
title: "AloePri 论文勘误 E04：BlockPerm 循环变量 t 未更新"
date: "2026-08-06"
---

# 结论

这是一个确定的控制流错误。BlockPerm 初始化 $t=1$，循环条件为
$t<m_{\mathrm{blocks}}$，但循环体只计算 $c$、采样 $w$、采样置换矩阵并追加到
列表，没有任何语句修改 $t$。

# 形式化证明

设 $m_{\mathrm{blocks}}>1$。初始化后

$$
t_0=1<m_{\mathrm{blocks}}.
$$

由于循环体中没有对 $t$ 的赋值，对任意迭代次数 $n\ge0$，有

$$
t_{n+1}=t_n.
$$

归纳得到

$$
t_n=1<m_{\mathrm{blocks}},\qquad \forall n\ge0.
$$

因此循环条件永久为真，算法不返回 $Z_{\mathrm{block}}$。Qwen2.5-0.5B 的
$d_{\mathrm{head}}=64$，故 $m_{\mathrm{blocks}}=32>1$，必然触发该问题。

# 最小修正

在每次追加一个大小为 $w$ 的窗口置换后加入

$$
\boxed{t\leftarrow t+w}.
$$

这只能修复终止性；原伪代码的边界仍少处理一个 block，另见 E05。仓库实现
明确加入了该更新，并用“结果是 $0,\ldots,m_{\mathrm{blocks}}-1$ 的双射”作为
单元测试条件。

# 来源定位

论文版本：arXiv:2603.01499v2，Algorithm 2，BlockPerm 第 10--17 行；本地源文件
`data/papers/arxiv-2603.01499v2-source/chap/4.method.tex`。
