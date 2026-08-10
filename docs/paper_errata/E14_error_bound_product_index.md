---
title: "AloePri 论文勘误 E14：整模误差上界的连乘下标未使用求积变量"
date: "2026-08-08"
---

# 结论

论文 Section 5.4 定义

$$
\mathcal M_i=M^{head}\prod_{j=i+1}^{L}M_i^{decoder}.
$$

连乘指标是 $j$，但被乘项仍写 $M_i^{decoder}$。按字面计算得到同一个常数的
幂，不能表示误差经过后续不同 decoder 层逐层传播。被乘项下标应为 $j$。

# 推导

第 $i$ 层引入误差 $e_i$ 后，误差依次经过第 $i+1$ 到第 $L$ 层和 model head。
若第 $j$ 层的 Lipschitz 常数为 $M_j^{decoder}$，则

$$
\|\Delta_L\|
\le
\left(\prod_{j=i+1}^{L}M_j^{decoder}\right)\|\Delta_i\|,
$$

再经过 head 得到

$$
\mathcal M_i
=M^{head}\prod_{j=i+1}^{L}M_j^{decoder}.
$$

原式则为

$$
M^{head}\left(M_i^{decoder}\right)^{L-i},
$$

只有所有后续层常数都恰好等于第 $i$ 层常数时才与正确传播式一致。

# 来源定位

论文版本：arXiv:2603.01499v2，第 11 页 Section 5.4；LaTeX 源文件
`chap/4.method.tex` 第 261 行。

