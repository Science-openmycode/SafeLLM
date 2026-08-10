---
title: "AloePri 论文勘误 E11：求和组合定理使用了未定义的去混淆映射"
date: "2026-08-08"
---

# 结论

论文第 7 页 Summation Composition Theorem 的边界条件写成

$$
\left.\psi_Y\right|_{C_1}=\left.\psi_X\right|_{C_2}.
$$

但协变混淆五元组只定义输出去混淆映射
$\psi_Y:\widetilde Y\rightarrow Y$，没有定义 $\psi_X$。这是确定的符号错误。

# 类型检查

定理中的两个函数为

$$
f:X\times\Theta\rightarrow Y,
\qquad
g:X\times\Xi\rightarrow Y.
$$

两条并行分支具有相同输入空间 $X$ 和相同输出空间 $Y$。要在混淆空间中进行
求和，必须同时共享：

$$
\left.\phi_X\right|_{C_1}=\left.\phi_X\right|_{C_2},
\quad
\left.\phi_Y\right|_{C_1}=\left.\phi_Y\right|_{C_2},
\quad
\left.\psi_Y\right|_{C_1}=\left.\psi_Y\right|_{C_2}.
$$

第三项两侧的定义域和值域均为 $\widetilde Y\rightarrow Y$，类型一致。原文右侧
$\psi_X|_{C_2}$ 既未定义，也不符合该定理的共同输出边界。

# 最小修正

把原式

$$
\left.\psi_Y\right|_{C_1}=\left.\psi_X\right|_{C_2}
$$

改为

$$
\left.\psi_Y\right|_{C_1}=\left.\psi_Y\right|_{C_2}.
$$

# 来源定位

论文版本：arXiv:2603.01499v2，第 7 页，Theorem 3；LaTeX 源文件
`chap/4.1.covariant_obfuse.tex` 第 422 行。该问题由独立子 Agent 复核确认。

