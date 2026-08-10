---
title: "AloePri 论文勘误 E03：BlockPerm 参数 gamma 在采样分布中未使用"
date: "2026-08-06"
---

# 结论

这是一个确定的伪代码接口错误。Algorithm 2 把 $\gamma$ 定义为窗口采样参数，
调用形式为

$$
Z_{\mathrm{block}}\leftarrow
\operatorname{BlockPerm}(\beta,\gamma,\zeta,m_{\mathrm{blocks}}),
$$

附录又给出默认值 $\gamma=10^3$。但 BlockPerm 第 14 行实际使用的分布是

$$
u=\operatorname{softmax}
\left(\{\zeta_{t+i}-\zeta_t\mid 1\le i\le c\}\right),
$$

右侧完全没有 $\gamma$。

# 为什么该参数在原式中不可能生效

设候选窗口的 logit 向量为

$$
a_i=\zeta_{t+i}-\zeta_t.
$$

原伪代码定义的概率为

$$
p_i=\frac{e^{a_i}}{\sum_j e^{a_j}}.
$$

对 $\gamma$ 求偏导，因 $a_i$ 与 $\gamma$ 无关，有

$$
\frac{\partial p_i}{\partial\gamma}=0.
$$

所以按论文原式运行时，$\gamma=1$、$10^3$ 或任意正数都会得到完全相同的
采样分布；附录中的 $\gamma$ 设置无法被复现，也无法做该参数的消融。

# 两种可能修正

若 $\gamma$ 表示逆温度，一个自然修正是

$$
p_i=\operatorname{softmax}(\gamma a_i).
$$

若 $\gamma$ 表示温度，则应写为

$$
p_i=\operatorname{softmax}(a_i/\gamma).
$$

这两种解释在 $\gamma=10^3$ 时行为相反，不能由现有文字唯一确定。因此仓库
显式提供 `paper-distribution-boundary-corrected` 与 `gamma-corrected` 两种模式。
前者只保留论文展示的 softmax 分布，同时修复循环和边界；后者再加入
$\gamma$。任何实验记录必须写明所选模式，两者都不能称为对原伪代码的逐字执行。

# 来源定位

论文版本：arXiv:2603.01499v2，Algorithm 2 的输入说明、第 3、9、14 行，以及
Appendix D.2 超参数表；本地源文件
`data/papers/arxiv-2603.01499v2-source/chap/4.method.tex` 与
`chap/appx/exp.tex`。
