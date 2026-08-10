# E17：Definition 2 在整个 Token 序列空间上不是有限度量

## 结论

论文 Definition 2 使用同一个词表置换 $g\in S_n$ 作用于序列的所有位置。该作用保持序列位置之间的“token 是否相等”关系。因此，并非任意 $x,y\in\mathbb Z_n^l$ 都能通过有限次词表换位互相转换。

所以原定义在每个置换轨道内部是度量，但在整个 $\mathbb Z_n^l$ 上只能解释为取值允许 $+\infty$ 的扩展度量。论文把它直接称为整个空间上的 metric，缺少这一限制。

## 不变量证明

对任意词表置换 $g$ 和任意两个位置 $i,j$，由于 $g$ 是双射，

$$
x_i=x_j
\iff g(x_i)=g(x_j).
$$

因此任意有限个换位的复合也保持相等模式：

$$
x_i=x_j
\iff (g_k\cdots g_1x)_i=(g_k\cdots g_1x)_j.
$$

若 $x$ 和 $y$ 的相等模式不同，则不存在论文要求的有限换位链。

## 最小反例

取 $n\ge2$、$l=2$：

$$
x=(0,0),\qquad y=(0,1).
$$

对任意 $g\in S_n$，

$$
gx=(g(0),g(0)),
$$

两个位置始终相等，不可能得到两个位置不同的 $(0,1)$。所以

$$
d((0,0),(0,1))=+\infty.
$$

## 对 M1 的影响

若按

$$
p_{M_1(x)}(y)\propto e^{-\epsilon_1d(x,y)}
$$

解释，则不同轨道的输出概率为零。M1 只能在输入 $x$ 的置换轨道上归一化。例如 $x=(0,0,1)$ 的任何可达输出都必须满足前两个 token 相等；$(0,1,1)$ 的概率为零。

这意味着论文若要在整个序列空间上声明 RmDP，需要明确使用扩展度量，或者重新定义能连接不同相等模式的邻接操作。

## 工程处理

- 精确 oracle 只枚举 $x$ 的可达轨道；
- 不可达序列不写入 outcome 表，等价于概率为零；
- 产品逐 token M1 每次处理长度 1 的序列；长度 1 没有相等模式分裂问题；
- 报告不得把逐 token 组合称为原文长序列联合 M1；
- 不擅自为论文添加 token 替换距离，因为这会改变 Theorem 4 的假设。

## 代码证据

- `src/aloepri/privacy/rmdp.py:exact_sequence_m1_distribution`
- `tests/unit/test_rmdp.py:test_exact_sequence_m1_preserves_equality_pattern_and_is_reproducible`

验证命令：

```powershell
uv run pytest -q tests/unit/test_rmdp.py
```
