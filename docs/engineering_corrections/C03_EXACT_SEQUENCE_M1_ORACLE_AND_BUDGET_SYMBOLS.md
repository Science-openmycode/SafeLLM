# C03：论文序列 M1 精确 Oracle 与 Theorem 4 符号修正

## 修改结论

仓库现在明确分开两个实现：

1. `exact_sequence_m1_distribution()`：严格按论文 Definition 2 和指数机制枚举，作为小词表公式 oracle；
2. `perturb_tokens_m1()`：Qwen 产品使用的长度 1 精确机制逐 token 组合，继续标记为 `rmdp-tokenwise`。

不能把第二项写成论文长度 $l$ 的联合 M1 实测结果。

## 原文对象

论文把长度为 $l$、词表大小为 $n$ 的序列空间记作

$$
\mathbb Z_n^l.
$$

Definition 2 使用 $g_i\in S_n$ 的换位作用于每个 token 值，并定义

$$
d(x,y)=\min\left\{k:\;y=g_{k-1}\cdots g_0x,\;g_i\text{ 为换位}\right\}.
$$

在线指数机制为

$$
p_{M_1(x)}(y)=\frac{\exp(-\epsilon_1 d(x,y))}
{\sum_z\exp(-\epsilon_1 d(x,z))}.
$$

## 为什么预算参数必须使用词表大小

原文同时出现两个维度：

- $n$：token alphabet 大小，置换群为 $S_n$；
- $l$：token 序列长度。

Theorem 4 中的 $(n-1)$ 来自 $S_n$ 的置换几何，不是序列长度。因此预算函数的首选参数改回 `vocab_size`。旧 `sequence_length` 只作为兼容别名保留并发出弃用警告。

Qwen2.5 的预算命令应使用：

```powershell
uv run aloepri rmdp-budget `
  --epsilon1 1.0 `
  --vocab-size 151936 `
  --embedding-sigma 1.0 `
  --head-sigma 1.0 `
  --embedding-top1 1.0 `
  --embedding-top2 1.0 `
  --head-top1 1.0 `
  --head-top2 1.0
```

## 精确枚举算法

对 `vocab_size <= 8`：

1. 枚举 $S_n$ 的全部 $n!$ 个置换；
2. 对每个置换 $g$ 计算 $y=gx$；
3. 用

   $$
   d_p(g,id)=n-c(g)
   $$

   计算置换的最少换位数，其中 $c(g)$ 是循环个数；
4. 多个置换得到同一 $y$ 时取最小距离；
5. 对所有可达 $y$ 计算 `softmax(-epsilon1 * distance)`；
6. 可选使用该精确分布抽样。

复杂度为

$$
O(n!\,l),
$$

所以该实现用于公式核对、反例和统计测试，不用于 Qwen 的 151936 词表在线推理。

## 已验证实例

当 $n=3$、$x=(0,1)$ 时，精确结果为：

| 输出 $y$ | 距离 $d(x,y)$ |
|---|---:|
| $(0,1)$ | 0 |
| $(0,2)$ | 1 |
| $(1,0)$ | 1 |
| $(2,1)$ | 1 |
| $(1,2)$ | 2 |
| $(2,0)$ | 2 |

测试同时验证

$$
\frac{p((1,0))}{p((0,1))}=e^{-\epsilon_1}.
$$

## 代码与测试

- `src/aloepri/privacy/rmdp.py`
  - `exact_sequence_m1_distribution()`
  - `sample_exact_sequence_m1()`
  - `calculate_rmdp_budget(vocab_size=...)`
- `src/aloepri/cli.py`
  - `rmdp-budget --vocab-size`
- `tests/unit/test_rmdp.py`
  - 全距离表、概率比、归一化、相等模式不变性和固定种子抽样。

执行：

```powershell
uv run pytest -q tests/unit/test_rmdp.py
```
