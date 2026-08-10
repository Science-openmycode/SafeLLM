# E15：一般矩形密钥矩阵下，标量 $\kappa$ 不能使 RMSNorm 精确协变

## 结论

论文给出的标量 $\kappa$ 只能校正随机输入分布下的平均范数偏差，不能对一般的 Algorithm 1 矩形矩阵 $P$ 保证逐输入函数等价。若要求无噪声模型与原模型逐 token 等价，必须改用明文度量下的 RMSNorm。

## 论文原式

设明文隐藏维度为 $d$，私有维度为 $D=d+2h$，私有残差状态为

$$
z=xP, \qquad P\in\mathbb{R}^{d\times D}.
$$

论文用

$$
\kappa=\mathbb{E}\left[\frac{\lVert xP\rVert_2}{\lVert x\rVert_2}\right]
$$

调整私有 RMSNorm，并把原 RMSNorm 权重融合到相邻线性层。

忽略逐元素权重后，明文 RMSNorm 的标量分母为

$$
r_d(x)=\sqrt{\frac{x x^\top}{d}+\varepsilon},
$$

论文私有路径使用

$$
\widetilde r_D(xP)=\sqrt{\frac{xPP^\top x^\top}{D}+\varepsilon}.
$$

## 标量校正成立的必要条件

先取 $\varepsilon=0$。若单个标量 $\kappa$ 对任意 $x$ 都精确成立，则必须有

$$
\frac{\kappa}{\widetilde r_D(xP)}=\frac{1}{r_d(x)}, \qquad \forall x.
$$

平方并整理得到

$$
xPP^\top x^\top=\frac{D\kappa^2}{d}xx^\top, \qquad \forall x.
$$

因此必要且充分的矩阵条件为

$$
PP^\top=\frac{D\kappa^2}{d}I_d.
$$

即 $P$ 的所有行必须两两正交且具有相同范数。论文 Algorithm 1 只保证存在 $Q$ 使 $PQ=I_d$，并不保证上述紧框架条件。

## 反例

取 $d=2,D=3$，

$$
P=\begin{bmatrix}
1&0&0\\
0&2&0
\end{bmatrix}.
$$

对 $x_1=[1,0]$，有 $\lVert x_1P\rVert_2/\lVert x_1\rVert_2=1$；对 $x_2=[0,1]$，该比值为 $2$。所以不存在同时适用于 $x_1$ 和 $x_2$ 的标量 $\kappa$。

## 精确修正式

由 Algorithm 1 的右逆关系

$$
PQ=I_d
$$

定义服务器可公开使用的派生度量

$$
G=QQ^\top\in\mathbb{R}^{D\times D}.
$$

因为 $z=xP$，所以

$$
zGz^\top=zQQ^\top z^\top=\lVert zQ\rVert_2^2=\lVert x\rVert_2^2.
$$

精确私有 RMSNorm 写为

$$
\operatorname{PrivRMS}(z)=
\frac{z}{\sqrt{zGz^\top/d+\varepsilon}}.
$$

若后续明文投影为 $W$，RMSNorm 权重为 $w$，私有投影取

$$
\widetilde W=W\operatorname{Diag}(w)Q^\top,
$$

则

$$
\operatorname{PrivRMS}(xP)\widetilde W^\top
=
\frac{x}{\sqrt{\lVert x\rVert_2^2/d+\varepsilon}}
\operatorname{Diag}(w)W^\top,
$$

与明文路径完全相同。

## 代码位置

- `src/aloepri/models/modeling_aloepri_qwen2.py`：`AloePriMetricRMSNorm`。
- `src/aloepri/conversion/paper_qwen2.py`：计算并写入 $G=QQ^\top$，精确模式下私有 Norm 权重固定为 1。
- `scripts/verify_layerwise_equivalence.py`：逐层把私有残差映射回明文坐标并核对 Norm 输入、输出和后续投影。

## 0.5B 实测

| 版本 | RMSNorm | Prefill 平均绝对误差 | Prefill Top-1 | 16-token greedy |
|---|---|---:|---:|---:|
| v18 | 论文标量 $\kappa$ | 2.631565 | 22.22% | 不一致 |
| v21 | 精确度量 $G$，$\beta=1$ | 0.00001842 | 100% | 一致 |
| v25 | 精确度量 $G$，FP64 离线投影，$\beta=1$ | 0.00001521 | 100% | 一致 |

固定 200 条中英文 prompt 的 v21 与 v25 FP32 结果均为 200/200 greedy 序列逐 token 完全一致。
