---
title: "AloePri 论文勘误 E06：Q/K 两侧的 Zblock 与 Zblock 转置不能一般相消"
date: "2026-08-06"
---

# 结论

这是一个确定的代数错误。Algorithm 2 对 Query 和 Key 右侧分别使用
$Z_{\mathrm{block}}$ 与 $Z_{\mathrm{block}}^\mathsf{T}$。在行向量推理约定下，
Key 进入注意力点积时还会整体转置，因此两者产生 $Z_{\mathrm{block}}^2$，而不是
单位阵。只有置换恰好是对合时才偶然正确。

# 推导

忽略已经在输入侧相消的 $P,Q$，令原始行向量为

$$
q=xW_q,\qquad k=xW_k.
$$

论文给出的右侧变换为

$$
T_q=RHZ,\qquad T_k=RH^{-1}Z^\mathsf{T},
$$

其中 $R$ 为分块旋转，$H$ 为成对缩放，$Z$ 为 block 置换。混淆后的点积是

$$
\widetilde q\widetilde k^\mathsf{T}
=qT_q(kT_k)^\mathsf{T}
=q\,T_qT_k^\mathsf{T}k^\mathsf{T}.
$$

因 $H^\mathsf{T}=H$、$R^\mathsf{T}=R^{-1}$、$(Z^\mathsf{T})^\mathsf{T}=Z$，

$$
T_qT_k^\mathsf{T}
=RHZ\,ZH^{-1}R^\mathsf{T}
=RHZ^2H^{-1}R^\mathsf{T}.
$$

一般置换并不满足 $Z^2=I$。例如三循环

$$
Z=\begin{bmatrix}0&1&0\\0&0&1\\1&0&0\end{bmatrix},
\qquad
Z^2=\begin{bmatrix}0&0&1\\1&0&0\\0&1&0\end{bmatrix}\ne I.
$$

所以注意力分数不能保持。

# 最小修正

将 Key 公式末尾改为同一个 $Z$：

$$
T_k=RH^{-1}Z.
$$

则

$$
T_qT_k^\mathsf{T}
=RHZ\,Z^\mathsf{T}H^{-1}R^\mathsf{T}=I.
$$

仓库 `src/aloepri/transforms/qwen_structural.py` 对 Q/K 使用同向 block map，并以
prefill、decode、KV Cache 和 logits 的数值等价测试验证该选择。

# 来源定位

论文版本：arXiv:2603.01499v2，Algorithm 2 第 6--7 行；本地源文件
`data/papers/arxiv-2603.01499v2-source/chap/4.method.tex`。
