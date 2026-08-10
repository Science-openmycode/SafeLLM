---
title: "AloePri 论文勘误 E07：RMSNorm 的 kappa 少了扩维维度因子且方向不匹配"
date: "2026-08-06"
---

# 结论

论文定义

$$
\kappa_{\mathrm{paper}}
=\mathbb E\left[\frac{\|xP\|_2}{\|x\|_2}\right],
$$

随后把扩维 RMSNorm 权重设为 $\kappa\mathbf 1$。但 RMSNorm 使用均方根而不是
二范数；从 $d$ 维扩到 $D=d+2h$ 维后，正确的缩放比必须包含
$\sqrt{d/D}$。因此原式与论文后续希望成立的 RMSNorm 近似式不一致。

# 推导

忽略数值稳定项 $\epsilon$，定义

$$
\operatorname{RMS}_d(x)=\frac{\|x\|_2}{\sqrt d},
\qquad
\operatorname{RMS}_D(xP)=\frac{\|xP\|_2}{\sqrt D}.
$$

论文希望

$$
\frac{xP}{\operatorname{RMS}_D(xP)}\kappa
\approx
\frac{xP}{\operatorname{RMS}_d(x)}.
$$

消去共同的 $xP$ 后，应取

$$
\kappa_{\mathrm{RMS}}
\approx\frac{\operatorname{RMS}_D(xP)}{\operatorname{RMS}_d(x)}
=\sqrt{\frac dD}\frac{\|xP\|_2}{\|x\|_2}.
$$

故

$$
\boxed{\kappa_{\mathrm{RMS}}
=\sqrt{\frac d{d+2h}}\,\kappa_{\mathrm{paper}}}.
$$

当 $d=896,h=128$ 时，$D=1152$，维度因子为

$$
\sqrt{896/1152}\approx0.881917.
$$

这不是半精度舍入项，而是由 RMS 定义产生的确定因子。

# 可复核特例

令 $P=[I_d\;0]$，则 $\|xP\|_2=\|x\|_2$。论文原式给
$\kappa_{\mathrm{paper}}=1$；实际需要

$$
\kappa_{\mathrm{RMS}}=\sqrt{d/D}.
$$

代回可得左右完全相等。使用 $\kappa=1$ 则整体放大 $\sqrt{D/d}$。

# 实现

仓库同时保留 `paper-norm-ratio-proxy` 与 `covariant-rms`。前者使用
$\|P\|_F/\sqrt d$ 作为二阶矩代理；一般矩阵下它不等于论文写出的精确期望
$\mathbb E[\|xP\|_2/\|x\|_2]$。后者用于可执行工程分支。验收记录必须标明
选择，不能把代理值、RMS 修正值和论文精确期望混为一项结果。

# 来源定位

论文版本：arXiv:2603.01499v2，Section 5.2.5 与 Section 5.4 的 Layer
Normalization 推导；本地源文件
`data/papers/arxiv-2603.01499v2-source/chap/4.method.tex`。
