# DeepSeek FP8与MTP实现说明

## FP8

官方格式：E4M3FN、128×128块、`weight`与`weight_scale_inv`配对。

对块`W_b`：

```text
s_b = max(abs(W_b)) / 448
Q_b = clamp(W_b / s_b, -448, 448) cast FP8 E4M3FN
W_b_hat = float32(Q_b) * s_b
```

全零块使用`s_b=1`。非128对齐边缘先零填充，计算scale后裁剪权重，scale网格尺寸为`ceil(rows/128) × ceil(cols/128)`。NaN、Inf、非正scale直接拒绝。

转换顺序：

```text
range读取FP8权重与scale
→ FP32反量化
→ AloePri坐标变换
→ 按输出新shape分块
→ 重新计算每块scale
→ FP8量化
→ weight与新weight_scale_inv写入同一分片
```

不会复制源scale。manifest应记录块大小、scale范围、饱和率和最大量化误差。

## MTP

官方权重位于`model.layers.61`，包括完整MLA/MoE层以及：

```text
embed_tokens.weight
enorm.weight
hnorm.weight
eh_proj.weight
shared_head.norm.weight
shared_head.head.weight
```

设主坐标`x_private=x_plain P`且`PQ=I`。把`eh_proj`写成`[W_e, W_h]`，则私有权重为：

```text
W'_eh = P^T [W_e diag(enorm) Q^T, W_h diag(hnorm) Q^T]
```

因此私有Embedding与私有hidden拼接后，输出等于明文MTP融合输出再乘`P`。`embed_tokens`和主Embedding使用同一`tau`；shared head和主LM Head使用同一输出token空间。MTP decoder层使用与主层相同的MLA、MoE、Residual、FP8流程。

`TinyMTPRuntime`实现候选token生成和主模型argmax验证，用于本地功能测试，不代表真实SGLang MTP加速性能。
