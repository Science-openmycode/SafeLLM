---
title: "AloePri 论文勘误 E12：在线文本接口假设任意 Token 序列可无损反分词再分词"
date: "2026-08-08"
---

# 结论

论文第 10 页的在线流程要求客户端先得到置换后的 Token ID 序列，再
detokenize 为文本；服务端收到文本后重新 tokenize。该流程隐含要求

$$
\operatorname{Encode}(\operatorname{Decode}(z))=z
$$

对任意置换后序列 $z=\tau(x)$ 成立。Qwen2.5 tokenizer 不满足这个条件。
因此论文文字接口不能保证服务端收到的 Token ID 仍为 $\tau(x)$。

# 为什么词表置换会触发问题

自然文本产生的 Token 序列通常位于 tokenizer 的规范编码集合中；随机词表置换
$\tau$ 会把它变成任意词表 ID 的组合。对 BPE、byte fallback、特殊 Token 和
Unicode 规范化路径，Decode 不是 Encode 的双射逆：

$$
\operatorname{Decode}:\mathbb Z_n^l\rightarrow\text{Text}
$$

可以把不同 ID 序列映射为同一字符串，也可能让一个 ID 解码出的字符串在重新
编码时被拆成多个 ID。因此一般只有

$$
\operatorname{Decode}(\operatorname{Encode}(t))\approx t,
$$

不能推出

$$
\operatorname{Encode}(\operatorname{Decode}(z))=z.
$$

# 0.5B tokenizer 的可复现反例

使用本地 Qwen2.5-0.5B tokenizer，单个非特殊 Token ID `140688` 的反分词文本
该字符串由 Unicode 码点
`U+0E41 U+0E21 U+0E49 U+0E27 U+0E48 U+0E32` 组成，再次编码得到三个 Token：

$$
[140688]\xrightarrow{Decode}\text{上述六个 Unicode 码点}
\xrightarrow{Encode}[124840,124379,64741].
$$

长度从 1 变为 3，服务端输入已经不是客户端生成的混淆 ID。在每种长度抽样
200 条非特殊 Token 序列时，长度 1、2、4、8、16 分别有
2、11、28、61、106 条失败；20 条真实 prompt 置换后有 6 条失败。

# 工程处理

本仓库正式接口直接发送 `input_ids`：

$$
x\xrightarrow{\tau}\tau(x)\xrightarrow{\text{HTTP Token IDs}}\text{server}.
$$

这保持了论文所需的 Token 坐标，但不是论文描述的文本传输接口。只有某个具体
tokenizer 和允许的 ID 子集通过完整 round-trip 证明后，才能开放文本接口。

# 来源定位

论文版本：arXiv:2603.01499v2，第 10 页 Section 5.3；LaTeX 源文件
`chap/4.method.tex` 第 185 行。复现实验将由
`scripts/verify_tokenizer_roundtrip.py` 生成机器可读 JSON。
