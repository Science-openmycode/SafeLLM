# AloePri 论文逐项勘误

本文档集针对 arXiv:2603.01499v2。每个问题单独给出原式、维度或代数推导、
最小修正和仓库实现选择。PDF 位于 `output/pdf/paper_errata/`。

| 编号 | 问题 | 判定 |
|---|---|---|
| E01 | Algorithm 1 的 $C$ 把“行”写成“列” | 确定错误 |
| E02 | Algorithm 1 的 $D$ 把“列”写成“行” | 确定错误 |
| E03 | BlockPerm 声明 $\gamma$ 但采样式未使用 | 确定错误，修正方向有两种 |
| E04 | BlockPerm 循环没有更新 $t$ | 确定错误 |
| E05 | 自然补上 $t\leftarrow t+w$ 后仍遗漏最后一个 block | 在该自然修复下确定 |
| E06 | Q/K 的 $Z$ 与 $Z^\mathsf{T}$ 一般产生 $Z^2$ | 确定错误 |
| E07 | RMSNorm 的 $\kappa$ 缺少 $\sqrt{d/(d+2h)}$ | 确定错误 |
| E08 | Algorithm 2 是否共享同一次 `Init` | 实现歧义，不列为确定错误 |
| E09 | Attn-IA 的 inverse-Gram 乘法维度不成立 | 确定错误 |
| E10 | BlockPerm 的频率指数与 Qwen 实际 RoPE 相差两倍 | 确定错误 |
| E11 | Summation Composition 使用未定义的 $\psi_X|_{C_2}$ | 确定符号错误 |
| E12 | 在线 detokenize/tokenize 不保证保持任意混淆 Token 序列 | 确定接口假设错误 |
| E13 | $s_i\in\mathbb R$ 未排除零，但后续使用 $H^{-1}$ | 确定定义域缺项 |
| E14 | 整模误差上界连乘使用 $j$，被乘项却写 $M_i$ | 确定下标错误 |
| E15 | 一般矩形 $P$ 下标量 $\kappa$ 不能使 RMSNorm 逐输入协变 | 确定公式错误；使用 $G=QQ^\top$ 修正 |
| E16 | 不同 RoPE 频率块之间的置换不与 RoPE 对易 | 确定函数保持条件缺失 |
| E17 | 同一词表置换保持序列的 token 相等模式，Definition 2 在全空间不总是有限 | 确定定义域/值域缺项 |
