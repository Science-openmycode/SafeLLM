# 不使用词表置换的机密 LLM 推理：检索记录

检索日期：2026-08-28

## 主要来源

1. Huang et al., **Your Inference Request Will Become a Black Box: Confidential Inference for Cloud-based Large Language Models**, ACL 2026. 论文提出 Talaria 和 Reversible Masked Outsourcing（ReMO）：可信 CVM 保存明文和结构运算，把加掩码的线性运算外包给普通 GPU，再在 CVM 中精确恢复。论文明确覆盖 prompt、response、KV cache 和 sampling，并报告固定采样条件下输出保持一致。
   - https://aclanthology.org/2026.acl-long.4/
   - https://aclanthology.org/2026.acl-long.4.pdf

2. Tramèr and Boneh, **Slalom: Fast, Verifiable and Private Execution of Neural Networks in Trusted Hardware**, ICLR 2019. 该工作使用 TEE、加法掩码和完整性校验，将线性层外包给不可信 GPU，是 ReMO 类设计的早期基础。
   - https://arxiv.org/abs/1806.03287

3. Chrapek et al., **Confidential LLM Inference: Performance and Cost Across CPU and GPU TEEs**, 2025. 该工作评估完整 LLM 在 CPU/GPU TEE 中的运行，报告 H100 机密 GPU 的吞吐损耗约 4%–8%，并指出 TEE 是当前兼顾通用性和效率的务实路线。
   - https://arxiv.org/abs/2509.18886

4. NVIDIA, **Confidential Computing Features of NVIDIA Hopper and Blackwell**。官方白皮书说明机密 GPU、CVM、远程证明、PCIe/内存保护以及单卡和多卡模式。
   - https://docs.nvidia.com/nvidia-secure-ai-with-blackwell-and-hopper-gpus-whitepaper.pdf

5. NVIDIA, **Attestation SDK / GPU Attestation**。官方文档给出 H100 或更新 GPU 的本地和远程证明要求及流程。
   - https://docs.nvidia.com/attestation/attestation-client-tools-sdk/latest/gpu_and_switch_attestation.html

6. Folkerts and Tsoutsos, **Tyche: Probabilistic Selection over Encrypted Data for Generative Language Models**, 2024。研究 FHE 条件下生成式模型的加密采样/argmax，说明只保护线性层还不足以完成自回归生成。
   - https://eprint.iacr.org/2024/1087

7. Li et al., **MPCFormer: Fast, Performant and Private Transformer Inference with MPC**, ICLR 2023。论文展示 MPC Transformer 的主要瓶颈来自非线性和通信；其 BERT-base 示例总运行约 59 秒，而明文不足 1 秒。
   - https://arxiv.org/abs/2211.01452

## 对当前问题最关键的事实

- 不使用词表置换时，原始输出 token 的编号必须由客户端、TEE/机密 GPU，或密码学安全计算的一方得到；若普通服务器直接执行原词表 argmax，输出隐私必然丢失。
- LM Head 是线性层，理论上可通过 Slalom/ReMO 型掩码外包；sampling/argmax 和下一 token 状态必须留在可信域。
- Talaria 的部署掩码 `M = M_pvt M_pub` 使用 `m < d`，附录明确承认存在 `d-m` 维残余子空间，因此它是强实用方案，但不是所有方向上的完美隐藏。
- 完整机密 GPU 可以让 Embedding、Transformer、LM Head、KV cache、sampling 和自循环都位于经证明的保护域内，是当前最直接的完整工程方案。
- 若拒绝硬件信任，只能使用 HE/MPC；当前性能与通信成本仍不适合 671B 商业推理。
# 2026-08-28 LM Head 安全拆分补充

用户明确排除把完整模型放入可信GPU/TEE。本轮将问题缩小到：不使用词表
置换时，只对Embedding、LM Head和采样进行可信拆分或安全外包。

新增资料：

- Slalom: https://arxiv.org/abs/1806.03287
- Talaria/ReMO: https://aclanthology.org/2026.acl-long.4/
- Hyperion: https://aclanthology.org/2026.acl-long.644/
- Cutmax: https://aclanthology.org/2025.findings-ijcnlp.111/
- BLB: https://www.usenix.org/conference/usenixsecurity25/presentation/xu-tianshi
- Full FHE Llama 2/3: https://arxiv.org/abs/2601.18511

结论变化：

- 完整机密GPU不再作为主推荐。
- 主推荐改为可信GPU网关头尾拆分、云端P/Q Transformer主体、每Token
  零空间随机掩码。
- 若可信网关只有CPU，研发head-only HE借用云GPU。因为LM Head是单层线性
  运算，它比完整FHE简单，但V约13万时的旋转、密文输出和带宽必须由原型确认。
- 纯矩阵行列拆分不能隐藏最终argmax；普通服务器看见logits或Embedding查表
  索引就会获知输出Token。

# 2026-08-29 LM Head 主乘法安全外包

本轮进一步限定目标：LM Head 的大矩阵乘法必须在不可信普通 GPU 上执行，
TEE 只做轻量工作，并且普通 GPU 不得看到真实隐藏向量、真实 logits 或
输出 Token。

Talaria/ReMO 给出可直接用于矩形 LM Head 的掩码外包公式。设 h 的形状为
1×d，W 为 d×V：

    h_masked = h + M
    gpu_result = h_masked W = hW + MW
    logits = gpu_result - MW

Talaria 使用 M = M_pvt M_pub，并预计算 R_pub = M_pub W，因此 TEE 每 Token
恢复为 gpu_result - M_pvt R_pub。论文说明结果是代数无损的，但 m<d 时存在
d-m 维未完全掩码的补空间；论文没有宣称所有方向的完美隐藏。其威胁模型是
honest-but-curious cloud。

对 DeepSeek 量级 d=7168、V=129280：

- 完整 LM Head 约 1.853 GFLOP/Token；
- m=0.5d 时，TEE 恢复仍约 0.927 GFLOP/Token，R_pub BF16约0.927GB；
- m=0.75d 时，TEE 恢复约 1.390 GFLOP/Token，R_pub约1.390GB；
- m=0.9d 时，TEE 恢复约 1.668 GFLOP/Token，R_pub约1.668GB。

因此 ReMO 对普通 Transformer 线性层很实用，但直接用于巨大 V 的 LM Head
时，若 m 足够大以增强隐藏，TEE 恢复乘法仍接近原 LM Head，不能彻底解决
CPU TEE 算力问题。

强隐藏版本应使用每 Token 一次性全维随机 r，并让 TEE 预持有 s=rW：

    send h+r
    GPU computes (h+r)W
    TEE subtracts s

在线 TEE 仅做 O(d+V) 加减。s 可由可信离线预处理、两个不串通服务，或
head-only HE 异步生成。若使用同一不可信 GPU，必须以 HE 形式输入 Enc(r)，
否则 GPU 知道 r 后可从 h+r 恢复 h。

代价是掩码池。BF16 下每 Token 的 r 与 rW 约273KB；30,000 Token/s 消耗
约8.2GB/s，运行1小时约29.5TB。它适合短时低延迟或研究原型，不适合无限
高并发流量。

相关一手资料：

- Talaria/ReMO公式与局限：https://aclanthology.org/2026.acl-long.4.pdf
- Slalom掩码外包：https://arxiv.org/abs/1806.03287
- HE矩阵外包：https://eprint.iacr.org/2018/1041
- BLB CKKS/MPC：https://www.usenix.org/conference/usenixsecurity25/presentation/xu-tianshi
