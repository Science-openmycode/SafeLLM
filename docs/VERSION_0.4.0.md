# AloePri 0.4.0：OpenSeek paper-complete 隐私转换

## 新增

- 新增 `aloepri_deepseek_v3` 自定义模型，支持扩维后的 DeepSeek-V3 私有 checkpoint。
- 新增 OpenSeek 全量流式转换：词表、Embedding/Head 噪声、Algorithm 1 P/Q、RMSNorm/Residual、MLA、Q/K 互逆缩放、RoPE BlockPerm、V/O 高斯映射、Dense/Shared/Routed FFN、Router 和专家置换。
- 新增在线/离线密钥拆分和净化 server package。
- 新增严格隐私完整性审计、真实问答、产品边界抓取、64 候选攻击冒烟和证据 SHA-256 索引。
- IMA 的论文攻击 backbone 固定为独立两层 Qwen2，不再错误继承 DeepSeek/MoE 目标配置。
- ISA 脚本支持注册 `aloepri_deepseek_v3` checkpoint。

## 固定工件

| 工件 | 路径 |
|---|---|
| 配置 | `configs/product/openseek_small_v1_sft_paper_complete.yaml` |
| 私有模型 | `data/packages/openseek-small-v1-sft-paper-complete` |
| 在线密钥 | `data/keys/openseek-small-v1-sft-paper-complete-online` |
| 离线主密钥 | `data/keys/openseek-small-v1-sft-paper-complete-offline` |
| 审计 | `artifacts/openseek-small-v1-sft-paper-complete/privacy-completeness-audit.json` |
| 边界测试 | `artifacts/openseek-small-v1-sft-paper-complete/product-privacy-boundary.json` |
| 证据索引 | `artifacts/openseek-small-v1-sft-paper-complete/evidence-index.json` |

## 版本结论

该版本对 OpenSeek 发布 checkpoint 中实际存在的模块完成隐私机制实现和自动审计。上游没有发布 MTP 张量，因此 MTP 不纳入该 checkpoint 的功能缺失。完整多密钥、全词表攻击和精度/性能验收仍是发布门禁，不由本版本的 64 候选冒烟替代。
