# Changelog

## 0.2.0 - 2026-08-11

### Added

- 将 Qwen2.5-0.5B 的正式产品目标固定为 v47，并增加单一验收配置。
- 增加目标密钥隔离的 Gate-IA、Attention-IA、IMA、ISA、TFMA、SDA 攻击与独立评分器。
- 增加真实频率攻击语料下载器，数据集 revision、切分规则和文件 SHA-256 写入 manifest。
- 增加私有 token 观测工件，工件只保存服务端可见的混淆 token。
- 增加 v47 验收报告生成器；证据缺失、字段缺失、hash 失配均返回 `NO-GO`。

### Changed

- Gate-IA 使用代数等价的结合律计算，避免构造词表大小乘 FFN 中间维度的临时矩阵。
- Attention-IA 使用维度成立的 leverage 特征和 Qwen 实际 RoPE 配对。
- IMA 训练数据只允许来自非目标转换密钥，训练、目标攻击和目标密钥评分分进程执行。
- TFMA 不再在候选搜索函数中读取 `inverse_tau`。
- SDA 不再把训练、目标攻击和评分混在一个进程，也不允许真值逆置换回退。

### Compatibility

- 历史 v15-v38 配置、脚本和工件保留用于复核；v47 验收器不会读取其结果。
- `run_gate_ia.py`、`run_tfma_curve.py`、`train_sda_smoke.py` 保留为历史实验入口；正式入口
  分别为 `run_gate_ia_isolated.py`、`run_tfma_isolated.py` 和
  `train_sda_transformer.py`/`run_sda_target_attack.py`。
