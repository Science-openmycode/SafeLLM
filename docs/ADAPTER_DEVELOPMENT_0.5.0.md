# 架构适配器开发手册

## 目录

```text
src/aloepri/catalog/          模型目录、头部扫描、架构指纹
src/aloepri/adapters/         族适配器、覆盖报告、V3静态计划
src/aloepri/tensor_io/        Safetensors range source/sink
src/aloepri/conversion/       现有转换器、tile与统一执行器
src/aloepri/formats/          FP8 codec
src/aloepri/transforms/       Qwen、MLA、MoE、MTP数学变换
```

## 新架构接入步骤

1. 从`config.json`定义计算图指纹，禁止使用repo名称判断；
2. 枚举所有required和optional张量；
3. 对每个投影写明输入坐标、输出坐标、形状和逆变换；
4. 为Embedding、Head、Norm、Residual、Cache和特殊token写端到端规则；
5. 若有量化，先反量化到FP32、变换、重新量化并生成新scale；
6. 未认领张量必须进入`unknown`，不得`copied_unknown`；
7. 配置声明能力但缺权重时返回`INCOMPLETE_CHECKPOINT`；
8. 增加tiny forward等价、tile一致性、resume一致性和包扫描测试；
9. 固定上游完整commit并保存官方索引审计；
10. 只有coverage missing=0且unknown=0才允许`SUPPORTED`。

## DeepSeek-V3官方索引审计

```powershell
uv run python scripts\audit_official_deepseek_index.py `
  --config <官方config.json> `
  --index <官方model.safetensors.index.json> `
  --revision bb399fea3bbfbea55d71cb018e12cdfb6b215179 `
  --out artifacts\audits\deepseek-v3-official-index.json
```

当前结果：91,991张量名、MTP层1,564个条目、missing=0、unknown=0。
