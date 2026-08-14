# AloePri 0.5.0版本说明

## 发布状态

```text
版本：0.5.0
发布状态：CODE_COMPLETE_MOCK_CLOUD_PASS
真实云状态：NOT_TESTED
真实DeepSeek-V3 671B状态：NOT_EXECUTED
```

## 新增

- 模型目录、结构指纹和Qwen2/DeepSeek-V2/DeepSeek-V3族适配器；
- exact required/optional张量覆盖，未知权重拒绝；
- Safetensors range source与预分配range sink；
- 256→32MiB tile降级、CPU回退和资源监控；
- DeepSeek FP8 E4M3FN 128×128反量化/重新量化；
- 主层＋MTP层统一转换，`eh_proj`坐标变换和tiny MTP runtime；
- 官方V3 61层/256专家/1层MTP静态计划与审计脚本；
- SQLite job、tile、事件和deployment状态；
- Mock S3、SSH、Inference Cluster、SSE与回滚；
- S3/SSH生产契约类；
- AES-256-GCM＋Scrypt离线密钥；
- 通用发布包命名和Mock真实性字段；
- 本地Studio六页控制台。

## 当前证据

- Ruff：PASS；
- Mypy strict：PASS；
- Pytest默认套件：256 passed，1 skipped；
- Qwen2.5-0.5B当前checkpoint前向：私有token 6287经逆置换恢复108386，与明文next-token一致；
- OpenSeek-Small-v1-SFT当前checkpoint前向：私有token 123415经逆置换恢复9707，与明文next-token一致；
- Tiny DeepSeek-V3 FP8＋MTP真实Safetensors转换：PASS；
- 官方固定commit索引：91,991张量，MTP 1,564，missing=0，unknown=0；
- 官方静态计划门禁：PASS；
- Mock上传/部署/聊天/回滚：PASS。

## 兼容性

旧命令`serve`、`chat --server`、`verify`、`inspect-package`和`convert --config`保留。新增`convert --plan`默认真实执行；`--schedule-only`只登记任务。

## 限制

- 未真实转换完整671B；
- 未真实启动多节点SGLang；
- Mock token输出不是模型回答；
- 真实0.5B集成测试默认跳过以避免自动占用GPU；当前工件已在CPU单独复测通过；
- 真实云S3、SSH、TLS、IAM、网络和性能仍需物理验收。
