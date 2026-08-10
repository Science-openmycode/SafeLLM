# 0.5B 独立子 Agent 审核与处置记录

审核范围：Qwen2.5-0.5B、论文 arXiv:2603.01499v2、核心转换代码、v14 工件、
E01--E09 文档。子 Agent 以只读方式审核；主 Agent 根据证据逐项处置。

| 级别 | 审核发现 | 客观复核 | 处置 |
|---|---|---|---|
| P0 | E08 被过度判为确定错误 | 成立。共享同一次 Algorithm 1 `Init` 时，任意合法 $P(C),Q(D)$ 可相消 | E08 降级为实现歧义，补共享 `Init` 推导并重编 PDF |
| P1 | `paper-literal` 名称不准确 | 成立。实现已修复原文的不终止循环和边界 | 改名 `paper-distribution-boundary-corrected`；与 `gamma-corrected` 分开 |
| P1 | 论文 κ 函数不是精确期望 | 成立。$\|P\|_F/\sqrt d$ 是二阶矩代理 | 改名 `frobenius_norm_ratio_proxy` 和 `paper-norm-ratio-proxy` |
| P1 | streaming 续传未绑定源权重，partial 只查存在 | 成立 | 指纹加入源 config/权重 SHA-256；partial 保存并复核大小与 SHA-256；补篡改单测 |
| P1 | `engineering_validation: GO` 范围过宽 | 成立 | 改为“已测证据 GO / 完整工程 PARTIAL-GO”，列出未验证项目；API 与 checkpoint 门禁加严 |
| P1 | 论文参数门禁不完整 | 成立 | 加入 Algorithm 2、BlockPerm 模式、Uvo、κ 模式、Q/K 非退化采样检查 |
| P1 | E01 有 `c_i^mathsf{T}` 排版错误 | 成立 | 修成 `c_i^{\mathsf T}`，重编并渲染检查 |
| P2 | E05 的确定性依赖先补 $t\leftarrow t+w$ | 成立 | 明确写成“采用自然循环修复时可严格推出” |
| P2 | Gate/Attn 攻击缺数学自动测试 | 部分成立。Attn 已有形状测试但只是 proxy | 新增 Gate-IA 协变恒等测试；Attn 继续明确标为非论文精确攻击，列入未完成 |
| P2 | Token-ID API 不是论文文字流程 | 成立 | README 明确列为协议层工程修正 |
| P1 | v14 API/验证文本为乱码 | 不成立。以 UTF-8 读取工件，恢复文本为正常中文 | 未修改结果；验收新增常见乱码标记检查，当前 `mojibake_detected=false` |

最终复核命令：

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m pytest -q

$env:ALOEPRI_RUN_MODEL_TESTS='1'
$env:ALOEPRI_PRIVATE_MODEL='data/checkpoints/qwen2.5-0.5b-paper-v14-engineering-secure-fp32'
$env:ALOEPRI_KEY_DIR='data/keys/dev-qwen05b-paper-v14-engineering-secure-fp32'
.\.venv\Scripts\python.exe -m pytest -q tests/integration/test_real_private_api.py
```

结果：Ruff 通过；mypy 通过；常规测试 72 passed、1 skipped；真实 v14 API
集成测试单独启用后 1 passed。全部 E01--E09 PDF 已重新编译并逐页渲染检查。
