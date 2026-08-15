# 隐变智模模型与部署审计（2026-08-15）

## 判定口径

模型目录使用三个独立结论，禁止相互替代：

| 结论 | 含义 |
|---|---|
| 可识别 | 固定 revision 的配置和张量布局能归入已知架构族 |
| 可转换 | 当前转换器认领全部需要处理的张量，不存在未知或缺失权重 |
| 可部署 | 该固定 revision 已完成转换、服务加载和真实私有 Token 生成验收 |

静态 Safetensors 头审计不等于完整权重转换，也不等于真实服务器加载。

## 本轮修复

| 问题 | 原因 | 处理 |
|---|---|---|
| Qwen3-4B 缺少 `lm_head.weight` | 官方检查点使用 tied embedding | 在 `tie_word_embeddings=true` 时复用 embedding，不再误报缺失 |
| Kimi/Moonlight 的逐层 `rotary_emb.inv_freq` | 新版 Transformers 可把派生 RoPE buffer 持久化到每层 | 识别为由配置重建的派生张量，不复制为私有权重 |
| OpenSeek 配置声明 MTP、权重却没有 MTP | 上游配置和发布工件不一致 | 固定 revision 先执行来源审计和规范化，只有证明 MTP 张量为零后才把有效 MTP 层数改为零 |
| GLM-4.6 MTP 缺少层内 embedding/head 副本 | 该版本复用全局 embedding 和 LM head | 两种共享布局均接受；全局权重仍只转换一次 |
| Kimi-K2-Thinking 被写成 FP8 | 官方固定版本实际是 packed INT4 专家权重 | 目录改为 mixed INT4/BF16；按 `weight_scale` 和 `weight_shape` 解码后转换 |
| packed INT4 只在 Kimi 多模态外壳生效 | 纯文本 Kimi 规范化后 `model_type` 已变成 DeepSeek-V3 | INT4 逻辑改为根据物理张量名识别，不再依赖外壳模型类型 |
| GLM 0414/Z1 被当作旧 GLM | 新图在 Attention 和 MLP 分支输出后各增加 RMSNorm | 标为 operator gap 并拒绝转换；不能丢掉算子后冒充成功 |
| 部署只检查 `/healthz` | 进程存活不证明模型可生成 | 部署器调用认证的私有生成接口执行一次 Token decode；另增 `/readyz` 供进程内检查；通过后才能切换 active 版本 |
| 同族模型全部显示可部署 | 架构兼容被错误当作检查点验收 | 只有 Qwen2.5-0.5B 固定 revision 保留部署资格，其余最多到转换阶段 |
| 缓存挤占系统盘 | 状态、密钥和可再生成缓存共用默认根目录 | 新增 `YINBIAN_CACHE_DIR` 独立覆盖，不移动状态库和密钥 |

## 固定版本审计

审计命令只读取 Hugging Face 固定 commit 的 `config.json` 与 Safetensors 元数据：

```powershell
$env:HF_HOME = "E:\YinbianCache\hf"
python scripts\audit_catalog_sources.py `
  --workers 2 `
  --output artifacts\audits\catalog-full-source-audit-20260815.json
```

“PASS”包含两种情况：转换型号的张量覆盖完整；禁止转换型号被正确拦截。GLM
0414/Z1 的正确结果是后者，不表示转换成功。

本轮结果：`54/54 PASS`。原始机器可读记录位于
`artifacts/audits/catalog-full-source-audit-20260815.json`。

## 自动化与真实模型

| 检查 | 结果 |
|---|---:|
| Ruff | PASS |
| Mypy | PASS（129个源文件） |
| 默认 Pytest | 336 passed，1 skipped |
| Qwen2.5-0.5B GPU 私有 API、SSE、错误密钥和 `/readyz` | 1 passed |

默认跳过项就是需要显式启用的真实模型测试；上一行给出同一测试在当前 RTX 3060
上的独立执行结果。

## 部署范围

| 模型 | 转换 | 一键 HF 部署 | 说明 |
|---|---:|---:|---|
| Qwen2.5-0.5B-Instruct 固定 revision | 是 | 是 | 当前产品验收基线 |
| 其他 Qwen2.5 / Qwen3 | 是 | 否 | 需要逐检查点完整 smoke 后才能开放 |
| OpenSeek-Small-v1-SFT | 是 | 否 | 已有转换证据，尚未形成当前产品部署验收记录 |
| DeepSeek / GLM4-MoE / Kimi / Moonlight | 按目录标记 | 否 | 超大模型主要运行时为 SGLang；当前一键部署器只有 HF 单主机实现 |
| GLM 0414/Z1 | 否 | 否 | 分支后置 RMSNorm 算子缺口 |

## 本机空间处置

清理范围仅包括 pip/uv 缓存、崩溃转储、GPU shader 缓存和两天前的临时文件。
未删除模型、转换工件、密钥、SQLite 状态库或聊天记录。用户级
`YINBIAN_CACHE_DIR` 已设置为 `E:\YinbianCache`，下次启动桌面程序生效。
