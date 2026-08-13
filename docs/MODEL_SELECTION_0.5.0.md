# 模型选择手册

## 内置目录

| catalog_id | 固定revision | 适配器 | 本地执行建议 | 运行时 |
|---|---|---|---|---|
| `qwen2.5-0.5b-instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | `qwen2` | 3060可完整转换与HF推理 | HF |
| `deepseek-v2-lite-chat` | `85864749cd611b4353ce1decdb286193298f64c7` | `deepseek_v2` | 本地流式转换；推理需更大显存或量化 | HF/SGLang |
| `deepseek-v3` | `bb399fea3bbfbea55d71cb018e12cdfb6b215179` | `deepseek_v3` | 3060按range/tile慢速转换；不适合本地推理 | SGLang |

## 识别规则

识别只使用：

1. `config.json`中的计算图字段；
2. Safetensors张量名；
3. 张量形状和dtype；
4. FP8 scale配对、专家数量和MTP层数。

仓库名中含“DeepSeek”不代表DeepSeek架构。DeepSeek蒸馏Qwen若`model_type=qwen2`，必须进入Qwen适配器。

## 状态含义

- `SUPPORTED`：结构、张量清单和必要能力完整，可以生成正式plan；
- `EXPERIMENTAL`：只允许开发适配器，不允许正式转换；
- `INCOMPATIBLE`：计算图不匹配；
- `INCOMPLETE_CHECKPOINT`：配置声明FP8/MTP等能力，但文件缺失；
- `UNKNOWN_TENSOR_LAYOUT`：存在适配器未认领的权重。

正式转换只接受`SUPPORTED`。未知权重不会静默复制。

## 检查命令

```powershell
uv run aloepri models list
uv run aloepri models recommend
uv run aloepri models inspect --model <本地模型目录>
uv run aloepri doctor --model <本地模型目录>
```

Hugging Face来源必须固定完整commit。下载后保留revision、文件字节数和SHA-256，不使用浮动`main`作为生产输入。
