# C02：乱码文本接口必须使用可逆 Token 文本封装

## 修改结论

保留 `/v1/private/generate` 的私有 token ID 数组作为可信主接口，同时增加：

- `POST /v1/private/generate-text`
- `POST /v1/private/generate-text/stream`

新接口发送的是以 `apids1.` 开头的 Base64URL 文本。该文本逐字节封装私有 token ID，不经过 tokenizer。服务端只做格式校验、校验和验证和整数解码，仍不持有原始 tokenizer、`tau` 或 `inverse_tau`。

## 为什么需要修改

项目 PPT 描述了以下路径：

```text
私有 token IDs → tokenizer.decode() 得到乱码 → 发送文本
→ 服务端 tokenizer.encode() → 恢复私有 token IDs
```

一般 tokenizer 不保证 `encode(decode(ids)) == ids`。原因包括：

- 多个 token 序列可能解码成相同 Unicode 文本；
- 字节回退 token、非法 UTF-8 组合和规范化会丢失边界；
- 空白、控制符和特殊 token 的 decode/encode 规则不对称；
- BPE 会重新选择另一种合法分词。

在本仓库的真实 Qwen2.5 tokenizer 上执行固定随机实验：

```powershell
uv run python -c "from transformers import AutoTokenizer; import torch; t=AutoTokenizer.from_pretrained('data/models/qwen2.5-0.5b',local_files_only=True); g=torch.Generator().manual_seed(20260809); ids=torch.randint(0,len(t),(200,),generator=g).tolist(); s=t.decode(ids,skip_special_tokens=False); back=t.encode(s,add_special_tokens=False); print(len(ids),len(back),ids==back)"
```

结果为：输入 200 个 ID，重新编码得到 207 个 ID，序列不相等，首个失配位置为 35。该路径会在输入到达模型前改变 prompt，因此必须修改。

## 可逆格式

逻辑格式为：

```text
apids1. + Base64URL(
    magic="APID" + version=1
    + uint32_be(token_count)
    + token_count × uint32_be(token_id)
    + SHA-256(payload)[0:8]
)
```

约束：

- token 数至少为 1；
- 单个 ID 必须位于无符号 32 位整数范围；
- 解码前检查版本、长度、token 数上限和校验和；
- 非规范、截断、篡改和超限文本返回 400；
- 错误响应不回显私有 token 文本或 ID。

该格式是传输编码，不是密码算法。隐私仍来自客户端词表置换和私有 checkpoint；文本编码的作用是满足“HTTP 字段为文本”同时保证 token 序列完全不变。

## 请求与响应

请求：

```json
{
  "model_id": "qwen05b-product",
  "key_id": "online-key-01",
  "private_text": "apids1....",
  "max_new_tokens": 128,
  "temperature": 0.0,
  "top_k": 0,
  "top_p": 1.0,
  "seed": 20260803
}
```

响应使用同一 `private_text` 格式封装输出 ID。SSE 每个事件封装一个私有输出 ID，客户端验证后执行 `inverse_tau` 并增量解码。

## 代码位置

- `src/aloepri/serving/private_text.py`：版本化编码、解码、长度和校验和验证；
- `src/aloepri/serving/protocol.py`：文本请求和响应模型；
- `src/aloepri/serving/app.py`：普通与 SSE 文本端点；
- `src/aloepri/client/sdk.py`：`transport_mode="encoded_text"` 客户端模式；
- `tests/unit/test_private_text.py`：格式、篡改、HTTP 和 SSE 测试；
- `tests/unit/test_client_sdk.py`：验证 SDK 请求中不存在 JSON `input_ids` 数组并能恢复输出。

## 验证命令

```powershell
uv run pytest -q tests/unit/test_private_text.py tests/unit/test_client_sdk.py
uv run ruff check src/aloepri/serving src/aloepri/client
uv run mypy src/aloepri/serving src/aloepri/client
```

## 未修改的安全边界

- 客户端仍在本地执行 Chat Template、tokenize、M1（可选）和 `tau`；
- 服务端仍只接收私有 token 序列；
- 服务端仍不执行 Qwen tokenizer；
- 服务端响应仍由客户端执行 `inverse_tau` 后才转成正常文本；
- 原 token-ID 主接口继续用于效率最高、歧义最少的部署。
