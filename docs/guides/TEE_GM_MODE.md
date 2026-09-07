# 隐变智模 TEE 国密模式实施与运行说明

## 1. 两种模式互不替代

旧部署继续使用：

```yaml
security_mode: permutation
boundary_mode: in_model
```

新增模式使用：

```yaml
security_mode: tee_gm
boundary_mode: tee_split
tee_backend: software_sim  # 开发
# tee_backend: intel_tdx   # 生产
```

旧配置没有这些字段时仍按置换模式读取。TEE 工件由原始模型重新生成，不能把旧
TokenKey、tau 或 inverse_tau 拼接到 TEE 部署中。

## 2. 实际数据流

1. 客户端应用 Qwen Chat Template 并在本地分词，得到普通 Token ID。
2. `intel_tdx` 客户端生成 32 字节新 nonce，请求 TD Quote。
3. 客户端用 Intel DCAP QVL 验证 Quote、TCB、MRTD、RTMR、Debug 位和
   REPORTDATA。REPORTDATA 同时绑定 nonce、临时 SM2 公钥、runtime、server
   manifest、model/key/version。
4. 客户端用 Quote 已认证的临时 SM2 公钥建立 RFC 8998 TLS 1.3，固定
   `TLS_SM4_GCM_SM3`，不允许非国密回退。
5. 普通 Token ID 只在该连接密文和 TD 私有内存中出现。
6. TD 查询 `embedding-private.safetensors`：

   ```text
   z0 = (E P)[token_id]
   ```

7. GPU 只接收私有坐标 `z0`，运行原项目已经实现的 P/Q Attention、RoPE、FFN、
   RMSNorm、Residual 与私有 KV Cache，返回最终私有隐藏状态 `zL`。
8. TD 恢复普通最终隐藏状态 `h = zL QL`，再使用完整 TEE Head，或使用经过认证
   的一次性掩码外包 Head。
9. 采样只在 TD 内产生普通 next Token ID。普通 Token 通过同一国密 TLS 返回，
   客户端继续增量解码。

GPU 与宿主机看不到普通 Token、普通 Embedding、`h`、真实 Logits、采样 Token、
边界解密密钥或一次性掩码。

## 3. 工件边界

转换器生成：

```text
tee-boundary-package/             server-body-package/
  embedding-private.safetensors     config.json
  exact-head-archive.safetensors    Transformer权重分片
  final-coordinate-key.safetensors  masked-head-worker.safetensors
  head-coordinate-key.safetensors   generation_config.json
  special-tokens.json               server-manifest.json
  tee-manifest.json                 server-manifest.sm2sig
  tee-manifest.sm2sig
```

软件模拟允许没有签名的开发工件；`intel_tdx` 转换必须提供 Tongsuo 原生 helper、
SM2 签名密钥引用和 SM4 边界密钥引用，否则转换器直接拒绝。TEE 边界文件以
8 MiB 独立块使用 SM4-128-GCM 加密，AAD 绑定路径、块号、model/key/version 和
manifest 摘要。

## 4. 软件模拟命令

现有真实 0.5B 软件模拟工件可按以下方式启动：

```powershell
uv run python -m aloepri.serving.tee_native_entry `
  --model artifacts/tee/qwen05b-tee-sim-v2-body `
  --tee-boundary artifacts/tee/qwen05b-tee-sim-v2-boundary `
  --host 127.0.0.1 --port 18129 --device cuda --head-mode local

uv run yinbian chat-tee `
  --server http://127.0.0.1:18129 `
  --tee-backend software_sim `
  --prompt "中国的首都是哪里？"
```

软件模拟运行同一拆分模型和 API，但证明响应明确为
`hardware_attested=false`，不能用于生产安全结论。

软件模拟的请求与SSE响应不再发送明文JSON：客户端和模拟TEE以临时X25519共享
秘密派生16字节会话密钥，随后使用随机96位Nonce的SM4-128-GCM进行应用层加密。
界面展示的是该次请求实际产生的Nonce、Ciphertext和Tag。X25519只属于开发模拟
握手；生产环境仍固定使用经过TDX Quote绑定的SM2与RFC 8998国密TLS。

## 5. 真实客户端命令

真实客户端不在 Python 内自行实现国密，而是调用 Tongsuo、Tongsuo-linked curl
和 Intel DCAP verifier：

```bash
cat request.json | yinbian-gm-client stream \
  --profile /etc/yinbian/gm-client-profile.json \
  --endpoint /v1/tee/generate/stream
```

客户端固定 TLS 1.3 和 `TLS_SM4_GCM_SM3`，使用 Quote 绑定的临时 SM2 公钥做
public-key pin。Bootstrap 响应本身不可信；只有本机 DCAP verifier 验证后的 Quote
字段才能进入策略判断。

## 6. Head 正确性和失败策略

`LocalTeeHeadEngine` 是正确性基线：`logits = (zL QL) W_exact`。

外包模式令 `x=hB`、`WB=B^-1 W_exact`，TD 发送一次性掩码后的
`u=xq+rho`，GPU 返回 `y=u WBq`，TD 用预计算的 `s=rho WBq` 恢复
`lq=y-s`。每 Token 至少两组秘密 Freivalds 校验。掩码一经预留，无论成功、
超时、取消或崩溃都不能复用。

Greedy 只有在误差区间认证真实 Top-1 后才采样；候选超过 256、NaN/Inf、校验
失败或掩码不足都回退完整 TEE Head。Top-p 首版始终使用完整 TEE Head。

## 7. 当前证据边界

当前仓库已完成并实测：

- Qwen2.5-0.5B TEE split 真权重转换；
- server package 秘密扫描通过；
- 软件模拟 Local TEE Head 的固定样本 greedy Token 完全一致；
- Masked Head 单 Token 真实权重闭环，认证 Token 与完整 Head 一致；
- 独立进程 API、SSE 与 `yinbian chat-tee` 中文问答闭环。

当前普通 RTX 3060/3090 不提供 Intel TDX，因而真实 DCAP Quote、TDX Guest、
RFC 8998 线上互操作和宿主机篡改实验仍必须在 TDX 服务器完成。在这些物理门禁
完成前，发布状态只能是 `TEE_GM_SOFTWARE_SIM_PASS`，不能写成
`TEE_GM_TDX_ATTESTED_PASS`。
