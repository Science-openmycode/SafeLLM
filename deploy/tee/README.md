# 隐变智模 TEE 国密部署边界

本目录只包含真实 TDX 部署模板，不会把普通 GPU 主机标记成 TDX。

部署前必须执行：

```powershell
yinbian doctor tee --server <server-id>
```

必须同时存在 Intel TDX Guest 设备、DCAP Quote 生成器、客户端 DCAP Quote
验证器、PCCS collateral、Tongsuo 和与其链接且支持
`--tls13-ciphers TLS_SM4_GCM_SM3` 的 curl。任何一项缺失都拒绝
`intel_tdx`，不得自动回退到 `software_sim`。

`gm-client-profile.example.json` 中所有测量值和版本必须由受控发布流程填写并
固定。客户端每次请求生成新 nonce、验证 Intel Quote，并把 Quote 中的
REPORTDATA 与临时 SM2 公钥、runtime、manifest、模型和密钥版本绑定；通过后才
允许国密 TLS 发送 Token。

软件开发闭环使用 `configs/tee/qwen05b_software_sim.yaml`。该配置明确包含
`hardware_attested: false` 和 `production_allowed: false`。

