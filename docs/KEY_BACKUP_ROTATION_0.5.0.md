# 密钥备份与换钥手册

## 三类工件

| 工件 | 位置 | 内容 | 是否上云 |
|---|---|---|---|
| 在线密钥 | 本地Agent | `tau`、`inverse_tau`、特殊token映射、model/key ID | 否 |
| 离线主密钥 | 离线归档 | P/Q、Attention、FFN、噪声和重建材料 | 否 |
| Server package | 云端 | 私有权重、无秘密配置、manifest | 是 |

产品plan默认`offline_key_encrypted=true`。执行前：

```powershell
$env:ALOEPRI_OFFLINE_KEY_PASSWORD = "<强密码>"
```

密码不写入YAML、SQLite、命令行或日志。离线目录最终只保留`offline_master_key.aloepri-key`，算法为AES-256-GCM，密钥由Scrypt派生。

## 备份

1. 分别备份加密离线密钥、在线密钥和server manifest；
2. 密码保存在独立密码管理器；
3. 记录model ID、key ID、revision和SHA-256；
4. 每次恢复先验证manifest，再尝试解密到临时受控目录；
5. 不把解密后的离线密钥长期留在工作目录。

## 换钥

- 仅更换`tau`：重新生成Embedding、LM Head、MTP shared embedding/head和特殊token配置；
- 更换P/Q或结构密钥：重新转换全部模型；
- 新旧模型使用不同deployment ID进行蓝绿切换；
- 请求的model/key不匹配时服务端拒绝；
- 新deployment健康后再停止旧版本；回滚只切换到已验证旧ID。
