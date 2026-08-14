# 隐变智模 1.0 迁移说明

## 兼容范围

- `aloepri` 命令保留一个版本周期，并提示改用 `yinbian`。
- 内部 Python 包名 `aloepri` 暂不修改。
- 旧 checkpoint 字段、旧 manifest 和 `.aloepri-key` 继续读取。
- 现有模型和密钥无需重新转换。

## 首次启动迁移

新状态库不存在且发现 `%LOCALAPPDATA%\AloePri\state.db` 时，程序执行以下操作：

1. 备份旧数据库；
2. 复制到 `%LOCALAPPDATA%\YinbianZhimo\state\state.db`；
3. 在事务内执行版本迁移；
4. 写入一次性迁移记录；
5. 保留旧数据库，不删除原目录。

迁移失败时，新库停止写入，旧库保持不变。

## 环境变量

| 新变量 | 旧变量 |
|---|---|
| `YINBIAN_STATE_DB` | `ALOEPRI_STATE_DB` |
| `YINBIAN_OFFLINE_KEY_PASSWORD` | `ALOEPRI_OFFLINE_KEY_PASSWORD` |

新变量优先。读取旧变量时记录迁移警告，敏感值不会写入日志。

## 旧密钥导入

```powershell
yinbian keys import-legacy --path <old-key-dir>
yinbian keys backup <key-id> --source <old-key-dir> --output backup.ybkey
```

便携 `.ybkey` 使用 Scrypt 和 AES-256-GCM 分块加密。恢复命令：

```powershell
yinbian keys restore --input backup.ybkey --destination <new-key-dir>
```
