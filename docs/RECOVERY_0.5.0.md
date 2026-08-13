# 故障恢复手册

## 查询

```powershell
uv run aloepri jobs list
uv run aloepri jobs status <job-id>
```

记录包含当前阶段、tensor、tile、错误、更新时间和plan。不要直接编辑SQLite。

## 转换中断

1. 保留`.partial`、SQLite和已验证分片；
2. 检查磁盘、显存和错误字段；
3. 使用同一plan执行`jobs resume`；
4. Resume重新校验SHA-256；
5. plan参数、source revision或文件hash变化时新建job，不复用旧断点。

## OOM

执行器自动把tile减半到32MiB，再回退CPU。若仍失败：

```text
停止其他GPU进程
确认至少1.2GiB空闲显存
降低plan.resources.tile_mib
确认host_memory_budget_gib不超过机器实际可用内存
重新resume
```

## 上传失败

Mock和S3 multipart以part编号、ETag和SHA-256记录。Resume只补传缺失或hash不一致的part，不重复上传已验证part。

## 部署失败

1. 保留旧RUNNING deployment；
2. 新deployment健康检查失败时标记UNHEALTHY；
3. 执行`deployment rollback <id>`；
4. 验证旧model/key ID恢复；
5. Mock回滚证据不得替代真实集群回滚测试。

## 不可恢复条件

- 源revision或源分片hash改变；
- `.partial`header与plan输出shape不一致；
- 已验证分片损坏；
- FP8权重缺scale；
- MTP配置有层但权重缺失；
- 离线密钥密码丢失。

这些情况应新建job或从备份恢复，不应绕过校验。
