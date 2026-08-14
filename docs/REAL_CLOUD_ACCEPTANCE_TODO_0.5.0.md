# 真实云端待验收清单

0.5.0未执行以下物理验收，状态均为`NOT_TESTED`或`NOT_EXECUTED`。

## 对象存储

- 真实S3兼容endpoint、IAM最小权限、multipart、断点和生命周期；
- 689GB级上传费用、出入站带宽和失败重传；
- 分片上传后本地删除策略；
- 服务端下载后的manifest与SHA-256复核。

## 远程主机

- SSH host key固定与密钥认证；
- 非root服务账户；
- 在线密钥路径不出现在命令、环境、镜像或日志；
- systemd/container重启和磁盘满恢复；
- TLS反向代理与Bearer Token轮换。

## DeepSeek-V3集群

- 真实DeepSeek-V3 671B权重转换完成；
- 多节点SGLang加载；
- TP/PP/EP拓扑；
- InfiniBand/RDMA；
- 私有EOS、token-ID输入、SSE输出；
- MTP开启/关闭一致性和加速；
- 节点故障、健康检查和蓝绿回滚；
- TTFT、TPOT、吞吐、显存、网络和加载时间。

## 精度与安全

- 明文与私有模型同硬件、同请求分布比较；
- 完整任务精度与95%置信区间；
- 多key攻击实验；
- 抓包、日志、异常堆栈和对象存储秘密扫描；
- 真实恶意或半诚实服务器威胁模型说明。

只有以上证据绑定当前checkpoint、key、代码commit和配置hash后，才能把`real_cloud_validated`改为`true`。
