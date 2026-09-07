# LM Head 掩码外包详细设计 v1

日期：2026-08-29

## 1. 目标

可信边界只有CPU TEE。外部GPU和宿主机均不可信。

必须满足：

- LM Head的d×V主乘法由外部GPU完成；
- CPU TEE在线不执行d×V矩阵乘法；
- 外部GPU看不到真实h、真实logits和输出Token；
- 每个掩码只能使用一次；
- 外部GPU返回错误结果时可以检测；
- 失败、超时或进程崩溃后不得重用不确定状态的掩码；
- 明确区分在线加速和离线掩码生成成本。

不保证：

- 阻止拒绝服务；
- 修复P/Q模型主体本身可能存在的隐藏状态攻击；
- 在浮点实现中自动获得信息论安全和BF16逐位一致。

## 2. 参与方

### CPU TEE

持有：

- Q和Final RMSNorm；
- 同态私钥sk；
- 掩码种子和恢复向量池；
- Sampling状态；
- 在线Token和Embedding；
- GPU结果校验密钥。

### 模型主体GPU

持有改造后的Transformer、MoE、MLA和KV Cache，输出私有末端状态zL。

### LM Head GPU

持有公开或允许外包的W_head，执行普通矩阵乘法。可以和主体GPU处于同一个
不可信管理域。

### 离线掩码工厂

使用外部GPU和HE生成恢复向量，但不能解密随机掩码。

## 3. 在线协议

设：

    h in Z_q^d
    W in Z_q^(d×V)
    logits = hW

TEE从池中原子取出一个未使用条目：

    mask_id
    seed_r
    s = rW

TEE通过PRG从seed_r重新生成r，计算：

    u = h + r mod q

TEE向GPU发送：

    request_id
    model_version
    head_hash
    mask_id
    u

GPU计算：

    v = uW mod q

TEE验证v，随后恢复：

    logits = v - s mod q

TEE执行Sampling，得到真实Token；该Token不返回LM Head GPU。无论请求成功、
失败、超时还是连接中断，mask_id均永久进入CONSUMED或BURNED。

## 4. 离线掩码工厂

TEE生成HE密钥：

    (pk, sk, evk) = HE.KeyGen()

对每个掩码条目：

1. TEE生成seed_r并展开r；
2. TEE发送Enc_pk(r)；
3. GPU计算Enc_pk(rW)；
4. TEE解密得到s=rW；
5. TEE验证s；
6. TEE只保存seed_r和s；
7. seed_r与s使用TEE密封密钥加密落盘。

同一不可信GPU可以同时承担掩码工厂和在线计算，因为它在工厂阶段只看到
Enc(r)，在线阶段只看到h+r。它不能解密r或rW。

## 5. 恶意GPU结果验证

仅保护机密性不够，GPU还可能返回错误v。

对固定W，TEE预先生成k组秘密随机挑战c_j，并计算：

    t_j = W c_j

收到v后检查：

    v c_j == u t_j mod q

每次检查只需要TEE执行两个向量点积。若q为大素数或足够大的二进制扩展域，
单次伪造通过概率不高于约1/q；使用2至3个独立挑战进一步降低风险。

head_hash、model_version和挑战向量必须绑定。模型更新后旧验证材料失效。

## 6. 安全性质

### 输入隐藏

若r在Z_q^d上均匀、独立且仅使用一次，则对任意h：

    Pr[u=a | h] = 1/q^d

所以u=h+r与h统计独立，提供一次性加法掩码的完美隐藏。

### 输出隐藏

GPU观察v=uW。因为v是u的确定函数，而u与h独立，v不会比u额外泄露h。
真实logits=v-rW只在TEE内出现。

### 多轮安全

不同Token使用独立r。若重复使用同一r：

    u1-u2 = h1-h2

会泄露隐藏状态差分。因此任何不确定执行都必须烧毁掩码。

### 与主体状态联合观察

若同一服务器同时观察zL和u，它仍不知道Q和r。Head协议保证u不泄露h；
但zL的安全性仍依赖现有P/Q、噪声和攻击门禁，不能用Head协议替代。

### 完整性和可用性

Freivalds型检查提供结果完整性。拒绝服务无法通过密码协议消除。

## 7. 数值实现

强安全证明要求有限环，不能直接把无限精度实数证明套到BF16。

### Ring-Q8模式

- h使用固定公共scale量化到8bit环；
- W使用逐输出通道INT8；
- GPU执行UINT8×INT8到INT32的模运算GEMM；
- TEE在Z_(2^32)中恢复INT32 accumulator；
- TEE按公共weight scale和私有activation scale恢复logits。

私有版本与非私有对照必须使用同一个量化Head。协议可保证两者逐位一致，
但相对原BF16 Head属于量化近似。

### FP32研究模式

- h、r、rW和恢复均使用FP32；
- 外部GPU使用FP32或TF32关闭模式；
- 结果在实数代数中正确，但浮点消减可能产生误差；
- 随机掩码范围越大，隐藏越强，消减误差越大；
- 只能声明实测攻击安全和数值一致，不能声明有限环的一次性完美隐藏。

首版应同时实现两种模式，不能把FP32实验结论写成密码学证明。

## 8. 状态机

掩码状态：

    GENERATED
    VERIFIED
    AVAILABLE
    RESERVED
    CONSUMED
    BURNED
    CORRUPT

取掩码时先在TEE密封日志中把AVAILABLE原子改为RESERVED，再生成网络请求。
只有从未发出过u的RESERVED条目才允许由恢复工具回到AVAILABLE；默认策略
是一律BURNED，避免崩溃窗口造成重用。

## 9. DeepSeek-V3量级性能

参数：

    d = 7168
    V = 129280
    BF16 W = 1.853GB
    LM Head = 1.853GFLOP/Token

CPU TEE完整Head仅扫描权重的下界：

| TEE有效内存带宽 | 下界 |
|---:|---:|
| 50GB/s | 37.1ms |
| 100GB/s | 18.5ms |
| 200GB/s | 9.3ms |
| 300GB/s | 6.2ms |

外包在线路径目标：

| 部分 | 目标 |
|---|---:|
| PRG生成r和h+r | 小于0.1ms |
| GPU Head | 1至2.5ms，H100级估算 |
| PCIe往返 | 小于0.2ms |
| 验证与v-s | 0.2至0.8ms |
| Sampling | 0.1至1ms，取决于策略 |
| 合计 | 约1.5至4.5ms |

相对CPU TEE，LM Head步骤保守预计4至11倍；弱TEE可能更高。

若模型主体每Token为33ms，外包Head取2ms：

| CPU TEE Head | 原总时间 | 外包总时间 | 整体加速 |
|---:|---:|---:|---:|
| 37.1ms | 70.1ms | 35ms | 2.00倍 |
| 18.5ms | 51.5ms | 35ms | 1.47倍 |
| 9.3ms | 42.3ms | 35ms | 1.21倍 |
| 6.2ms | 39.2ms | 35ms | 1.12倍 |

## 10. 带宽和掩码池

每Token GPU返回V维结果：

- BF16：258.6KB；
- FP32或INT32：517.1KB。

30,000 Token/s：

- BF16返回流量：7.76GB/s，约62Gbps；
- FP32/INT32返回流量：15.51GB/s，约124Gbps。

因此GPU与TEE必须优先共置于PCIe Gen4/Gen5节点。跨网络至少需要100GbE，
正式INT32模式建议200GbE。

掩码r不必存储完整向量，只保存32字节种子。恢复向量s仍需V个INT32：

    517.1KB/Token

30,000 Token/s消耗：

    15.51GB/s
    55.85TB/小时

因此强安全一次性池不适合无限期30,000 Token/s。必须设置：

- pool_low_watermark；
- pool_target_seconds；
- 池耗尽时拒绝或降级，禁止重用；
- 离线工厂吞吐监控；
- 按租户独立池和密钥；
- 明确的最大持续Token率。

## 11. 两个运行档位

### STRONG-OTP

- 全维一次性r；
- 有限环；
- HE生成rW；
- 完美输入隐藏；
- 在线4至11倍Head加速；
- 掩码池成本高。

适用于敏感请求、可预测流量和正式安全验证。

### PRACTICAL-REMO

- M=M_pvt M_pub；
- 可复用R_pub；
- TEE恢复成本O(mV)；
- m<d存在残余补空间；
- 需要TokenInfer、TokenInv、VMA等实测门禁；
- 不得标注为完美隐藏。

适用于持续高吞吐和honest-but-curious场景。

## 12. 项目模块

建议新增：

    src/aloepri/secure_head/
        protocol.py
        ring.py
        quantization.py
        mask_pool.py
        mask_factory.py
        gpu_worker.py
        verifier.py
        sampler.py
        service.py

    configs/secure_head/
        qwen05b_fp32_research.yaml
        qwen05b_ring_q8.yaml
        deepseek_v3_capacity.yaml

    tests/secure_head/
        test_mask_correctness.py
        test_mask_non_reuse.py
        test_malicious_gpu.py
        test_ring_overflow.py
        test_float_error.py
        test_pool_crash_recovery.py

## 13. 0.5B验证门禁

1. 明文Head、外包Head在Ring-Q8下逐位一致；
2. FP32模式报告最大误差、NRMSE和greedy一致率；
3. 10万次请求无mask_id重复；
4. 崩溃、超时和重放不会复用掩码；
5. 错误GPU结果在门禁概率内全部被发现；
6. GPU工件中不存在h、logits、r或rW；
7. 抓包只能看到u和v；
8. 报告单Token和batch吞吐；
9. 分开报告在线延迟、掩码生成吞吐和存储消耗；
10. 未达到掩码补充速度时必须标记NO-GO，不用旧池掩盖结论。

## 14. 结论

该设计能够把LM Head主乘法从CPU TEE移到不可信GPU，并在在线阶段将TEE
复杂度从O(dV)降为O(d+V)。强安全模式的代价不是在线算力，而是一次性恢复
向量的离线生成、存储和传输。对低至中等吞吐，它是可实现的；对持续
30,000 Token/s，必须引入巨大的掩码工厂，或接受ReMO的安全折中。

