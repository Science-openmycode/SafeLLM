# LM Head 掩码外包数值正确性设计

日期：2026-08-29

## 1. 问题

浮点掩码外包：

    v = (h+r)W
    logits = v-rW

在实数中严格成立，但在BF16/FP16/FP32中可能因大数相消、累加顺序和舍入产生
误差。扩大r可以增强经验隐藏，却会放大相消误差。

有限环掩码可以精确恢复整数结果，但量化Head与原始BF16 Head之间又存在
量化误差。因此不能同时把以下两句话混为一谈：

- 私有量化Head与非私有量化Head逐位一致；
- 私有Head与原始BF16 GPU Head逐位一致。

## 2. 方案：可认证候选修正 Head

正式方案命名为 Certified Candidate Correction Head，简称CCCH。

核心思路：

1. 外部GPU用有限环掩码计算全词表近似logits；
2. TEE为每个近似logit计算一个严格误差上界；
3. TEE找出所有可能成为真实Top-1的候选；
4. TEE只使用原始权重重算这些候选行；
5. 如果候选集合或误差证明异常，回退到CPU完整Head；
6. 永不在无法证明时直接采用近似Top-1。

## 3. 全词表安全近似Head

原始Head：

    l_v = h dot W_v

对h和W量化：

    hq = QuantizeActivation(h)
    Wq = QuantizeWeight(W)

外部GPU通过STRONG-OTP协议计算：

    aq_v = hq dot Wq_v  over Z_(2^32)

TEE恢复整数accumulator并反量化：

    lhat_v = scale_h × scale_W_v × signed(aq_v)

量化后的私有与非私有路径在同一环、同一scale、同一kernel下逐位一致。

首版使用：

- activation INT8；
- weight INT8逐输出通道或逐组scale；
- INT32模2^32累加；
- CUDA kernel必须验证溢出为确定性wrap，禁止饱和；
- 恢复后的真实未掩码accumulator必须落在有符号INT32安全范围。

DeepSeek量级的真实INT8点积上界：

    d × 127 × 127
    = 7168 × 16129
    = 115,612,672

小于2^31，真实恢复值可安全解释为signed INT32。被掩码的中间值允许在
Z_(2^32)中回绕。

## 4. 严格误差上界

定义：

    delta_h = h-hq
    delta_W_v = W_v-Wq_v

则：

    hW_v-hqWq_v
    = delta_h W_v + hq delta_W_v

将隐藏维按组g切分，使用Cauchy-Schwarz：

    error_v <= sum_g(
        norm2(delta_h_g) × norm2(W_v,g)
        + norm2(hq_g) × norm2(delta_W_v,g)
    )

转换时预计算：

    A[v,g] = norm2(W_v,g)
    B[v,g] = norm2(W_v,g-Wq_v,g)

在线TEE只计算：

    a[g] = norm2(delta_h_g)
    b[g] = norm2(hq_g)
    error = A a + B b

再加入：

- INT32反量化舍入界；
- FP32候选重算误差界；
- 参考GPU kernel与TEE dot kernel的实测最大差；
- 配置安全余量。

为了降低TEE工作量，采用两级证书：

### 一级全局界

每个词表行只保存两个全局范数，先快速生成较宽的error_v。

### 二级分组界

只对一级界未能排除的行使用group-wise范数。建议group_size为128或256。

二级证书仍然过宽时，进入完整CPU回退。

## 5. Greedy严格候选选择

对每个Token v得到区间：

    lower_v = lhat_v-error_v
    upper_v = lhat_v+error_v

计算：

    best_lower = max_v(lower_v)

候选集合：

    C = {v | upper_v >= best_lower}

任何不在C中的Token都有：

    upper_v < best_lower

因此它不可能是真实Top-1。真实Top-1必定位于C。

TEE从加密的原始Head归档中只读取C对应的权重行，使用原始h重新计算：

    exact_v = ReferenceDot(h, W_v), v in C

最终：

    token = argmax(exact_v)

如果C的大小超过max_candidates，例如256，则执行CPU完整Head回退。这保证
系统可能变慢，但不会返回未经证明的Token。

## 6. Top-k和Top-p

### Top-k

使用第k大的lower值作为门槛，将所有upper超过门槛的行加入C，重算后得到
严格Top-k。

### Top-p

Top-p依赖全词表exp和累计概率。仅重算Top候选不能一般性保证与原始分布完全
一致。

正式策略：

- EXACT_GREEDY：支持；
- EXACT_TOP_K：支持；
- EXACT_TOP_P：证书能证明尾部概率上界时支持，否则CPU完整Head回退；
- APPROX_TOP_P：允许时必须单独标注，不进入严格一致验收。

## 7. 原始BF16参考的定义

不同GPU、Tensor Core模式和累加顺序可能给出略有不同的浮点logits。必须固定
参考语义：

    dtype
    TF32开关
    accumulation dtype
    reduction order
    tie-breaking
    sampler
    random seed

推荐定义Yinbian Reference Head：

- 输入h为FP32；
- W从BF16精确转FP32；
- 固定分块顺序的FP32 dot；
- ties按最小Token ID；
- 同一参考kernel用于明文和候选重算。

这可以保证产品内部逐位可复现。与第三方PyTorch/CUDA kernel的关系作为
兼容性指标报告，不把硬件未规定的累加顺序当成数学真值。

## 8. 性能

DeepSeek量级完整Head：

    d×V = 926,679,040 MAC/Token

若候选数K=64，TEE精确重算：

    d×K = 458,752 MAC/Token

占完整Head：

    K/V = 64/129280 = 0.0495%

K=256时也只有：

    0.198%

误差证书的一级全局界为O(V)，二级分组界只对不确定行运行。正常情况下，
TEE在线主要消耗变为：

    O(d+V+Kd)

而不是：

    O(dV)

预计相对纯Ring-Q8外包增加0.2至1.5ms；若触发二级证书可能增加1至5ms；
CPU完整回退按原TEE Head时间计算。

平均时间：

    Tavg = Tfast
           + p_second × Tsecond
           + p_fallback × Tcpu_full

产品门禁建议：

- median |C| <= 16；
- p99 |C| <= 128；
- fallback_rate <= 0.1%；
- greedy一致率必须100%，不能用平均值替代；
- fallback请求也必须返回正确结果。

## 9. 存储

TEE不需要把1.853GB Head全部常驻普通内存，可保存：

- 行可寻址、TEE密封的原始BF16 Head归档；
- 每行offset和SHA-256/Merkle证明；
- 一级误差范数；
- 可选二级group范数。

K=64时每Token读取原始候选行：

    64 × 7168 × 2 Byte
    = 917,504 Byte

约0.92MB，远小于扫描完整1.853GB Head。

候选行应使用LRU缓存；缓存内容只能位于TEE内存。

## 10. 安全性质

- hq在发送前由全空间一次性环掩码保护；
- GPU只看到均匀的hq+r；
- 全词表近似logits只在TEE恢复；
- error、C、候选原始权重和最终Token不发给GPU；
- CPU完整回退也全部在TEE内；
- 候选集合大小可能通过时间侧信道泄露模型置信度，需固定时间桶和填充；
- 候选重算固定到K_pad行，真实候选不足时加入伪行；
- 回退请求使用统一响应时间或批次隔离；
- 掩码不能因回退而重用。

## 11. 为什么优于直接提高量化位宽

把INT8改为INT16或FP32仍不能同时解决：

- 有限环掩码；
- GPU高效Tensor Core；
- 原BF16结果；
- 溢出和浮点消减。

CCCH不要求全词表近似logits完全等于BF16，只要求误差界正确，并用极少量原始
行完成最终裁决。它把数值问题从“希望误差足够小”改为“用区间证明输出一定
正确”。

## 12. 实施门禁

### 数学单测

- 随机小矩阵穷举验证真实Top-1始终属于C；
- 故意放大量化误差时必须扩大C或回退；
- 模2^32 wrap恢复正确；
- 错误scale、overflow和NaN立即失败。

### Qwen0.5B

- 200条固定prompt逐Token greedy一致；
- 完整产品100题一致；
- 至少100万随机隐藏向量候选覆盖率100%；
- 报告候选数分布和回退率；
- 攻击者只观察masked integer tensor；
- GPU结果篡改被验证器发现。

### DeepSeek缩小配置

- 使用官方V和缩小d评估词表扫描；
- 外推d=7168时的候选读取、证书和PCIe成本；
- 不用Qwen结果直接宣称671B通过。

## 13. 最终决策

正式数值路径：

    STRONG-OTP Ring-Q8全词表近似
    + Certified error interval
    + TEE exact candidate correction
    + full CPU fallback

这样同时获得：

- 外部GPU承担接近100%的LM Head主计算；
- 一次性有限环掩码的强机密性；
- 正常路径TEE只重算极少数Token行；
- Greedy和Top-k具有可验证的输出一致性；
- 极端数值情况下宁可回退变慢，也不静默返回错误Token。
