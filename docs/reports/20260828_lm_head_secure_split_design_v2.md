# LM Head 安全拆分与云端 GPU 外包方案

日期：2026-08-28

## 1. LM Head 到底算什么

对隐藏维 d、词表大小 V 的模型，最后隐藏向量为 h，LM Head 权重为
W_head（形状 V×d）：

    logits = h × W_head^T
    next_token = Sample(logits)
    next_embedding = E[next_token]

因此，不使用词表置换时，只要普通云服务器看到完整 logits、最大值位置或
Embedding 查表地址，它就知道输出 Token。LM Head 虽只占 671B 模型很小
一部分，但每生成一个 Token 都要计算一次 d×V 的矩阵向量乘法。

## 2. 主方案：可信网关只保留模型头尾

不把完整模型放入可信域。可信网关只保留 Tokenizer、Embedding、Final
RMSNorm、LM Head 和采样；云端保留 Transformer、MLA、MoE、KV Cache。

完整流程：

1. 网关本地分词，得到真实 Token y。
2. 网关计算私有 Embedding：z0 = E[y]P。
3. 云端只收到 z0，在 P/Q 私有坐标中运行模型主体。
4. 云端返回最后私有隐藏状态 zL。
5. 网关计算 h = FinalRMSNorm(zL Q)。
6. 网关计算 logits = h W_head^T，并在本地采样 next_token。
7. 网关查 E[next_token]P，发送给云端继续下一步 decode。

云端不接触真实输入 Token、输出 Token、logits、Embedding 查表地址或采样
结果。

### 2.1 每 Token 新鲜随机化

当前扩维满足 P Q = I，且 D>d。网关可对每个 Token 生成随机 n，使 nQ=0，
然后发送：

    z_tilde = E[y]P + n

因为：

    z_tilde Q = E[y] P Q + n Q = E[y]

所以有效计算不变，而同一个 Token 每次发送的隐藏向量不同。该性质需要在
当前 exact-metric RMS、Residual 和 KV Cache 上做逐层等价测试后才能发布。

### 2.2 DeepSeek 671B 量级成本

按 V=129280、d=7168、D=7424：

- 私有 LM Head：959,774,720 参数，BF16 约 1.92 GB；
- 私有 Embedding：约 1.92 GB；
- 网关头尾权重：合计约 3.84 GB；
- LM Head：约 1.92 GFLOP/生成 Token。

1000 个持续生成会话、每个 30 Token/s：

    1000 × 30 × 1.92 GFLOP = 57.6 TFLOP/s

这需要 GPU 网关，但远小于在本地运行完整 671B。

BF16 下，每个生成 Token 要接收一个 D 维末端隐藏向量，并发送一个 D 维
下一步 Embedding：

    2 × 7424 × 2 Byte = 29,696 Byte/Token

30,000 Token/s 约 890.9 MB/s，即 7.13 Gbps 原始载荷。10GbE余量很小，
生产建议 25GbE。

## 3. 生产折中：只把头尾放入机密 GPU

如果终端或离线网关只有CPU，不需要把完整671B放进机密GPU。可以增加一个
独立的“机密头服务”，其中只加载约3.84GB的Embedding和LM Head，并完成：

    接收云端私有末端隐藏状态
    → 恢复有效坐标
    → Final RMSNorm
    → LM Head
    → 采样
    → 下一Token的私有Embedding

大模型主体仍在普通GPU集群；只有头服务使用带远程证明的机密GPU或受信任
GPU网关。客户端验证证明后建立加密通道。云主机管理员不能直接读取头服务
显存，普通模型集群也只看到私有隐藏向量。

这不是把全部模型放入机密环境：671B主体、MoE专家、MLA和KV Cache均留在
普通GPU；机密域仅保存约0.6%的头尾参数。该方案比完整机密推理更容易部署，
也比FHE实时性更强。它仍依赖机密GPU硬件、驱动和证明链，必须在目标云平台
做远程证明与攻击面验收。

头服务可以按词表行分片：

- Greedy：每个分片返回本分片最大值和索引，再做一次全局最大值；
- Top-k：每个分片返回本地Top-k，再合并为全局Top-k；
- 精确Softmax采样：可使用分布式max、sum-exp和抽样，或Gumbel-Max；
- Embedding：按最终Token路由到对应词表分片。

这样可扩展头服务吞吐，而无需在单张GPU保存全部LM Head。

## 4. 备选：只把 LM Head 用同态加密外包

如果可信网关没有 GPU，可让网关加密 h，普通云 GPU 在不知道 h 的情况下
计算 LM Head：

1. 普通模型云返回私有末端状态zL给网关。
2. 网关持有Q，恢复h = FinalRMSNorm(zL Q)，再计算c_h = Enc(h)。
3. 云GPU持有原始明文W_head，计算：
   c_logits = c_h W_head^T = Enc(h W_head^T)。
4. 网关解密logits，本地采样并查Embedding。

LM Head 是线性层，同态乘法深度只有一层；不必在密文里做 Softmax。若网关
解密 logits 后采样，也可避开昂贵的同态 argmax/top-p。

安全配置中，普通模型云不得同时取得Q或可直接作用于zL并产生真实logits的
完整变换后Head。否则服务器可以绕过同态计算，直接用明文zL恢复Token。

但主要代价仍很大：

- CKKS是近似计算，接近的logits可能改变Top-1；
- BFV/BGV定点方案必须证明缩放、溢出和排序边界；
- V约13万会产生大量密文打包、旋转、密钥切换和输出密文；
- HE使用GPU不等于普通GEMM速度；
- 多用户SIMD可提高吞吐，但会增加批等待。

公开的完整FHE 7B/8B实现即使使用8张RTX4090仍约33秒/生成Token。head-only
会简单很多，但必须先做原型，不能直接宣称实时可用。

原型顺序：

1. Qwen2.5-0.5B实现CKKS head-only；
2. 分别测加密、GPU MatVec、回传、解密；
3. 测logits误差、Top-1一致率和margin；
4. 增加多用户SIMD；
5. 扩展到DeepSeek-V3缩小配置并拟合V、d复杂度。

## 5. 一次性掩码外包

选取每Token唯一随机a：

    h = (h-a) + a
    server_result = (h-a) W_head^T
    recovered_logits = server_result + a W_head^T

服务器看不到h和真实logits，在线只运行普通GEMM。但网关必须预先持有每个
Token唯一的aW_head^T。按d=7168、V=129280、FP32估算，每个掩码对约545KB；
30,000 Token/s会消耗约16.4GB/s掩码池。生成掩码池本身又要做同等规模的
LM Head乘法，因此只是把算力移到离线阶段，适合突发或演示，不适合作为长期
高并发主方案。

可用Slalom/Freivalds方法校验服务器是否返回了正确乘法结果。

## 6. 两个不串通服务器

网关秘密分享h：

    h = r + (h-r)

服务器A计算rW^T，服务器B计算(h-r)W^T，网关相加。单台服务器看不到h，
但两台都要做完整LM Head，而且两个V维结果都要回传；若继续在服务器间做
安全argmax，还需要对约13万项做安全比较。成本和不串通假设都不适合作为
首选。

## 7. 为什么普通矩阵拆分不够

LM Head按行、按列、按GPU或低秩拆分只能分摊计算。只要某个不可信节点最终
看到完整logits、最大值位置或Embedding索引，它仍知道输出Token。

对logits施加普通秘密线性变换也不存在免费方案：能对任意logits保持argmax
的共同正比例缩放和共同偏移不会隐藏最大值索引；任意坐标混合又会改变argmax。
因此必须使用可信头尾、同态计算、MPC，或恢复词表秘密置换中的至少一种。

## 8. 结论

正式主路线：

    可信GPU网关：
    Tokenizer + Embedding + Final Norm + LM Head + Sampling

    不可信大模型云：
    P/Q坐标中的Transformer + MLA + MoE + KV Cache

    增强：
    每Token零空间随机掩码 nQ=0

它没有把完整671B放入可信域，只把约3.84GB头尾权重和约1.92GFLOP/Token
的LM Head放到网关。

若产品要求网关只有CPU，则研发head-only HE，让云GPU做密文线性层。该路线
密码学边界更强，但在Qwen0.5B实测前只能列为研发方案。

如果目标云支持可信执行硬件，优先级为：

    小型机密GPU头服务
    > 可信本地GPU网关
    > head-only HE研发原型
    > 一次性掩码或双服务器

## 9. 资料

- Slalom: https://arxiv.org/abs/1806.03287
- Talaria/ReMO: https://aclanthology.org/2026.acl-long.4/
- Hyperion: https://aclanthology.org/2026.acl-long.644/
- Cutmax: https://aclanthology.org/2025.findings-ijcnlp.111/
- BLB: https://www.usenix.org/conference/usenixsecurity25/presentation/xu-tianshi
- Full FHE Llama 2/3: https://arxiv.org/abs/2601.18511

## 10. 2026-08-29：将 LM Head 主乘法移出 TEE

如果要求 TEE 不执行 d×V 主乘法，可以采用一次性掩码外包：

    TEE:     choose fresh r
    TEE→GPU: u = h + r
    GPU:     v = u W_head^T
    TEE:     logits = v - r W_head^T

只要 r 是在正确有限环上的全空间均匀一次性掩码、每 Token 只使用一次，
GPU 观察的 u 不泄露 h。GPU执行完整普通LM Head，TEE在线仅执行向量加减和
采样。

关键不是公式，而是 TEE 必须提前得到 s=rW_head^T，同时不能让同一个不可信
GPU知道 r。可选来源：

1. 可信离线预处理器生成(r,s)；
2. 两个严格不串通的GPU分别承担预处理和在线计算；
3. TEE发送Enc(r)，普通GPU用HE异步生成Enc(s)，TEE解密后存入掩码池。

第三种是最符合“安全借用GPU”的混合方案：昂贵HE不在在线请求路径中，
在线LM Head仍是普通BF16/FP16 GEMM。

但掩码池不能无限扩展。DeepSeek量级下，BF16的(r,s)约273KB/生成Token；
30,000 Token/s每小时消耗约29.5TB。该方案适合小规模、突发流量或作为
HE异步补充池，不适合单独承担长期高并发。

Talaria/ReMO用可复用R_pub降低掩码存储：

    M = M_pvt M_pub
    R_pub = M_pub W
    logits = (h+M)W - M_pvt R_pub

但当m足够大以覆盖主要隐藏空间时，TEE仍需执行m×V恢复乘法。m=0.5d时
恢复成本就是约半个LM Head；论文也明确指出m<d留下d-m维残余补空间。

因此，没有硬件机密GPU且要求强隐藏时，三种成本不可同时消失：

- 直接HE：在线密码计算昂贵；
- 一次性掩码：在线快但预处理和存储昂贵；
- 两服务器MPC：在线普通计算但带宽翻倍并依赖不串通。

## 11. 相对在 TEE 内完整计算 LM Head 的加速估算

比较必须区分 CPU TEE 与机密 GPU TEE。

DeepSeek-V3量级取d=7168、V=129280，BF16 Head约1.853GB，每Token约
1.853GFLOP。

CPU TEE完整Head是GEMV，主要受每Token扫描1.853GB权重的内存带宽约束。
若有效带宽为50、100、200、300GB/s，仅权重扫描下界分别约37.1、18.5、
9.3、6.2ms；实际还包含矩阵核、归一化和采样。

一次性掩码外包的在线阶段，普通GPU执行Head，TEE只做O(d+V)加减和采样。
共置PCIe条件下估计：

- H100级普通GPU：Head步骤约1至2.5ms；
- A100级普通GPU：约1.5至3.5ms；
- RTX4090级普通GPU：约3至5ms。

因此相对CPU TEE：

- 强64核CPU TEE：LM Head步骤约3至12倍；
- 常见4至16 vCPU TEE：约6至30倍；
- 工程保守口径：约4至11倍，与Slalom公开的私有外包实测范围一致。

以上只计在线路径，并假设一次性(r,rW)已经生成。如果把HE掩码池生成成本
全部摊回每Token，总资源成本不一定下降。

相对H100机密GPU TEE则不同：公开研究报告机密模式吞吐损耗通常约4%至8%，
另有并发测试观察到约11.5%至20.2%。普通GPU外包最多省掉这部分，但增加
掩码、传输和恢复后预计只有0.8至1.1倍，可能反而更慢。因此已有机密GPU时
不应采用本外包协议。

端到端加速使用Amdahl公式：

    S_total = 1 / ((1-f) + f/S_head)

其中f是基线中LM Head占单Token时间的比例。若Head加速10倍：

- f=5%：端到端约1.047倍；
- f=10%：约1.099倍；
- f=20%：约1.220倍；
- f=40%：约1.563倍。

对671B，Head FLOP占比不大；但若主体在GPU集群、Head却在CPU TEE，Head的
时间占比可能因设备不匹配达到20%至40%，此时外包对端到端延迟才有明显价值。
