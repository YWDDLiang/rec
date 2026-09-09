# 自审与修正记录

日期2026-09-09。审计对象为本轮新增 `rec_pool` 与本轮文档；不宣称重新审计了远端每个历史快照。外部仓库只读，最新rec与Research commit记录在environment与sources文件。

## 已发现并修正

**1. 固定来源配额可能让选择器失去选择作用。** 初版先用最低成本行填满min quota。在min=max固定来源数量的实验里，所有自由度在几何评分前已被耗尽，方法退化为选最短样本。改为只计算剩余配额的最小完成成本，greedy逐条选择真实记录。新增测试证明固定配额时仍能选择高utility记录，并且不会因贪选昂贵样本破坏后续来源可行性。此问题来自本轮静态自审，不伪造一个未保存的失败训练结果。

**2. 训练plan可被伪造成本。** 原训练核对record_id/content，但成本日志可能直接信任plan。改为训练前逐项核对response/input token与cell，检查occurrence唯一性。增加伪造1000 response tokens的拒绝测试。

**3. 缓存只核对形状不足以发现内容被改。** 新增raw/unit/norm/loss/bucket/sign逐文件SHA256。变更缓存任意有限数值也会拒绝，code与字典另有hash。base_revision仍依赖调用者提供正确的固定版本；没有遍历全部基础权重计算hash，不夸大其校验范围。

**4. refilling可能遗失选择保护规则。** 选择JSON现在保存配置，回填继承protected与provenance cap。交换严格匹配task×domain和response长度；不为了达到请求数量偷偷跨源替换。

**5. 负边际候选不能被机械填满。** 加入可配置stop_nonpositive；没有未满足配额时，不因预算尚未用尽就强塞负边际样本。小池keep-all是另一个明确的保守策略，非效用最优定理。

**6. 排名评测中非法输出不能先删。** 新增交叉检查器保留无效和重复项原位置，避免把原第2名正确项洗成第1名；空输出为零，未知目标报错。

## 通过的回归检查

response-only一次causal shift；逐样本完整可训练梯度不改参数/模式/旧grad；OMP带符号；非正交字典Gram；符号/置换不变性；稀缺数据全保留；未知/不可行quota失败；同root不同任务不误合并；回填token/来源守恒；schedule重复曝光计数；梯度累积与整batch更新一致；actual训练确实消费选中的ID；跨partition的ID/root/编码泄漏失败。

## 仍然存在的研究和工程风险

logdet是方向覆盖代理，不是用户满意度或真实信息增益；本轮反例中它会放大错误稀有监督。目标centroid只是投影重构方向，不是精确优化器作用。固定checkpoint梯度会陈旧，训练阶段/跨模型稳定性尚未验证。字典是否优于直接raw梯度仍未证明。

greedy在综合约束下无全局最优保证；大量候选的吞吐未测试；Record仍整体加载，root reservoir保留所有派生行不保证row内存界。反传成本是主要开销，不能将memmap降低存储等同算力降低。

真实HF/PEFT权重、第三方native encoding、SID目录扩展、packing/DDP/RL、官方排序评测均未实际运行。轻量HF trainer重置AdamW，不恢复native optimizer/scheduler；严格复现请把ID计划接回native trainer。默认选择保护配置需要按真实业务填入；空protected不是自动保护所有少数能力。

Amazon示例task builder只做可核查历史统计和物料映射，不是完整LC-Rec任务复制。其已知用户时间划分与select/test用户独立性需在正式实验前统一；不能把诊断builder直接当已审核的论文split。

## 决策

算法内核和训练选择线路可以进入真实小规模验证；不通过“真实推荐有效性”门槛，不追加虚构提升百分比。下一步先完成原生平台基线与固定预算矩阵，而非继续增加在线控制模块。

## 最后一次交付检查

发现运行文档使用 `--batch-size`、`--reference-cache`，而入口最初只接受短参数名。已同时支持两组别名，并加入两项命令help回归检查；不需要真实模型即可检测这类文档—代码接口漂移。
