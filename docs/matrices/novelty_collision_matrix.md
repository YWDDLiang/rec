# 新颖性碰撞矩阵：命名不同不是贡献不同

| 候选论点 | 已有直接覆盖 | 本库允许保留的差异 | 最危险反证 | 决定 |
|---|---|---|---|---|
| 梯度筛选LLM4Rec | DEALRec/LESS/GORACS | 推荐反馈的带符号互补、数据不足诊断 | 简单分层/直接LP解释全部 | 单独不通过 |
| 多目标靠采样 | MoRec/AMA/DoReMi | 当前池可达域与完整对偶面获取 | MoRec+简单主动获取相同 | 条件保留 |
| 面向下一更新 | Ren/Filter-then-Weight | 互补批次无单例增益、可达性区分 | 只是已有目标重写 | 不独立声明新 |
| 动态atom/domain | GradientAtoms/DoGraph/CLIMB | 控制共享因素后可预测领域能力 | 随机字典/普通聚类相同 | 本轮降为可选 |
| 时序/先修 | Stagewise/MiniRec/CAPT | 预测并干预跨反馈互补顺序 | 同多重集顺序无差异 | 候选贡献 |
| RL奖励+梯度去冗余 | MiniRec/I-PPO/ReCast | 固定estimand的vector完整组预算 | 标量方差+相同纠正已足够 | 扩展，不主打搬运 |
| 观测校正 | IPS/DR | 观测冲突与学习冲突分离 | 仅mask或DR足够 | 领域前提 |
| 主动获取合成数据 | 既有主动学习/GAIA/NASH | 独立验真的互补批次；完整face | 免费使用真实未标注梯度 | 原型非完成系统 |
| PageRank探索疲劳 | PIE/FRec/KuaiSim | 发现/切换/停止分开识别且可校准 | 固定ε/曝光惩罚已足够 | 独立研究线 |
| logits+speculation | AtSpeed/SpecGR/经典SD | 新q的策略一致高效执行 | 仅复现已有恒等式 | 当前不独立通过 |
| 科学论文/数据集推荐 | SPECTER/SciRepEval/DataFinder | 独立验证method–dataset接口互补 | 规则join胜出 | 应用方向 |
| 前驱体/实验推荐 | ScienceAdv2023/失败实验/主动学习 | 失败类型与资源约束的实际证据 | 标准BO或检索足够 | AI4Science替代 |

详细一手来源见[索引](../../references/README.md)。没有任何一项被标为“首次”；检索未发现完全同构工作不是全世界不存在的证明。本库倾向主线01+06，03作为可证伪的时序扩展；如果03无效则删去，不为凑两条贡献保留。
