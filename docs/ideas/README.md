# 十一个方向：逐项数学、领域与验证（2026-09-09 同步）

| ID | 文档 | 当前决定 |
|---|---|---|
| **01** | **[多角色联合学习机会](01_joint_learning_and_complementarity.md)** | **主机制候选：从严格A+B互补升级为 coarse-mixing vs fine-grained Joint Learning Opportunity；真实推荐收益待证** |
| **02** | **[Behavioral Gradient Atoms](02_personalized_atoms.md)** | **表示/结构主候选：完整signed code、magnitude、Gram geometry；必须战胜raw gradient/KMeans才能保留核心贡献** |
| 03 | [时序互补与先修](03_trajectory_complementarity.md) | 精确二次原型；LLM轨迹未实现 |
| 04 | [观测/延迟感知价值](04_observation_aware_valuation.md) | 必要测量前提；真实日志识别待证 |
| 05 | [多目标完整组RL预算](05_vector_rl_allocation.md) | 分类策略原型；LLM-GRPO未接通 |
| 06 | [独立验真的批次获取](06_verified_batch_acquisition.md) | 可与01/11结合；未标注响应预测未完成 |
| 07 | [跨场景发现、切换和停止](07_cross_scene_exploration.md) | 独立方向；真实留存未证 |
| 08 | [策略一致SID与投机](08_policy_consistent_decoding.md) | 数学正确性模块；不独立主打新颖性 |
| 09 | [科学方法–数据组合](09_scientific_bundle_recommendation.md) | Science差异化应用；独立执行数据待建 |
| 10 | [前驱体与实验推荐](10_experiment_and_precursor_recommendation.md) | AI4Science替代方向；无物理实验 |
| **11** | **[预算条件下的行为训练选择](11_budget_conditioned_behavioral_selection.md)** | **regime主线：candidate richness × fixed train budget；连接01与02，但不替代它们** |

## 现在的主论文结构

不是“只做 Idea 11”。

更合理的三层结构是：

\[
\boxed{
\text{Idea 01: 是否存在细粒度联合学习机会}
\rightarrow
\text{Idea 02: 用什么 learned behavioral units 表示它}
\rightarrow
\text{Idea 11: 在什么数据/预算 regime 下它有价值}
}
\]

### 01
回答 **why selection / fine-grained mixing is needed**。

### 02
回答 **why gradient atoms rather than source labels / semantic clusters / raw sample scores**。

### 11
回答 **when selection should beat full-data or uniform exposure**。

旧 `exp/01_02` 负实验不删除、不美化：它们是新版问题定义的直接动因。
