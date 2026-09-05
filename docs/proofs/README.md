# 数学核验索引

全部证明均写在对应方向文档中；本文件避免重复抄写而造成版本漂移。

| 对象 | 证明位置 | 自动检验 |
|---|---|---|
| max-min强对偶、可达性、完整face批次等价、非次模反例、误差界 | [01](../ideas/01_joint_learning_and_complementarity.md) | test_frontier/test_uncertainty |
| 加权残差正交、不变性、候选提名的full-pricing边界 | [02](../ideas/02_personalized_atoms.md) | test_atoms_trajectory |
| 伴随权重导数与二次顺序括号 | [03](../ideas/03_trajectory_complementarity.md) | test_atoms_trajectory有限差分 |
| MAR与policy DR条件期望 | [04](../ideas/04_observation_aware_valuation.md) | test_observational_allocation |
| p/q无偏、二阶矩凸性、LOO基线、成本分配 | [05](../ideas/05_vector_rl_allocation.md) | test_rl/test_observational_allocation |
| 获取成本、完整face必要充分、独立标签噪声风险 | [06](../ideas/06_verified_batch_acquisition.md) | test_science_acquisition/test_frontier |
| PPR收缩、吸收状态逆、单步混合界 | [07](../ideas/07_cross_scene_exploration.md) | test_exploration_decoding |
| trie望远镜、投机接受残差恒等式 | [08](../ideas/08_policy_consistent_decoding.md) | test_exploration_decoding |
| 概率覆盖次模、基数greedy、pair互补反例 | [09](../ideas/09_scientific_bundle_recommendation.md) | test_science_acquisition |
| Gaussian信息熵差、后验、次模条件 | [10](../ideas/10_experiment_and_precursor_recommendation.md) | test_science_acquisition |

通过数值恒等式不证明统计假设成立。光滑度、误差界、positivity、独立反馈、正确状态转移和科学兼容性都需要外部证据。旧优化理论的有效迁移可以支撑方法，但不自动构成理论新颖性。
