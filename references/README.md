# 一手来源索引

核验日期：2026-09-05。共 49 项。没有把摘要核验标成全文审计，没有独立复现第三方论文结果。

本索引保存短事实摘要；idea 文档中的新机制和证明是本项目推导，不是被引用论文已经提出。未核验的代码 commit 留空，不伪造 pin。

## tiger
[Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) · 2023 · paper
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：Semantic ID 将物品组织为可生成的离散序列。
本库判断：生成式推荐不自动等于通用语言预训练模型。

## lcrec
[Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation](https://arxiv.org/abs/2311.09049) · 2023/2024 · paper
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：LC-Rec 对齐语言语义与推荐协同语义。
本库判断：与随机初始化 ID 模型对比时必须控制文本、预训练和 SID 质量。

## minionerec
[MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation](https://arxiv.org/html/2510.24431v1) · 2025 · preprint/project
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：开放 SID、SFT 与推荐导向 RL 的实验路径。
本库判断：框架存在不等于本交付已复现其结果；与 MiniRec 是不同工作。
已定位的项目：[仓库](https://github.com/AkaliKong/MiniOneRec)；未在本次运行其完整基线，license/commit 必须在正式实验前记录。

## openonerec
[OpenOneRec Technical Report](https://arxiv.org/html/2512.24762v2) · 2025 · preprint/project
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：公开推荐基础模型和 RecIF-Bench 多类推荐任务。
本库判断：pro 模型额外数据优势不可归因给选择算法。
已定位的项目：[仓库](https://github.com/Kuaishou-OneRec/OpenOneRec)；未在本次运行其完整基线，license/commit 必须在正式实验前记录。

## genrec
[GenRec: An LLM-Backed Recommendation Ranker at Netflix](https://arxiv.org/html/2608.10257) · 2026 · preprint
阅读层级：L0+ selected HTML methods inspected; not full reproduction。
论文/官方材料支持：语言模型主干接目录感知的排序输出，采用奖励加权训练。
本库判断：不能把文中讨论的 RL 潜力写成本文已经部署 GRPO；也不是本项目自己的线上证据。

## recast
[ReCast: Recasting Learning Signals for Reinforcement Learning in Generative Recommendation](https://arxiv.org/html/2604.22169v1) · 2026 · preprint
阅读层级：L0+ selected HTML methods inspected; not full reproduction。
论文/官方材料支持：研究生成式推荐中 RL 的学习信号组织。
本库判断：RL 增益可能来自奖励信号重构，不能都解释成数据选择收益。

## dealrec
[Data-efficient Fine-tuning for LLM-based Recommendation](https://arxiv.org/html/2401.17197v2) · 2024 · SIGIR
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：用推荐数据影响力与学习难度组织 LLM 微调样本。
本库判断：为推荐挑高价值 SFT 数据不是新问题。
已定位的项目：[仓库](https://github.com/Linxyhaha/DEALRec)；未在本次运行其完整基线，license/commit 必须在正式实验前记录。

## goracs
[GORACS: Group-level Optimal Transport-guided Coreset Selection for LLM-based Recommender Systems](https://arxiv.org/html/2506.04015v1) · 2025 · KDD
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：将组级选择、梯度信息与最优传输结合。
本库判断：从单样本改成组或加入覆盖度不足以独立主张新颖。
已定位的项目：[仓库](https://github.com/Mithas-114/GORACS)；未在本次运行其完整基线，license/commit 必须在正式实验前记录。

## morec
[A Data-Centric Multi-Objective Learning Framework for Responsible Recommendation Systems](https://arxiv.org/html/2310.13260v1) · 2024 · WWW
阅读层级：L0+ selected HTML methods inspected; not full reproduction。
论文/官方材料支持：通过目标导向数据采样与控制机制协调推荐的多个目标。
本库判断：数据视角的多目标推荐已有直接 WWW 基线。

## minirec
[MiniRec: Data-Efficient Reinforcement Learning for LLM-based Recommendation](https://arxiv.org/html/2602.04278v1) · 2026 · preprint
阅读层级：L0+ selected HTML methods inspected; not full reproduction。
论文/官方材料支持：RL 选样结合可学习性、梯度代表性、多样性与课程。
本库判断：单纯从 SFT 迁移到 RL，或奖励加梯度加去重，会直接碰撞。

## capt
[Data-Efficient Adaptation to Contextual Shifts in LLM-based Conversational Recommendation](https://aclanthology.org/2026.findings-acl.1114/) · 2026 · Findings of ACL
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：研究会话推荐在上下文变化下的数据高效适配。
本库判断：时序变化与选择也不是空白；当前只做出版页/摘要层核验。

## less
[LESS: Selecting Influential Data for Targeted Instruction Tuning](https://proceedings.mlr.press/v235/xia24c.html) · 2024 · ICML
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：低秩梯度信息用于面向目标的指令数据选择。
本库判断：LoRA 梯度与验证集对齐是工具，不是本项目首创。

## ogs
[Training Data Selection with Gradient Orthogonality for Efficient Domain Adaptation](https://arxiv.org/html/2602.06359v1) · 2026 · preprint
阅读层级：L0+ selected HTML methods inspected; not full reproduction。
论文/官方材料支持：以梯度几何关系进行领域适配样本选择。
本库判断：正交、保护已有能力等关键词已被使用，须比较机制而非名称。

## ren-reweight
[Learning to Reweight Examples for Robust Deep Learning](https://arxiv.org/html/1803.09050v3) · 2018 · ICML
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：利用干净验证集的元梯度调节训练样本权重。
本库判断：验证驱动重加权不是新的数学原语。

## filter-weight
[Filter-then-Weight: Online Data Selection and Reweighting for LLM Fine-Tuning](https://arxiv.org/html/2604.00001v2) · 2026 · preprint
阅读层级：L0+ selected HTML methods inspected; not full reproduction。
论文/官方材料支持：按当前优化器状态构造目标导向更新，并解耦过滤与权重优化。
本库判断：更新导向、优化器感知、样本交互都已有工作，不能作为独占 claim。

## partition-match
[Minibatch Selection for Language Models via Partition Matroid Constrained Gradient Matching](https://arxiv.org/html/2606.07954v2) · 2026 · preprint
阅读层级：L0+ selected HTML methods inspected; not full reproduction。
论文/官方材料支持：研究分区拟阵约束下的梯度匹配选批。
本库判断：加入场景配额和集合约束也需直接比较。

## gaia
[Online Data Selection for Instruction Tuning via Gaussian Processes](https://arxiv.org/abs/2606.30077) · 2026 · preprint
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：将高斯过程用于在线指令数据选择。
本库判断：预测数据价值加不确定度不是从无到有。

## atoms
[Gradient Atoms: Unsupervised Discovery, Attribution and Steering of Model Behaviors via Sparse Decomposition of Training Gradients](https://arxiv.org/html/2603.14665v2) · 2026 · preprint
阅读层级：L0+ selected HTML methods inspected; not full reproduction。
论文/官方材料支持：在 EKFAC 预条件特征中稀疏分解文档梯度，分析共享行为更新。
本库判断：原文边界包括表面行为检测和低一致性原子；不是推荐去重有效性的证明。
已定位的项目：[仓库](https://github.com/jrosseruk/gradient_atoms)；未在本次运行其完整基线，license/commit 必须在正式实验前记录。

## rise
[Sketching the Readout of Large Language Models for Scalable Data Attribution and Valuation](https://arxiv.org/html/2604.16197v2) · 2026 · preprint
阅读层级：L0+ selected HTML methods inspected; not full reproduction。
论文/官方材料支持：RISE 从 readout 的词汇残差和语义误差信号构造归因近似。
本库判断：PPT 中 PPO rollout 过滤内容不能直接归于此论文；原文全文未检出 PPO。

## ippo
[Learning from the Right Rollouts: Data Attribution for PPO-based LLM Post-Training](https://arxiv.org/abs/2604.01597) · 2026 · preprint
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：I-PPO 使用验证梯度方向识别并过滤反向影响的 episode。
本库判断：PPT 对 rollout 过滤的描述更对应本篇；本库不把 episode 过滤无条件用于 GRPO 组内。

## nash
[Is Data Shapley Not Better than Random in Data Selection? Ask NASH](https://arxiv.org/abs/2605.10684) · 2026 · preprint
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：对 Shapley-informative components 进行非线性聚合。
本库判断：NASH 的展开不是 Nash bargaining；本文不能仅靠强调组合收益宣称首次。

## stagewise
[Influence Dynamics and Stagewise Data Attribution](https://arxiv.org/abs/2510.12071) · 2025 · preprint
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：研究归因随训练阶段变化。
本库判断：时序敏感归因已存在；本库二次模型只是精确机制检验。

## dograph
[Rethinking Data Mixing from the Perspective of Large Language Models](https://aclanthology.org/2026.acl-short.28/) · 2026 · ACL short
阅读层级：L0+ selected HTML methods inspected; not full reproduction。
论文/官方材料支持：DoGraph 根据训练梯度结构学习和调整数据混合单元。
本库判断：动态梯度域不是新的切入；PPT 的预印本状态已经有正式会议页面。

## climb
[Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](https://arxiv.org/abs/2504.13161) · 2025 · paper/project
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：从数据表征结构出发搜索预训练混合配置。
本库判断：learned domain 本身已有依据，不应继续只按人工场景桶讨论。

## automixalign
[AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs](https://aclanthology.org/2025.acl-long.990/) · 2025 · ACL
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：使用 specialist 参考与任务缺口组织偏好数据混合。
本库判断：specialist 的实测损失是经验参照，不自动是真实最优下界。

## mde
[Optimizing Pre-Training Data Mixtures with Mixtures of Data Expert Models](https://aclanthology.org/2025.acl-long.1564/) · 2025 · ACL
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：以数据专家行为描述并预测数据混合结果。
本库判断：专家结构参照不等于简单权重之外必然有新信息。

## phase
[Data Mixing Can Induce Phase Transitions in Knowledge Acquisition](https://arxiv.org/abs/2505.18091) · 2025 · paper
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：研究混合比例与模型容量变化下的知识获取阈值。
本库判断：不能从合成知识实验推断推荐兴趣一定发生相变。

## doremi
[DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429) · 2023 · NeurIPS
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：用代理模型和 minimax 域重加权优化预训练数据混合。
本库判断：minimax 配比、代理模型外推都不是新提法。
已定位的项目：[仓库](https://github.com/sangmichaelxie/doremi)；未在本次运行其完整基线，license/commit 必须在正式实验前记录。

## mgda
[Multi-Task Learning as Multi-Objective Optimization](https://proceedings.neurips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html) · 2018 · NeurIPS
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：将多任务学习显式视为多目标梯度优化。
本库判断：共同下降/Pareto 驻点属于已有理论，本库 LP 定理不包装成新优化理论。

## pie
[PIE: Personalized Interest Exploration for Large-Scale Recommender Systems](https://arxiv.org/html/2304.06844v1) · 2023 · WWW source
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：PIE 结合个性化图搜索、新创作者发现与在线探索机制。
本库判断：PageRank 加 bandit 进行兴趣探索已经出现，不能作为新颖性的主张。

## frec
[Modeling User Fatigue for Sequential Recommendation](https://arxiv.org/html/2405.11764v2) · 2024 · SIGIR
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：FRec 显式建模推荐中的用户疲劳。
本库判断：重复曝光导致疲劳本身不是新发现；日志停止与真实疲劳需区分。

## atspeed
[Efficient Inference for Large Language Model-based Generative Recommendation](https://arxiv.org/abs/2410.05165) · 2024/2025 · ICLR
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：研究 LLM 推荐的投机推理与候选验证。
本库判断：严格/放宽验证和 Top-K 任务不能偷换成单次精确抽样。
已定位的项目：[仓库](https://github.com/Linxyhaha/AtSpeed)；未在本次运行其完整基线，license/commit 必须在正式实验前记录。

## specgr
[Inductive Generative Recommendation via Retrieval-based Speculation](https://arxiv.org/html/2410.02939v1) · 2024/2026 · AAAI
阅读层级：L0+ selected HTML methods inspected; not full reproduction。
论文/官方材料支持：用具有归纳能力的候选提议器与生成式验证器支持新物品。
本库判断：draft-verify 还可能改善冷启动；并非仅计算加速的空白。
已定位的项目：[仓库](https://github.com/Jamesding000/SpecGR)；未在本次运行其完整基线，license/commit 必须在正式实验前记录。

## specdecode
[Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html) · 2023 · ICML
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：通过草稿提议、接受和残差纠正保持目标生成分布。
本库判断：接受纠正恒等式是已有算法基础，不是本库原创。

## kuairand
[KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos](https://kuairand.com/) · 2022 · CIKM/data
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：含标准和随机曝光、多种反馈及场景字段；另有文本补充文件。
本库判断：Pure 序列不完整；click 依 UI 有不同定义；随机标记不等于每个事件已知 propensity。

## kuaisar
[KuaiSAR: A Unified Search And Recommendation Dataset](https://kuaisar.github.io/) · 2023 · dataset
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：同用户的搜索与推荐行为提供跨服务观察。
本库判断：所选用户在观察期内同时使用两类服务，不能代表全部用户。

## kuaisim
[KuaiSim: A Comprehensive Simulator for Recommender Systems](https://arxiv.org/abs/2309.12645) · 2023 · NeurIPS
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：提供多反馈与会话相关的推荐模拟环境。
本库判断：模拟器上的策略收益不是新真实用户上的因果效果。
已定位的项目：[仓库](https://github.com/Applied-Machine-Learning-Lab/KuaiSim)；未在本次运行其完整基线，license/commit 必须在正式实验前记录。

## amazon23
[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) · 2023/2024 · dataset
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：商品评论与物品元数据可用于文本和序列推荐研究。
本库判断：评论数据不是完整曝光和未点击集合，不能据此证明 CTR 或留存。

## dr
[Doubly Robust Policy Evaluation and Learning](https://arxiv.org/html/1103.4601v2) · 2011 · ICML
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：结合奖励模型与日志策略信息进行 contextual-bandit 评估。
本库判断：双重稳健不消除 positivity/MAR 要求，不自动解决长序列 OPE。

## datafinder
[DataFinder: Scientific Dataset Recommendation from Natural Language Descriptions](https://aclanthology.org/2023.acl-long.573/) · 2023 · ACL
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：从自然语言研究描述检索科学数据集，并考虑研究约束。
本库判断：研究描述加约束推荐数据集已经存在；新贡献需落在可执行配对及独立证据。

## scirepeval
[SciRepEval: A Multi-Format Benchmark for Scientific Document Representations](https://arxiv.org/html/2211.13308v4) · 2022/2023 · paper/benchmark
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：科学文献表征的多格式评估，并与 SPECTER2 相关。
本库判断：文本相近不等于方法、数据和设备相容；不同版本任务计数不能混用。

## specter
[SPECTER: Document-level Representation Learning using Citation-informed Transformers](https://aclanthology.org/2020.acl-main.207/) · 2020 · ACL
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：利用引用信息学习科学文献表征。
本库判断：引用反映研究社区行为，不能直接当成方法可复现性或实验成功标签。

## precursor
[Precursor recommendation for inorganic synthesis by machine learning materials similarity from scientific literature](https://arxiv.org/html/2302.02303v2) · 2023 · Science Advances
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：从文献合成记录和材料相似性进行前驱体推荐。
本库判断：学习已发表选择习惯并不等于识别最优路线或所有失败机制。

## failed-experiments
[Machine-learning-assisted materials discovery using failed experiments](https://doi.org/10.1038/nature17439) · 2016 · Nature
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：失败实验可成为材料发现的学习信息。
本库判断：加入负实验或主动学习不是新贡献，应审计失败定义和实验条件。

## synthesis-reflection
[A critical reflection on attempts to machine-learn materials synthesis](https://doi.org/10.1039/D4FD00112E) · 2025 · Faraday Discussions
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：反思从已有合成数据推断新合成决策的局限。
本库判断：科学数据生成和报告选择过程必须与模型目标分开。

## www2027
[ACM Web Conference 2027 Research Track Papers](https://www2027.thewebconf.org/research-track-papers/) · 2026 · official CFP
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：征稿要求明确说明 Web 相关性。
本库判断：仅在网上获取的数据集上训练不自动满足领域相关性。

## greats
[GREATS: Online Selection of High-Quality Data for LLM Training in Every Iteration](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ed165f2ff227cf36c7e3ef88957dadd9-Abstract-Conference.html) · 2024 · NeurIPS
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：通过Taylor近似与greedy进行每轮在线batch选择。
本库判断：数据交互、二阶Taylor批次选择已有工作，不能作为新颖性的全部依据。

## gradmatch
[GRAD-MATCH: Gradient Matching based Data Subset Selection for Efficient Deep Model Training](https://arxiv.org/abs/2103.00123) · 2021 · ICML
阅读层级：L0 primary page/abstract/documentation inspected。
论文/官方材料支持：通过匹配训练或验证梯度与OMP选择子集。
本库判断：匹配验证梯度也属于已有梯度选择；本项目不以保留原训练更新为目标。
已定位的项目：[仓库](https://github.com/decile-team/cords)；未在本次运行其完整基线，license/commit 必须在正式实验前记录。

## globe
[GLOBE: Trajectory-Aligned Gradient Matching with Structured SparseOptimization for Coreset Selection](https://arxiv.org/abs/2608.02690) · 2026 · preprint; search-index abstract only
阅读层级：L0 primary arXiv abstract in search index only; direct page fetch failed。
论文/官方材料支持：检索到的一手arXiv摘要描述多checkpoint梯度轨迹和高阶匹配。
本库判断：全文直接打开失败，仅作为新近碰撞线索；不能声称已完成全文审计。
