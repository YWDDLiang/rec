# 一手来源索引

核验日2026-09-09。共25项外部一手来源，另有用户PDF。阅读深度逐项记录；不是25篇全文复现报告。

## S01 — LC-Rec: Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation

年份：2023/2024；类别：任务环境；深度：摘要与HTML方法部分；独立复现：否。

来源：https://arxiv.org/abs/2311.09049

补充一手来源：https://arxiv.org/html/2311.09049v4

多角色索引/语言/偏好对齐；部分监督是生成标签。原模型不能与迁移Qwen混称严格复现。

## S02 — MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation

年份：2025；类别：任务环境；深度：一手摘要；独立复现：否。

来源：https://arxiv.org/abs/2510.24431

公开SID→SFT→RL框架；本轮不复现其完整训练与结果。

## S03 — Aligning Large Language Models for Controllable Recommendations / RecLM-gen

年份：2024；类别：任务环境；深度：摘要与官方README；独立复现：否。

来源：https://arxiv.org/abs/2403.05063

补充一手来源：https://github.com/microsoft/RecAI/tree/main/RecLM-gen

推荐与条件控制；冻结动态生成器，区别teacher监督和观测真值。

## S04 — Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5)

年份：2022；类别：任务环境；深度：一手摘要；独立复现：否。

来源：https://arxiv.org/abs/2203.13366

推荐相关任务统一文本框架；encoder-decoder不能直接用本包causal loss。

## S05 — OpenP5 official repository

年份：2024；类别：工程环境；深度：官方README；独立复现：否。

来源：https://github.com/agiresearch/OpenP5

T5/LLaMA开发环境；公开主要两类推荐任务，不自动等于P5所有任务。

## S06 — Amazon Reviews 2023 dataset

年份：2023/2024；类别：数据；深度：官方字段与发布页；独立复现：否。

来源：https://amazon-reviews-2023.github.io/

规模丰富、有文本/时间戳；评论不是完整曝光，历史可用性需核验。

## S07 — KuaiSAR: A Unified Search And Recommendation Dataset

年份：2023；类别：数据；深度：官方字段与统计；独立复现：否。

来源：https://kuaisar.github.io/

补充一手来源：https://kuaisar.github.io/detailed_statistics.html

共享用户搜索/推荐；关键词及caption匿名，部分标签有条件定义。

## S08 — KuaiRand dataset

年份：2022+；类别：数据；深度：官方发布页；独立复现：否。

来源：https://kuairand.com/

含随机曝光部分和多种反馈；不将缺失反馈直接转负例。

## S09 — Data-efficient Fine-tuning for LLM-based Recommendation (DEALRec)

年份：2024；类别：选择基线；深度：一手摘要；独立复现：否。

来源：https://arxiv.org/abs/2401.17197

推荐SFT的数据选择，代理影响力与学习难度；需要正式复现。

## S10 — GORACS: Group-level Optimal Transport-guided Coreset Selection for LLM-based Recommender Systems

年份：2025；类别：选择基线；深度：一手摘要；独立复现：否。

来源：https://arxiv.org/abs/2506.04015

组级与OT/梯度选择已有研究；组结构不是我们的独占新颖性。

## S11 — LESS: Selecting Influential Data for Targeted Instruction Tuning

年份：2024；类别：选择基线；深度：正式出版页与摘要；独立复现：否。

来源：https://proceedings.mlr.press/v235/xia24c.html

低秩梯度、目标导向选择；简单cosine不可冒充LESS。

## S12 — GRAD-MATCH: Gradient Matching based Data Subset Selection for Efficient Deep Model Training

年份：2021；类别：选择基线；深度：一手摘要；独立复现：否。

来源：https://arxiv.org/abs/2103.00123

梯度匹配与OMP已有框架；不把实现OMP当新贡献。

## S13 — A Data-Centric Multi-Objective Learning Framework for Responsible Recommendation Systems (MoRec)

年份：2023/2024；类别：混合基线；深度：一手摘要；独立复现：否。

来源：https://arxiv.org/abs/2310.13260

数据驱动多目标推荐已有方法；来源配比需要强对照。

## S14 — Rethinking Data Mixing from the Perspective of Large Language Models (DoGraph)

年份：2026；类别：混合基线；深度：一手摘要；独立复现：否。

来源：https://arxiv.org/abs/2604.07963

从梯度角度讨论domain和mixing；不能称首次从模型结构定义数据单元。

## S15 — Filter-then-Weight: Online Data Selection and Reweighting for LLM Fine-Tuning

年份：2026；类别：在线基线；深度：一手摘要；独立复现：否。

来源：https://arxiv.org/abs/2604.00001

优化器感知在线更新匹配；本轮保持离线定位，不偷换问题。

## S16 — MiniRec: Data-Efficient Reinforcement Learning for LLM-based Recommendation

年份：2026；类别：RL基线；深度：一手摘要；独立复现：否。

来源：https://arxiv.org/abs/2602.04278

RL的reward/gradient/diversity选择；不是现阶段SFT直接结果。

## S17 — Gradient Atoms: Unsupervised Discovery, Attribution and Steering of Model Behaviors via Sparse Decomposition of Training Gradients

年份：2026；类别：表示基础；深度：一手摘要；独立复现：否。

来源：https://arxiv.org/abs/2603.14665

共享更新方向稀疏分解已有研究；PDF简化signed-hash方案不是全文所有实现。

## S18 — Beyond neural scaling laws: beating power law scaling via data pruning

年份：2022；类别：理论先例；深度：一手摘要；独立复现：否。

来源：https://arxiv.org/abs/2206.14486

裁剪与数据规模区域已有先例；不同任务的成功不能直接外推推荐。

## S19 — Scaling Data-Constrained Language Models

年份：2023；类别：数据区域；深度：一手摘要；独立复现：否。

来源：https://arxiv.org/abs/2305.16264

数据受限时重复和算力分配值得研究；不能直接给推荐万能保留率。

## S20 — SemDeDup: Data-efficient learning at web-scale through semantic deduplication

年份：2023；类别：语义基线；深度：一手摘要；独立复现：否。

来源：https://arxiv.org/abs/2303.09540

语义近重复对照；文本相似不等于监督作用相同。

## S21 — D4: Improving LLM Pretraining via Document De-Duplication and Diversification

年份：2023；类别：语义基线；深度：一手摘要；独立复现：否。

来源：https://arxiv.org/abs/2308.12284

去重＋多样化已有研究；不能把这两个词的组合当新颖性。

## S22 — Qwen3-1.7B official model card

年份：2025；类别：模型；深度：官方模型卡；独立复现：否。

来源：https://huggingface.co/Qwen/Qwen3-1.7B

一个可用因果LM骨干；不声称当前最强推荐模型。

## S23 — Llama-3.2-3B-Instruct official model card

年份：2024；类别：模型；深度：官方模型卡；独立复现：否。

来源：https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

跨家族验证；本地权重与授权由用户准备。

## S24 — Recommender Systems with Generative Retrieval (TIGER)

年份：2023；类别：模型基线；深度：一手摘要；独立复现：否。

来源：https://arxiv.org/abs/2305.05065

生成式检索并不自动等于大规模语言预训练。

## S25 — Self-Attentive Sequential Recommendation (SASRec)

年份：2018；类别：模型基线；深度：一手摘要；独立复现：否。

来源：https://arxiv.org/abs/1808.09781

传统序列推荐强对照；teacher与独立评测的职责需分开。

## S26 — Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping

年份：2025；类别：learned-domain / data mixture；深度：OpenReview论文 + NVIDIA公开recipe；独立复现：否。

来源：
- https://openreview.net/pdf?id=aBlqKPkc4a
- https://docs.nvidia.com/nemo/curator/curate-text/tutorials/nemotron-climb

要点：语义embedding聚类得到learned domains，训练/评估大量proxy mixtures，再拟合predictor迭代搜索最优mixture。它不是简单semantic dedup。对本项目最重要的限制是：**learned domain本身不是新颖点。**

## S27 — Optimizing Pre-Training Data Mixtures with Mixtures of Data Expert Models

年份：2025；类别：structured mixture reference；深度：ACL正式页面/摘要；独立复现：否。

来源：https://aclanthology.org/2025.acl-long.1564/

要点：用domain experts的预测构造mixture行为特征，帮助预测candidate mixture loss。说明raw mixture weights可能不足以表示数据作用。

## S28 — AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs

年份：2025；类别：adaptive multi-task mixture；深度：ACL正式页面/摘要；独立复现：否。

来源：https://aclanthology.org/2025.acl-long.990/

要点：先训练task specialists作为强性能参照，再根据generalist相对specialist的excess loss动态reweight或resample。对我们的直接启发是：**rare data不等于rare valuable capability，应围绕独立能力缺口定义保护。**

## S29 — Data Mixing Can Induce Phase Transitions in Knowledge Acquisition

年份：2025；类别：mixture regime / scaling；深度：arXiv摘要；独立复现：否。

来源：https://arxiv.org/abs/2505.18091

要点：在受控biography+web mixture中，知识获取对model size与mixing ratio可能出现临界行为。不能直接外推LLM4Rec，但要求我们不要假设selection/mixture效果随模型和比例平滑变化。

## S30 — Dynamic Data Mixing Maximizes Instruction Tuning for Mixture-of-Experts

年份：2025；类别：dynamic instruction mixture；深度：NAACL正式页面/摘要；独立复现：否。

来源：https://aclanthology.org/2025.naacl-long.80/

要点：在有限训练预算下，根据任务/数据集间冗余动态调整instruction-tuning mixture。其场景为MoE，不是推荐；因此“limited-budget + redundancy-aware dynamic mixture”不能单独作为我们的新颖性。

## S31 — Sketching the Readout of Large Language Models for Scalable Data Attribution and Valuation (RISE)

年份：2026；类别：scalable attribution；深度：arXiv摘要；独立复现：否。

来源：https://arxiv.org/abs/2604.16197

要点：聚焦输出层的influence hotspots，并对lexical residual / semantic projected-error进行sketch，降低大模型attribution成本。它是Idea 02成本故事的重要强替代路线：不能只说“512维projected gradient存储小”就声称大规模选择成本已经解决。
