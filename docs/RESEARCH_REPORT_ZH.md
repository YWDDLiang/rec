# 面向多目标LLM4Rec的数据选择：从单点评分到互补学习机会

**研究日期：2026-09-05。** 本报告与实施仓库配套；由用户PPT、Research仓库方法论、一手网络资料以及本次实际编写/运行的代码构成。来源与推断分开，真实效果不由理论代替。

## 一、最终判断：有更强的方向，但不能靠改名变新

我不再建议把任务定义为“挑一个更小的数据集，保留全数据能力”。用户关心的是模型能否因为数据选择**更会推荐**，尤其在多目标、多场景与个体差异下。这需要改变优化目标和验证方式，而不是把压缩改名为能力提升。

推荐主问题：

> 给定多个真实反馈目标，如何识别单独看不出价值、但共同训练或在合适阶段训练才能带来联合改善的数据；当现有日志不包含这种学习机会时，怎样判断继续调权无效，并获取有独立证据的互补反馈？

这个问题可形成两个候选贡献：**带符号、目标相关的互补学习诊断**；**由诊断指导的成组训练/证据获取，而非固定单点评分或同向去重**。时序传播是更强但风险更高的扩展。它们是否足够新颖、是否在真实推荐有效，必须由直接相关工作和危险基线判定，本库不宣称已获得WWW结果。

## 二、为什么原来的atom去重故事仍然太弱

[DEALRec](papers/dealrec.md)已为LLM推荐选择高价值微调数据，[GORACS](papers/goracs.md)已做组级梯度/OT选择，[MoRec](papers/morec.md)已从采样组织多目标推荐，[MiniRec](papers/minirec.md)已在RL中结合奖励、代表性、多样性与课程。通用领域的[Filter-then-Weight](papers/filter-weight.md)、[GREATS](papers/greats.md)、[GRAD-MATCH](papers/gradmatch.md)也覆盖目标导向更新、批次交互和梯度匹配。

所以三种表述不能直接通过：首次将梯度用于推荐；首次将数据选择用于多目标；首次将SFT选择搬到RL。继续加PageRank或Agent只会扩大工程，并不自然填补这些空缺。

真正可追问的是：**“有用性”能否分解成稳定的单样本总分？** 当目标相互补偿、训练阶段变化、观测过程不同，答案未必。它给出明确可证伪的机制，而不只是一个新模块名。

## 三、核心数学：互补不是重复

固定θ，设目标梯度v_m，训练更新g_i，带符号响应A_mi=v_mᵀg_i/s_m。不是要求选中数据重构原平均梯度，而是求

\[
t^*(A)=\max_{q\in\Delta}\min_m(Aq)_m
       =\min_{\lambda\in\Delta}\max_i\lambda^\top A_{:i}.
\]

它是局部诊断和配置，不是全局Pareto最优证明。若参数子空间中存在共同方向而该有限数据凸包中没有，则“多调几次配比”不能创造缺失方向。

考虑零更新旧池和两条新响应 `(2,−1)`、`(−1,2)`。单独加入任一条都不能实现共同改善，组合则实现 `(0.5,0.5)`。因此价值可以是非加性且非次模的；把每条样本预先评分、每个atom保留一个，很可能漏掉这种组合。

设旧最优对偶面Λ*，批次判据

\[
h_A(B)=\min_{\lambda\in\Lambda^*}\max_{b\in B}\lambda^\top b-t^*(A).
\]

在固定精确响应、固定旧可行域下，`加入B严格提高联合margin ⇔ h_A(B)>0`。只看一个最优λ会在退化情形误报。证明、数值差分和实际二次更新反例见[IDEA01](ideas/01_joint_learning_and_complementarity.md)。这些是标准LP对偶/敏感性推导，本库不冒充原创通用优化定理；论文贡献要来自它揭示并解决的推荐数据问题。

## 四、更进一步的改进点与取舍

**从相似性改为带符号互补。** 同向数据多不一定坏，反向数据也不一定应删；需保留完整目标向量而不是过早相加。[01](ideas/01_joint_learning_and_complementarity.md)

**从当前点扩展到学习阶段。** 前置表征数据可能当前不对齐，却改变之后的可学习性；二次伴随和顺序括号可精确验证。但时序归因已有工作，需要真实相同多重集换序干预，而不是“课程学习”命名。[03](ideas/03_trajectory_complementarity.md)

**从继续配重扩展到反馈获取。** 旧池不能提供方向时，请求互补数据，而不是永久提高相同样本频率。未标注响应必须预测和付费验证；免费使用真实未标注梯度是oracle泄漏。[06](ideas/06_verified_batch_acquisition.md)

**从线性分数增加曲率和误差检查。** 大正梯度可能导致overshoot；本库新增smoothness下界优化器，已知二次环境可避免该反例，未知人口梯度/曲率时不能认证。[01第9节](ideas/01_joint_learning_and_complementarity.md)

**atom只保留被证明不可替代的部分。** 指定因素投影可以剥离模板/前缀等线性影响，但它既非因果兴趣识别，也不创造额外信息。本次atoms+精确定价与直接LP结果相同，速度更慢，因此不预先把它写成大贡献。[02](ideas/02_personalized_atoms.md)

**观测与奖励先于选择。** 标签缺失、点击语义和曝光支持出问题，选择器可能只是放大观测偏差。[04](ideas/04_observation_aware_valuation.md)

**RL分清目标改变和计算分配。** 固定p时按q抽完整组并用p/q纠正，vector二阶矩配置具有明确数学对象；它不是把组内低分rollout删掉。[05](ideas/05_vector_rl_allocation.md)

## 五、原始随机游走与投机解码想法：完整保留，但重新定义

这部分没有被丢弃，而是拆成[07](ideas/07_cross_scene_exploration.md)与[08](ideas/08_policy_consistent_decoding.md)。PageRank解决候选图分布；兴趣覆盖解决“还有没有新类别”；停止模型解决会话继续概率；策略决定是否探索；投机验证解决计算而不是自动改变兴趣分布。

[PIE](papers/pie.md)已做个性化兴趣探索，[FRec](papers/frec.md)已建模疲劳，[AtSpeed](papers/atspeed.md)/[SpecGR](papers/specgr.md)已经覆盖推荐投机路线。可研究的差异需落在可识别的状态、独立反馈和目标一致执行上。当前代码可证明有限目录概率一致，但没有真实留存或解码加速证据。独立把它写成主线，需要另一个实验体系，不建议强塞入数据选择论文。

## 六、Science背景怎样形成真正的差异

[09科学组合推荐](ideas/09_scientific_bundle_recommendation.md)更适合WWW：面向研究用户推荐可执行的方法–数据集组合，而非只有主题相关的论文。DataFinder已考虑研究描述和约束，因此新的信息必须是接口兼容、资源可行与独立执行证据。单独相关的method和dataset可能不能一起工作，正好形成另一种可验证互补。

[10实验/前驱体推荐](ideas/10_experiment_and_precursor_recommendation.md)更贴近AI4Science：同时考虑成功概率、信息价值、成本和不同失败类型。已有前驱体推荐和失败实验学习，Gaussian信息增益原型不是物理验证。除非明确服务科学信息用户和Web决策过程，否则材料优化本身不自动符合WWW。

用户能贡献的主要资源是科学任务定义、验证条件和对失败原因的判断，不能在没有确认的情况下假定现成专家标注集、完整实验日志或特定GPU数。

## 七、执行框架与实际完成范围

仓库有10方向数学内核、数据schema、时间/缺失验证、本地KuaiRand CSV适配、共享tiny/HF反馈训练接口、完整组分类RL、概率/科学决策核验、来源索引和审计。每条idea文档列出函数、假设、反例和未完成部分。所有命令见[README](../README.md)与[IMPLEMENTATION](IMPLEMENTATION.md)。

这不是完整工业推荐栈：未提供线上服务、百万样本分布式逐例梯度、真实HF-GRPO引擎、未标注响应预测器、真实退出模型估计、GPU speculative tree verifier或科学执行沙箱。这些缺口显式保留，而不是用pass/TODO空函数冒充实施。

## 八、本轮实验结果改变了什么判断

实际跑了多种子tiny SFT、完整组分类RL、数学机制和二次互补干预。结果不是“所有方向成功”：SFT均值下uniform的两个BCE均低于frontier，atoms版本与frontier相同。简单LP对齐小参考集不够支撑泛化结论。这使本库把atoms从主贡献降为可选，把真实标签、参考误差、阶段互补和危险基线提升为下一轮的关键。

反过来，精确例子确实证明存在单例无益、组合有益，以及单一dual witness不充分、有限步overshoot、未知支持不能OPE等情况。这些支持研究问题的逻辑可能性，不支持真实推荐效果。具体数值见[VALIDATION](VALIDATION.md)。

## 九、WWW论文成立的最小闭环

研究问题必须明确对应多反馈推荐中的用户异质性、目标冲突与学习机会，而不是网上数据上的通用算子。[WWW 2027研究征稿](papers/www2027.md)要求清楚说明Web相关性。

较强的论文应闭环：真实日志存在可重复的互补现象 → 静态总分/去重与简单配比无法处理 → 本方法预测哪些组合/时段有帮助 → 独立干预证实而非只有benchmark均值 → 在严格预算和真实多目标上保留优势。

不要求十条idea全放入论文。建议先冻结01+06的问题表述；03只有通过换序实验才进方法；02只有比简单提名更强才进主贡献；04始终做正确；05待SFT机制成立再扩；07/08另行研究；09作为可独立验真的差异化应用。每次失败都应改变claim，而不是只换标题。

## 十、完整阅读路径

[LLM4Rec进展与基座](landscape/01_llm4rec_progress_and_backbones.md) → [数据与可辨识性](landscape/02_data_tasks_and_identifiability.md) → [10方向](ideas/README.md) → [数学索引](proofs/README.md) → [碰撞矩阵](matrices/novelty_collision_matrix.md) → [证据账本](matrices/claim_evidence_ledger.md) → [实验闸门](landscape/04_experiment_gates.md) → [来源/导师材料核验](landscape/03_source_and_mentor_audit.md) → [代码与审计](AUDIT.md)。
