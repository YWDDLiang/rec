# LLM4Rec进展、通用基座与本研究应选的任务

检索截止2026-09-05；本页是主题级综合分析，不是声称逐篇完成全文/源码复现。一手来源和核验层级见[来源索引](../../references/README.md)。

## 一、先不要把LLM4Rec等同于一个SID模型

| 范式 | 学习对象与输出 | 数据选择关键变量 | 主要来源 |
|---|---|---|---|
| 语义增强推荐 | LLM表征/文本供传统推荐使用 | 文本质量、语义与协同信号是否一致 | [LC-Rec](../papers/lcrec.md)等相关语义对齐路线 |
| 共享LLM主干的排序/反馈预测 | 候选物品的点击/点赞/偏好等 | observation mask、多反馈目标、真实候选集 | [GenRec](../papers/genrec.md) |
| 自回归生成式检索 | 历史条件下生成ID/SID | SID学习、协同对齐、合法目录、同token多物品 | [TIGER](../papers/tiger.md)、[LC-Rec](../papers/lcrec.md) |
| 指令/交互推荐 | 历史+用户当前约束→推荐或解释 | instruction遵循与偏好是否矛盾、上下文切换 | [OpenOneRec](../papers/openonerec.md)、[CAPT](../papers/capt.md) |
| RL后训练 | 依reward更新候选策略 | 支持、完整组、reward构造、估计方差 | [MiniOneRec](../papers/minionerec.md)、[MiniRec](../papers/minirec.md)、[ReCast](../papers/recast.md) |

TIGER式离散生成可由随机初始化序列模型完成，不能仅因生成token称之为语言基础模型；而采用LLM主干的排序模型也不必生成物品token。模型输出范式与是否预训练是两个维度。自回归SID是可行载体，不是唯一“通用基座”。

2026年公开工作进一步拓展了基础模型/多任务和目录排序路径，GenRec的公开技术描述说明不应把所有LLM4Rec研究压到next-item ID生成；其结果是作者报告，不是本仓库线上实验。OpenOneRec的公开模型/benchmark适合评价指令和多领域，但多任务不自动等于真实多反馈效用。

## 二、最接近的选择/配比工作不能只列论文名

DEALRec已经回答“哪些推荐微调样本更有价值”；GORACS已将组级信息、梯度和OT结合；MoRec直接把多目标优化转成数据采样；MiniRec把学习难度、代表性、冗余和课程引入推荐RL。它们分别压缩了四种看似自然的增量空间：[DEALRec](../papers/dealrec.md)、[GORACS](../papers/goracs.md)、[MoRec](../papers/morec.md)、[MiniRec](../papers/minirec.md)。

通用LLM路线也不能忽略：LESS让低秩梯度选择可实施；Filter-then-Weight已经针对当前优化器与更新目标做在线过滤/配重；OGS关注梯度几何；Partition-matroid gradient matching还约束组配额。仅将“去重”改名“塑造梯度”“能力保护”无法避开这些工作。[LESS](../papers/less.md)、[Filter-then-Weight](../papers/filter-weight.md)、[OGS](../papers/ogs.md)、[Partition matching](../papers/partition-match.md)。

剩余问题需要更具体：**某些多反馈学习机会只能由相互补偿的数据共同提供，而现有日志是否包含这种机会，不能从单样本总分或固定atom频次判断。** 这是本库的假设与任务重写，不是通过检索就证明了世界上没有其他相似研究。

## 三、推荐基座选择，不让工程代替科学问题

优先真实多反馈问题时，建议先用共享小LLM主干+固定反馈头。优点是每个目标有清楚的损失和缺失掩码，能直接审计带符号梯度；缺点是语言预训练贡献可能被浅层统计模型解释。必须加随机初始化同架构、冻结LLM、传统序列推荐与文本打乱控制。

需要研究生成策略或SID token时，再接MiniOneRec/LC-Rec式生成框架。先固定ID tokenizer和目录映射，防止选择方法因SID先验或目标泄漏获利。MiniOneRec是有用实现基础，但本轮没有clone其完整第三方代码或复现它；正式对照需固定commit与修复记录。MiniRec是另一篇RL数据选择论文，两者不能混称。

OpenOneRec可用于公开模型上的适配和更丰富指令任务，但不能把更大模型、额外数据、专属预训练与数据选择收益混在一起。不建议为验证一个选择假设重新进行基础模型预训练。

## 四、规模与训练形式

本库不假定用户本次可用多少GPU。CPU原型验证之后，可以从本地可得的较小模型和LoRA子空间开始，先固定数据、预训练与目标头，再考虑规模外推。小模型上的recipe可能失效；[知识获取阈值](../papers/phase.md)提醒我们不能假设配比连续、平滑、自动随规模迁移，但不能据此宣称推荐也已出现相变。

实际交付HF载体是multi-feedback supervised ranker，不是已复现的MiniOneRec。默认gradient-filter=feedback_head仅用该子空间估计选样，实际LoRA也可训练，因此不得把子空间矩阵证书称作整个参数更新证书。ALL选项仍受内存guard和单进程限制。
