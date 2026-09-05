# Data Mixing Can Induce Phase Transitions in Knowledge Acquisition

**证据深度：L0 primary page/abstract/documentation inspected。不是 L1 全文逐项完成，不是 L2 代码审计或 L3 独立复现。**

一手入口：[phase](https://arxiv.org/abs/2505.18091)；最后核验 2026-09-05。

## 来源支持的事实

研究混合比例与模型容量变化下的知识获取阈值。

## 对本项目的直接约束

不能从合成知识实验推断推荐兴趣一定发生相变。

## 十二项审计中的当前覆盖与缺口

| 项目 | 本轮状态 |
|---|---|
| Scientific problem | 已按来源摘要和问题定义进行定位，属于 `mixture` 问题簇。 |
| Mathematical task | 可迁移结构在对应 idea 文档中推导；未将本库公式冒充该文原公式。 |
| Data-generating process | 正式复现前须逐项记录该文数据版本、日志来源和过滤。当前未独立核验原始数据。 |
| Core mechanism | 只认可上方短事实所述机制，不因模型命名推断额外能力。 |
| Claim–evidence alignment | 作者结果与本项目结果分开；本库未复现该文数值。 |
| Hidden assumptions | 针对迁移后的 assumptions 在方向文档列出，原文全部假设未完成穷尽审计。 |
| Strongest alternative | 对本项目而言，先检验更简单的同预算重加权、采样或检索能否解释增益。 |
| Missing baseline | 该文作为 related work / baseline candidate；不把论文未列某基线直接当成其方法无效。 |
| Killer experiment | 见各 idea 的匹配预算干预，不能仅凭相关可视化判断。 |
| Contribution type | 不在 L0 阶段给主观总分；按问题、信息、算子、计算、证据、知识分别讨论。 |
| Transferable abstraction | 本库使用其问题启示或公开算法原语，不直接包含第三方代码。 |
| Final verdict | 可用于限定相关工作边界；不支持“本文方法已经优于该文”的 claim。 |

## 下一层核验要求

下载并固定全文版本、作者代码 commit、依赖版本、数据拆分和评价配置，运行关键基线后再将状态升级为 L1/L2/L3。不得通过改 metadata 自动升级证据等级。
