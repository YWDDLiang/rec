# 输入材料、导师框架与外部核验：哪些是来源，哪些是本库修订

## 一、直接输入

用户PPT `DataAttribution&Mixing.pptx` 共38页。提取文本及SHA见[来源清单](../provenance/input_manifest.json)与[逐页文本](../provenance/ppt_text_extraction.json)。没有将PPT全文截图或第三方论文/代码打包分发。

用户Research仓库主分支读取时为 `299bd85752ce1a59ea5cce8c1b145777e33cba6f`。读取README、CONTRIBUTING与目录，不改远端。导师框架强调问题、变量、假设、机制、证据、反例和边界；每条idea遵循这一结构。直接来源：[Research README](https://github.com/YWDDLiang/Research/blob/299bd85752ce1a59ea5cce8c1b145777e33cba6f/README.md)、[CONTRIBUTING](https://github.com/YWDDLiang/Research/blob/299bd85752ce1a59ea5cce8c1b145777e33cba6f/CONTRIBUTING.md)。

## 二、支持什么，不支持什么

PPT p16支持“用少量共享方向描述梯度组合”，不支持“atom去重必提高推荐”。用户报告的SFT现象是单独的经验输入，未提供数据、随机种子和日志；本库既不否认，也不把它改写为已复现证据。

PPT后段的轨迹干预/null hypothesis、learned domains、specialist reference和比例阈值，为本项目提出可证伪假设提供启示。但它们不是本项目真实推荐有效的证明。Data Mixture从source label转成learnedunit，已经有DoGraph/CLIMB；需要进一步解释新选择对象改变了什么。

## 三、外部核验后明确提出的修订

**RISE / PPO归属：** PPT有一页把rollout影响过滤置于“Sketching the Readout…”题名下。已读RISE HTML未检出PPO，而I-PPO论文明确研究PPO episode归因过滤，因此该页描述更对应[I-PPO](../papers/ippo.md)。这是外部核验判断，不静默重写原PPT，也不宣称已找到所有页面的原始制作链。

**NASH：** [NASH](../papers/nash.md)指Shapley-informative components的非线性聚合，不是Nash bargaining。不能因为名字而把其理论与多目标博弈解等同。

**specialist loss：** 实测专用模型损失是经验reference，不自动是总体最优的数学下界。把“generalist减specialist”定义为缺口可以，但不能因此断言缺口非负或该specialist全局最优。

**中间表示的信息论：** Research README用 `I(Z;Y|X)>0`表达中间变量带来新增信息。本库补充一个适用边界：若Z=f(X)且不额外访问反馈，条件互信息严格为0。这样的Z仍能通过归纳偏置、计算结构、优化条件改善有限模型学习，不能要求它创造输入之外的信息。外部专家/实验返回的新证据才可能使条件互信息增加。这是对该公式使用条件的本库分析，保留原仓库不动。

## 四、来源证据等级

49项记录覆盖LLM4Rec、SFT/RL选择、mixing、图探索、解码、科学推荐与数据。多数核验至一手摘要/页面；部分阅读HTML机制段落，标为L0+，不称L1全文审计。没有独立复现这些论文主结果，没有核验每个官方仓库commit与license。引用卡刻意保留未知字段；不会通过填写表格自动升级证据。

本库数学证明和自写代码是独立推导/实现；可运行并不意味着比论文强。未来必须锁定baseline版本和实际数据，再将claim从“可检验”升级为“在特定条件有效”。
