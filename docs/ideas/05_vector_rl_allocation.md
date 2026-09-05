# IDEA 05 — 面向多目标完整 rollout 组的 RL 预算分配

**定位：SFT机制成立后可扩展；当前只有完整反馈的合成分类策略实验，不是 LLM-GRPO 复现。**

## 1. 文献空间与问题定义

[MiniRec](../papers/minirec.md) 已把奖励、梯度代表性、多样性、课程用于LLM推荐的RL选样；[I-PPO](../papers/ippo.md) 已研究不利episode的影响过滤；[ReCast](../papers/recast.md) 说明学习信号设计本身也会影响生成式推荐。不能把“将SFT去重搬到RL”当贡献。

这里要区分：改变训练人群目标，还是在**保持目标人群分布p**的前提下更有效地分配完整组rollout预算。本方向先选择后者，使数学估计对象清楚。每个prompt是用户历史、当前场景及合法候选；一次抽样获取完整G个rollout，不在算组内基线后偷偷删除样本。

## 2. 无偏性所对应的准确对象

设 X_im 是在prompt i上，由完整组得到的第m目标梯度估计，E[X_im]=μ_im。按q抽prompt，则

\[
\hat g_m=\frac1B\sum_{b=1}^B\frac{p_{I_b}}{q_{I_b}}X_{I_bm},
\quad E[\hat g_m]=\sum_i p_i\mu_{im}.
\]

这是对**X定义的组级surrogate**无偏。若X含GRPO组标准差归一化、PPO clipping、KL项，它不自动等于原始期望奖励的梯度；目标必须如实写出。重要性权重不自归一化，否则有限样本一般有偏。q_i≥εp_i保证支持，且p/q≤1/ε。

## 3. 方差配置的数学依据

令 a_im=E||X_im||²（预先固定目标尺度），单次估计二阶矩为

\[
M_m(q)=\sum_i\frac{p_i^2a_{im}}{q_i}.
\]

因均值固定，方差等于该量减去固定||Σpμ||²。单目标无下限时由拉格朗日或Cauchy–Schwarz得

\[
q_i^*\propto p_i\sqrt{a_i}.
\]

多目标解 `min_{q∈Δ,q≥εp} max_m M_m(q)`，各1/q项凸，因此为凸epigraph问题。对任何非负且和为1的目标配比λ，`||Σλ_m X_im||²≤Σλ_m||X_im||²`；控制每个M_m同时给出混合目标二阶矩上界，但不声称最优控制全部协方差。

独立分层均值估计且每层成本c_i时，连续预算问题给出 `n_i∝p_i√(variance_i/c_i)`。它不是对整数分组、同一prompt重复组相关性或实时设备吞吐的完整解。

## 4. 为什么完整组很重要

对IID actions和逐动作reward，leave-one-out基线 `b_-j=(Σ_{k≠j}r_k)/(G−1)` 与第j动作独立。score-function恒等 `E[∇logπ(a_j)]=0`，故 `E[(r_j−b_-j)∇logπ(a_j)]` 等于原奖励梯度。

若用含自己的组均值，未归一化估计带(G−1)/G系数；再除随机组标准差，变化更复杂。若奖励依赖整个推荐列表，LOO独立性也未必成立。因此代码区分LOO和GRPO，不将两者证明混用。组内删除高/低奖励候选会改变基线和标准差，必须另建估计器。

## 5. 如何接到推荐与更多提升点

按目标保留完整reward向量，而非先标量化再把同分组当冗余。优先对低成本probe估计a_im，并用探索floor防止困难群体永久失去rollout。估计要随策略更新，不可永久缓存旧policy的梯度矩；预算必须包含pilot、刷新和已丢弃的rollout。

可与01区分职责：01有意改变训练分配改善目标；本方向的p/q纠正保持固定目标，降低估计方差。若混用，则应把第一层定义为目标课程p_t，第二层为计算分配q_t，不要一边改p一边声称整个训练目标不变。

## 6. 领域评估与反证

先比较均匀、单标量rewardvariance、每目标minimax、MiniRec风格组选择；固定总rollouttoken、有效完整组数量、参考policy和reward来源。离线命中奖励不是长期满意度；如果候选没有真实反馈，不能将未知当0构造“完整reward矩阵”。

主要区分实验是在已知完整reward的小环境中核对梯度估计均值和二阶矩，再在真实日志支持的任务上看目标测试收益。若二阶矩下降但最终收益不增，不能越级宣称更有效学习；若pilot成本吃掉节省，效率主张失败。

## 7. 实现和状态

`allocation.py` 提供闭式和minimax配置，`rl.py` 提供完整组优势、p/q修正、surrogate和KL；`experiments/run_tiny_rl.py` 运行3种子、两种配置。该实验用单rollout score二阶矩作为组级预算proxy，明确不是精确完整组方差oracle。真实LLM rollout引擎、奖励服务、分布式GRPO尚未接通。本方向数学内核通过，真实效果未通过；固定矩阵上二阶矩由2.9降至约2.536不是推荐指标提升。
