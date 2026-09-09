# Data-Rich LLM4Rec 数据选择与 Data Mixture 背景（2026-09-09）

本文将这一轮新增调研同步到 `rec`，作为 Idea 01 / 02 / 11 的共同背景。  
它不是某个 idea 的论证替代，而是用来限定新颖性、设计强 baseline 和确定实验边界。

---

## 1. 研究坐标：从“选哪些样本”到“怎样定义训练调配单元”

可以把相关工作整理成五层：

### Level 1 — 单样本 attribution / selection

关注：

\[
\text{sample}\rightarrow\text{scalar / target influence}.
\]

代表思路：

- influence / gradient matching；
- LESS；
- DEALRec；
- RISE 等 scalable attribution。

局限：单条分数难表示数据之间的互补、重叠和集合价值。

### Level 2 — 组合 / 行为单元

关注：

\[
\text{samples}
\rightarrow
\text{shared behavior / gradient unit}.
\]

代表：

- Gradient Atoms；
- GORACS group-level selection；
- ordinary gradient clustering。

Idea 02 位于这里，但不能因为“用了 group / atoms”就自动具有新颖性。

### Level 3 — learned domain

关注：

> mixing 的基本 unit 是否应由人工 source label 给定？

代表：

- Nemotron-CLIMB：embedding → clusters/domains → proxy mixture search；
- DoGraph：gradient structure → learned domains / reweighting。

这直接限制 Idea 02 的 claim：
**“用模型内部结构重新定义 domain”已经不是空白。**

### Level 4 — structured mixture optimization

关注：

> mixture weights 本身是否太粗？模型当前缺什么能力，怎样显式表示？

代表：

- Mixture of Data Experts：specialist behavior → mixture-loss prediction；
- AutoMixAlign：specialist losses → generalist excess loss → adaptive reweight/sample。

这直接启发 Idea 01：
训练分配应对齐 **capability gap / target reference**，而不是只看数据频率。

### Level 5 — regime / threshold

关注：

> 数据比例、模型尺度和能力学习之间是否始终平滑？

代表：

- Data Mixing Can Induce Phase Transitions in Knowledge Acquisition。

对 Idea 11 的启发：
不要假设一个 keep ratio / mixture recipe 在所有数据规模和模型规模上平滑有效。

---

## 2. Nemotron-CLIMB：learned semantic domains + proxy mixture search

CLIMB 的重要结构是：

```text
documents
 -> embedding
 -> KMeans semantic clusters
 -> prune / merge into domains
 -> sample candidate mixture weights
 -> train proxy models
 -> evaluate
 -> fit predictor
 -> iteratively search mixture
```

其 NVIDIA 公开 recipe 说明 paper-scale 搜索需要大量 proxy runs，目标是找到好的 pretraining mixture，而不是做简单 semantic dedup。

### 对我们意味着什么

不能说：

> “人工 domain 不好，所以我们首次学习 domain。”

可以说：

> “语义 learned domain 已经存在；我们研究的是 recommendation SFT 中 model-centric gradient units 是否比 semantic/source units 更能预测实际训练作用，并在固定 source exposure 下改善推荐能力。”

---

## 3. DoGraph：gradient space learned domain 已经出现

DoGraph 从 per-sample gradient 角度讨论 domain 与 mixing，并把数据调度写成 graph-constrained optimization。

### 对 Idea 02 的危险解释

如果：

- projected gradient KMeans；
- domain mean gradient；
- 简单 reweighting

就能达到相同结果，那么 dictionary atoms 只是复杂化。

因此必须把 **DoGraph-style gradient groups / KMeans** 作为强对照。

---

## 4. Mixture of Data Experts：mixture rate 不是充分特征

MDE 训练 domain specialists，用 specialist predictions 构造 candidate mixture 的行为特征，再用少量 proxy runs 拟合 mixture loss。

### 可迁移抽象

\[
\text{mixture}
\not\equiv
\text{raw weights only}.
\]

更合理的是：

\[
\text{mixture}
+
\text{behavioral reference}
\rightarrow
\text{performance}.
\]

对 Idea 01：
JLO / capability response 应当围绕当前目标 reference 定义。

---

## 5. AutoMixAlign：相对 specialist frontier 优化

AutoMixAlign 在多任务 preference optimization 中：

1. 为每个任务训练 specialist；
2. 用 specialist loss 代表相应任务的强性能参照；
3. generalist 动态优先当前 excess loss 最大的任务；
4. 提供 reweighting 与 resampling 两种算法。

### 对 LLM4Rec 的启发

我们不一定训练完整 specialist，但需要一个类似的思想：

> 保护任务不应只按“样本少”或“atom 小”，而应按独立 capability reference 判断模型是否真的缺它。

因此 rare atom ≠ rare valuable capability。

---

## 6. Data Mixing phase transition：为什么 Idea 11 必须做二维/三维 scaling

该工作在受控 biography + web mixture 上观察到：

- model size 存在临界区间；
- mixing ratio 存在临界区间；
- below threshold 时，增加训练量可能仍学不到相应知识；
- recipe 不一定能从小模型平滑迁移到大模型。

### 对本项目的严格边界

我们可以研究：

\[
\text{candidate richness}
\times
\text{train budget}
\times
\text{model size}
\]

是否存在 selection benefit 的明显 regime change。

但只有真实曲线出现跳变/临界关系时，才能使用 “phase transition” 术语。  
否则应写 “regime transition / saturation boundary”。

---

## 7. Dynamic Data Mixing for instruction tuning

NAACL 2025 的 Dynamic Data Mixing for MoE instruction tuning 也强调：

- task importance 会随训练状态变化；
- 固定 mixture 可能浪费有限预算；
- 数据集间 redundancy 可以驱动动态权重。

虽然其模型是 MoE，不是我们的 generative recommender，但它限制了一类过强 claim：

> “有限训练预算下按数据冗余动态 mixing”本身已经有先例。

我们的新颖性必须来自 **LLM4Rec 特有任务结构 + model-centric behavioral unit + capability-preserving evaluation**。

---

## 8. RISE：大模型 gradient attribution 的可扩展替代路线

RISE 主要在 readout 层做 influence sketching，拆成 lexical residual 和 semantic projected-error channel，以降低大模型 attribution 存储/计算。

### 对本项目的意义

Gradient atoms 的一个现实风险是 per-example full LoRA backward 成本太高。

所以成本实验必须至少比较：

- full LoRA gradient；
- readout / low-rank sketch；
- atom learned from cheap sketch + expensive confirm；
- total wall-clock / GPU-hour。

若最终必须对全部样本做 full gradient，不能把 512 维 projection 的存储减少写成整体 selection cost 已解决。

---

## 9. 01 / 02 / 11 的统一论文结构

### Idea 01 — Joint Learning Opportunity

研究：

\[
\text{coarse source mixing}
\quad\text{vs.}\quad
\text{fine-grained data action}.
\]

核心诊断：

\[
\Delta_{\text{JLO}}
=
t(\mathcal Q_{\text{fine}})
-
t(\mathcal Q_{\text{coarse}}).
\]

它回答：
**有没有必要做细粒度选择？**

### Idea 02 — Behavioral Gradient Atoms

研究：

\[
\text{what is a useful fine-grained mixing unit?}
\]

它回答：
**如果需要细粒度选择，什么结构比人工 source / semantic cluster / raw sample 更好？**

### Idea 11 — Budget-conditioned regime

研究：

\[
\text{when does selection matter?}
\]

它回答：
**在哪种候选池丰富度、训练预算与模型尺度下，01/02 的价值出现？**

---

## 10. 当前最强论文故事

不是：

> “我们用 gradient atoms 去重，所以训练更快。”

而是：

> Multi-role LLM recommendation creates supervision pools whose row count can grow much faster than the optimization budget. We first diagnose whether fine-grained learning opportunities exist beyond task/domain mixture control; then learn model-centric behavioral update units and allocate training exposure while protecting independently validated capabilities; finally characterize the data-rich regimes where this helps and the data-limited regimes where full-data training remains preferable.

中文：

> 多角色 LLM 推荐的监督池可以快速膨胀，但模型可消费的训练预算并不同比增长。我们先判断任务/场景配比之外是否真的还存在细粒度联合学习机会，再用模型更新结构学习调配单元，并保护经独立验证的稀有能力；最后系统刻画在什么数据丰富度与训练预算条件下选择有效、什么时候应当继续使用全量数据。

---

## 11. 关键 novelty collision matrix

| 我们可能想说的 claim | 已有工作会如何反驳 | 允许保留的更强版本 |
|---|---|---|
| 首次 learned domain | CLIMB / DoGraph | 在 multi-role LLM4Rec 中学习 model-centric update units，并证明其训练价值 |
| 首次 group-level data selection | GORACS / Gradient Atoms | full signed multi-atom units + capability-preserving fixed-budget allocation |
| 首次 data mixing for multi-objective | MoRec / AutoMixAlign | source mixing 之后的 within-source behavioral opportunity |
| 首次 target-gradient selection | LESS / GRAD-MATCH | 证明 atoms 在 effect estimation / stability / cost / rare capability 上不可替代 |
| 首次 limited-budget dynamic mixture | dynamic data mixing works | 推荐特有的多角色监督、合法 item generation、control satisfaction 与用户条件能力 |
| 首次发现数据规模不同策略不同 | pruning/scaling/phase-transition literature | 系统测量 LLM4Rec 的 candidate-richness × train-budget regime interaction |

---

## 12. 当前研究优先级

1. **Idea 01：先测 JLO gap 与独立 effect prediction。**
2. **Idea 02：只有 JLO gap 存在时，再比较 raw/KMeans/atoms。**
3. **Idea 11：用 nested pools + fixed train budget 做主论文 scaling 图。**
4. SFT 主线成立后才扩 RL。
5. 旧 `exp/01_02` 负结果继续冻结，作为研究判断演化证据。
