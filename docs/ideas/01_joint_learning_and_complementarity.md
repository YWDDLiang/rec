# IDEA 01 — 多角色联合学习机会：从“严格互补”到预算条件下的带符号训练作用

**当前定位：主机制候选。**  
Idea 01 不再要求所有有效结果都满足“样本 A 单独无效、B 单独无效、A+B 才有效”的严格互补故事。严格 A/B 互补保留为机制实验；论文层面的核心问题改为：

> 在多角色 LLM4Rec 的固定训练预算下，粗粒度 task/domain mixing 已经决定“每类数据占多少”之后，来源内部是否仍存在可利用的带符号训练作用差异？如果存在，哪些数据组合能创造 source-level mixing 看不到的联合学习机会？

这不是数据压缩，也不是“少数据一定更好”。当候选池小、有效数据都能得到充分训练暴露时，full-data 是强默认；当候选监督比可用训练 token/step 更丰富时，训练机会本身成为稀缺资源。

---

## 1. 与 Idea 02 / Idea 11 的分工

三个方向不互相替代：

- **Idea 01：What to optimize?**  
  定义并诊断“联合学习机会”：哪些训练数据对多个推荐/辅助能力的作用是正、负、互补还是不可判定；区分目标冲突、数据支持不足与估计误差。
- **Idea 02：What are the mixing units?**  
  用完整带符号 gradient atoms 把大量逐样本更新组织为可复用的 behavioral update units；它是表示/结构假设，必须与 raw gradient、KMeans 等强基线竞争。
- **Idea 11：When should selection matter?**  
  用候选池丰富度 × 固定训练预算建立 regime 研究，决定什么时候 full-data 是合理默认，什么时候细粒度训练暴露分配才值得启用。

因此，Idea 01 可以在不用 atoms 的 raw-gradient 版本上独立成立；Idea 02 只有在真实实验里提供额外作用时才保留为方法贡献；Idea 11 负责把二者放入真实的数据丰富度与训练预算坐标系。

---

## 2. 为什么旧的“严格互补”不足以作为整篇论文主线

旧版本强调响应

\[
b_1=(2,-1), \qquad b_2=(-1,2)
\]

单独不能共同改善，组合后可获得正的联合收益。这个反例仍然重要，因为它证明：

1. 单样本 scalar score 可能看不到集合价值；
2. 数据价值可以是非加性的；
3. 先独立打分再硬去重可能误删互补训练信号。

但真实推荐数据并不保证存在大量这种严格形式。若某一组样本对所有目标都普遍有益，选择它同样是合理结果；不能为了故事排除更简单的有效解。

所以严格互补降级为 **killer mechanism test**，不再作为“所有主实验必须出现”的论文前提。

---

## 3. 多角色 LLM4Rec 的数学任务

设候选训练样本为 \(i=1,\dots,N\)，目标/能力为 \(m=1,\dots,M\)。

对固定模型状态 \(\theta\)：

\[
v_m=\nabla_\theta L_m(\theta),
\qquad
g_i=\nabla_\theta \ell_i(\theta).
\]

定义带符号局部训练响应：

\[
A_{mi}
=
\frac{v_m^\top g_i}{s_m},
\qquad s_m>0.
\]

约定 \(A_{mi}>0\) 表示按 \(g_i\) 的下降方向更新在一阶上有利于目标 \(m\)。  
**不要在进入优化前对负值 ReLU。** 负作用是多目标冲突与补偿结构的一部分。

---

## 4. 先把“source mixing”与“source 内选择”严格分开

令人工来源桶

\[
b(i)=(\text{task},\text{domain},\text{source}).
\]

### 粗粒度 mixture 可行域

只允许改变来源质量 \(\lambda_b\)，桶内仍保持原条件分布：

\[
q_i=\lambda_{b(i)}p(i\mid b(i)),
\qquad \lambda\in\Delta_{|\mathcal B|}.
\]

记该可行域为 \(\mathcal Q_{\text{coarse}}\)。

### 细粒度行为选择可行域

允许在相同来源质量约束下改变桶内训练暴露：

\[
\sum_{i\in b}q_i=\pi_b,
\qquad
q_i\ge 0.
\]

记为 \(\mathcal Q_{\text{fine}}\)。

显然：

\[
\mathcal Q_{\text{coarse}}
\subseteq
\mathcal Q_{\text{fine}}.
\]

定义 max-min 局部联合学习值：

\[
t(\mathcal Q)
=
\max_{q\in\mathcal Q}
\min_m (Aq)_m.
\]

于是定义 **Joint Learning Opportunity Gap**：

\[
\boxed{
\Delta_{\text{JLO}}
=
t(\mathcal Q_{\text{fine}})
-
t(\mathcal Q_{\text{coarse}})
\ge 0.
}
\]

解释：

- \(\Delta_{\text{JLO}}\approx 0\)：粗粒度 task/domain mixing 已经解释了局部可用机会；
- \(\Delta_{\text{JLO}}>0\)：来源内部仍存在 coarse mixing 无法利用的训练作用结构；
- 只有当 \(\Delta_{\text{JLO}}\) 大于估计误差与有限步长误差时，才有理由投入 atoms / 细粒度选择。

这是 Idea 01 现在最重要的诊断量。

---

## 5. 固定来源比例后为什么仍可能有收益

若保持每个来源桶质量 \(\pi_b\) 不变，baseline 为 \(p\)，新分布为 \(q\)，则对目标 \(m\)：

\[
\mathbb E_q[A_m]-\mathbb E_p[A_m]
=
\sum_b
\pi_b
\operatorname{Cov}_{p(\cdot\mid b)}
\left(
\frac{q_i}{p_i},
A_{mi}
\right).
\]

因此：

> 多场景本身既不充分，也不必要；真正需要的是来源内部存在稳定、可估计、可利用的训练作用异质性。

这解释了为什么旧 Office + Industrial 两域实验仍可能失败：有两个 domain 并不自动意味着桶内存在可利用结构，也不意味着当前估计器可靠。

---

## 6. 三类失败必须区分

### 6.1 参数/目标本身局部冲突

即使允许参数空间内任意信赖域更新，也找不到共同方向。  
此时不能简单归因为“数据选得不好”。

### 6.2 参数空间可行，但训练池支持不足

存在理想共同下降方向，但当前样本更新的可达集合不能实现。  
这才支持“需要互补数据 / 新数据获取 / 更丰富监督”。

### 6.3 当前池理论上有机会，但估计器不可靠

例如：

- probe prompt 与正式训练模板不一致；
- 只用 readout gradient 学组，却用 LoRA/full gradient 训练；
- 同一 query 集既搜索候选又选择赢家；
- gradient-atoms 字典不稳定；
- 局部一步预测不能外推到 200+ steps；
- 小 reference set 的 winner's curse。

这类失败应先修 estimator，而不是简单扩大数据或加大选择力度。

---

## 7. 数据不足与数据丰富 regime

设候选池的总监督成本：

\[
C(\mathcal P)=\sum_i c_i
\]

训练预算为 \(B\)，定义：

\[
\rho=\frac{C(\mathcal P)}{B}.
\]

### \(\rho\lesssim 1\)：候选池不足/未饱和

如果数据质量正常，所有样本都可获得充分暴露，则 full-data 应作为强默认。  
此时选择若有收益，通常需要额外证据说明存在：

- 噪声标签；
- 明确负作用；
- 极端重复 occurrence；
- 非均匀训练成本导致的预算竞争。

### \(\rho>1\)：预算竞争

候选监督已经多于能被模型充分消费的训练预算。  
此时才研究如何分配 exposure。

**但 \(\rho>1\) 仍不是 atoms 必然有效的充分条件。**  
还需要 \(\Delta_{\text{JLO}}\) 和独立 effect prediction 显示来源内选择机会真实存在。

---

## 8. Idea 01 的实际目标函数

候选方案不能只追求一个绝对的预测下降，而应和同预算 baseline 比较。

令 baseline 分布为 \(p\)，候选为 \(q\)，优化：

\[
\max_q
\sum_m\omega_m A_m(q-p)
\]

并加入能力保护约束：

\[
A_m(q-p)\ge-\tau_m,
\]

以及：

- response-token 预算；
- task/domain/root 质量约束；
- density-ratio 上下界；
- rare validated capability 最小暴露；
- 必要时有限步 smoothness / trust-region 约束。

主推荐目标可进入收益项，物料映射、用户理解、合法 SID、约束遵循等必要能力可进入 non-degradation guard。

---

## 9. 与最新 data-mixture 研究的关系

### Learned domain

Nemotron-CLIMB 用 embedding 聚类得到语义 domain，再用 proxy model + predictor 搜索 mixture。DoGraph 从 per-sample gradient 出发重新思考 domain 与 mixing。

对 Idea 01 的启发：

> 优化之前先问“mixing unit 是否合理”，但我们仍需要一个独立指标判断 learned unit 是否产生了 source-level mixing 看不到的联合学习机会。

### Structured reference

Mixture of Data Experts 用 specialist behavior 特征预测 mixture loss；AutoMixAlign 用 specialist loss frontier 定义 generalist 的任务缺口并动态调权。

对 Idea 01 的启发：

> 仅仅优化 \(\lambda\) 太粗；训练分配应该相对于“当前能力缺口/目标 reference”来定义，而不是只按样本频率。

### Non-smooth / threshold behavior

“Data Mixing Can Induce Phase Transitions in Knowledge Acquisition”展示了一个受控场景：mixing ratio 与 model size 的响应可能出现临界阈值，而非始终平滑。

对本项目的使用边界：

- 可以把“selection benefit 是否随 supervision richness / model size 出现 regime transition”作为经验问题；
- 不能在尚未观察到临界现象前，把我们的推荐实验称为 phase transition；
- 小模型 recipe 不能默认平滑外推到大模型。

---

## 10. 任务场景

### 主场景 A：LC-Rec / MiniOneRec 式多角色生成推荐

同时包含：

- history → next item/SID；
- text ↔ item identifier；
- user-history → preference/intention supervision；
- 可选跨表示预测。

适合检验“热门、容易的 item grounding 是否挤占稀有用户条件信号”。

### 主场景 B：RecLM-gen 式 controllable recommendation

同时包含：

- 普通推荐；
- 正类别约束；
- 负类别约束；
- 个人化控制；
- 比例/数量控制。

适合检验“文本高度相似但训练作用相反”的监督是否会被错误去重。

### 扩展：Amazon Reviews 2023 多类目

用于扩大独立 root 与派生任务池，研究 candidate-pool richness。  
不能把 review 当作曝光日志。

### 结构补充：KuaiSAR

适合跨 search/recommend service 的结构验证；匿名 query/caption 不适合作为自然语言理解主证据。

---

## 11. Baselines

### coarse mixing / multi-objective

- full pool；
- uniform fixed-budget；
- task/domain/length/root stratified random；
- MoRec；
- DoGraph-style learned groups；
- AutoMixAlign-style task gap reweighting（若可适配 SFT）。

### fine-grained selection

- DEALRec；
- GORACS；
- LESS；
- GRAD-MATCH；
- raw full-gradient target alignment；
- KMeans in projected gradient space；
- PDF dominant-atom rule；
- full signed atom code；
- Gram-corrected code。

外部算法未复现时必须标 not-run；内部简化 cosine 不得冒充 LESS。

---

## 12. 必须先跑的机制实验

### E01-A：JLO gap 测量

在同一 checkpoint、同一 reference、同一训练预算下比较：

\[
t(\mathcal Q_{\text{coarse}})
\quad\text{vs.}\quad
t(\mathcal Q_{\text{fine}}).
\]

若差值小于误差，不进入复杂 atoms 长训。

### E01-B：effect prediction

候选分布由 probe block 提名；用独立 confirm block 验证 8 / 16 / 32 个真实 optimizer steps 后的变化。

评价：
- predicted vs realized gain correlation；
- sign accuracy；
- winner bias；
- cross-window stability。

### E01-C：richness × budget

独立改变：

- root 数量；
- 每个 root 的派生 task 数；
- domain 数；
- 训练 response-token budget。

主图：

\[
\text{fine-grained selection gain over coarse/uniform}
\quad
\text{vs. candidate-pool richness at fixed }B.
\]

---

## 13. Killer tests

Idea 01 降级或停止，如果：

1. \(\Delta_{\text{JLO}}\) 长期接近 0；
2. source-level mixing 已解释所有收益；
3. 独立短训与局部 response 方向不相关；
4. 只有同一 reference 上有提升，独立用户/时段不成立；
5. 简单 task/domain/root stratification 等效；
6. equal-total-compute 的更长 uniform 训练追平；
7. 多场景增加后选择收益没有更明显，而只是数据量效应；
8. 真实训练收益无法跨 seed 复现。

---

## 14. 当前证据与边界

必须保留 `exp/01_02` 的历史负结果：

- 旧严格 A/B complementarity 没有稳定通过；
- Office/Industrial 两域本身没有建立稳定负迁移/正迁移；
- 旧 atoms 配方在 E006 没有超过 fixed / cluster control，并在 Industrial 开发集明显为负；
- 单个训练 seed 不能推广为“所有 gradient atoms 无效”。

因此新版 Idea 01 是**由负结果推动的问题重定义**，不能重写历史成“旧实验已经证明多场景方案有效”。

当前状态：

- 数学诊断：可实施；
- 推荐真实收益：未建立；
- SFT 主线：允许继续；
- RL：在 SFT regime interaction 建立前不优先。
