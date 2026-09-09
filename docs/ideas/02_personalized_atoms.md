# IDEA 02 — Behavioral Gradient Atoms：面向多角色 LLM4Rec 的可学习 Mixing Units

**当前定位：表示/结构主候选，但必须“凭实验赢得贡献资格”。**  
Idea 02 不再局限为“个性化残差 atom”，也不把 atom 去重直接写成贡献。新版问题是：

> 人工 task/domain 标签是否过粗？能否从真实 SFT 梯度中学习可复用的 behavioral update units，用来表示多角色推荐训练中的重复、互补和稀有能力，并在固定训练预算下改善训练暴露分配？

这与 Idea 01 / 11 的关系：

- Idea 01 判断 **是否存在 source-level mixing 看不到的联合学习机会**；
- Idea 02 学习 **比人工 source label 更细的 mixing units**；
- Idea 11 研究 **这些 units 在什么数据丰富度/训练预算 regime 下真正有用**。

---

## 1. 必须严格继承 PDF 的方法定义

用户提供的 `gradient atom(1).pdf` 给出的核心链路是：

```text
SFT sample
 -> response-only SFT loss
 -> per-example LoRA gradient
 -> L2 normalization + fixed signed-hash projection
 -> low-dimensional gradient direction
 -> Dictionary Learning
 -> sparse signed code
 -> selection / distribution comparison / refill
```

核心近似：

\[
x_i\approx c_iD.
\]

其中：

- \(x_i\)：固定坐标系中的低维梯度方向；
- \(D\)：共享 gradient atoms；
- \(c_i\)：样本的稀疏带符号 code。

PDF 明确强调：

1. atom 是共享更新方向，不是样本类别；
2. 一条样本通常由多个 atom 构成；
3. dominant atom 只是摘要；
4. 稀有 atom 不能自动删除；
5. random projection 必须冻结并复用；
6. 方向与幅度应该分别保存；
7. gradient atoms 不能直接预测最终榜单，必须用训练消融验证。

这些边界在新版 Idea 02 中全部保留。

---

## 2. 与 Gradient Atoms 原论文的关系

Gradient Atoms (2026) 已经提出：

- 对训练梯度做稀疏分解；
- 用共享 gradient directions 发现可解释行为；
- 在权重空间进行 steering。

因此我们不能声称“首次提出 gradient atom / 稀疏梯度字典”。

而且用户 PDF 的工程管线使用的是固定 signed-hash projection + MiniBatchDictionaryLearning / OMP；这**不是**对原论文 preconditioned eigenspace 后端的完整复现。

我们的潜在新意只能来自：

> 将 gradient atoms 作为 **multi-role LLM recommendation 的 learned mixing units**，并证明它们在固定 source exposure、固定 train budget 和强 raw-gradient / KMeans / data-selection baselines 下提供不可替代的训练收益或估计优势。

---

## 3. 第一处必须修正：probe 与真实训练必须同构

旧 PDF 已指出 atom prompt 显式加入 `task_type`，而正式训练使用另一个 chat template。这可能人为放大 task separation。

新版强制：

\[
\text{encode}_{probe}(x_i,y_i)
=
\text{encode}_{train}(x_i,y_i)
\]

至少在以下方面一致：

- tokenizer；
- system/chat template；
- response-only mask；
- EOS；
- truncation；
- max prompt/response tokens；
- special item tokens；
- loss normalization。

`task_type/domain/source` 只能作为 metadata，除非正式训练输入本来就显式包含它。

---

## 4. 第二处必须修正：方向与幅度分开保存

对 full trainable gradient：

\[
g_i.
\]

固定投影：

\[
z_i=Rg_i.
\]

保存三种对象：

\[
\texttt{projected\_raw}=z_i,
\qquad
\texttt{projected\_unit}
=
\frac{z_i}{\|z_i\|},
\qquad
\texttt{full\_grad\_norm}
=
\|g_i\|.
\]

原因：

- unit direction 回答“往哪里更新”；
- norm 回答“局部更新有多强”；
- raw sketch 近似 magnitude-aware attribution。

只保留 unit vector 会把作用强度差异抹掉。

---

## 5. 第三处必须修正：不能再把 dominant atom 当最终选择标签

若：

\[
c_i=(c_{i1},\ldots,c_{iK}),
\]

dominant atom 只是：

\[
k_i^*
=
\arg\max_k |c_{ik}|.
\]

PDF 在 task-split-v3 上报告的 dominant share 大部分并不高，说明多数样本是 multi-atom mixture。

因此：

- dominant atom：只用于可视化、粗统计、代表样本；
- final selection：使用完整 signed code / reconstructed gradient；
- high-entropy sample：不能因为“不纯”自动删掉；
- rare atom：不能因为小自动保留，也不能因为小自动删除。

---

## 6. 第四处必须修正：code-space geometry

Dictionary atoms 不保证正交。

若：

\[
\hat x_i=c_iD,\qquad
\hat x_j=c_jD,
\]

则：

\[
\langle \hat x_i,\hat x_j\rangle
=
c_i(DD^\top)c_j^\top.
\]

所以直接用 \(c_ic_j^\top\) 当“梯度余弦”一般是错的。

定义 Gram matrix：

\[
G_D=DD^\top.
\]

则 code-space 的正确重构内积为：

\[
\langle c_i,c_j\rangle_{G_D}
=
c_iG_Dc_j^\top.
\]

新版所有：

- nearest-neighbor；
- directional refill；
- code diversity；
- atom-code clustering

都必须使用：

1. reconstructed projected gradients；或
2. Gram-corrected metric。

---

## 7. 第五处：atom 的正负激活都要保留

OMP code 有正有负。  
同一个 atom 方向与 code 可以共同翻转而不改变重构，因此 atom ID 与方向符号不是跨字典的天然语义。

在固定字典内，可定义：

\[
w_{ik}^{+}=[c_{ik}]_+,
\qquad
w_{ik}^{-}=[-c_{ik}]_+.
\]

对目标 \(m\) 的 atom-side response：

\[
R_{mk}^{+}
=
\frac{\sum_iw_{ik}^{+}A_{mi}}
{\sum_iw_{ik}^{+}+\epsilon},
\]

\[
R_{mk}^{-}
=
\frac{\sum_iw_{ik}^{-}A_{mi}}
{\sum_iw_{ik}^{-}+\epsilon}.
\]

它们不是“atom 的固定语义”，而是在当前模型/字典/任务 reference 下的经验带符号作用。

跨 seed / checkpoint 比较时，需要：

- atom matching；
- sign alignment；
- subspace angle；
- response profile；

不能直接比较“Atom 37”。

---

## 8. Idea 02 的真正价值假设

Atoms 不会扩大真实样本可实现更新的凸包：

\[
\mathcal Q_{\text{atom}}
\subseteq
\mathcal Q_{\text{sample}}.
\]

因此在“每个样本精确完整 gradient + 精确 response 全已知”的理想条件下，atom 不是因为可行域更大而获益。

它只有四种可能的真正价值：

### A. 更可靠的有限样本 effect estimation

同一 behavioral unit 聚合多个 noisy per-example responses，降低估计方差。

### B. 更低的高成本 query 数

用廉价 sketch/code 组织大池，只对候选组做完整 gradient/reference 查询。

### C. 跨窗口/跨任务结构复用

某些 behavioral units 的作用在训练阶段变化中比逐样本 score 更稳定。

### D. 稀有能力保护

在 source label 内部发现少量但被 held-out capability 验证为必要的更新结构。

如果四项都没有成立，而 raw gradient 或 KMeans 已经达到相同效果，则 atoms 应从论文核心模块降级。

---

## 9. 与最新 learned-domain / data-mixture 研究的关系

### Nemotron-CLIMB

先 embedding + clustering 得到 semantic domains，再通过大量 proxy mixture runs 和 predictor 搜索 mixture。

区别：

- CLIMB 的 learned units 来自语义 embedding；
- Idea 02 的 units 来自模型当前 SFT 梯度；
- 我们不能声称“首次 learned domain”；
- 需要证明 gradient units 比 semantic cluster 更适合推荐训练作用。

### DoGraph

从 per-sample gradient 视角重新定义 domain 并用于 data mixing。

直接碰撞非常强：

> 如果简单 gradient clustering / domain reweighting 已经够用，dictionary atoms 没有理由成为核心贡献。

### Mixture of Data Experts / AutoMixAlign

它们说明 mixture weights 太粗，需要 structured reference：

- MDE 用 specialist behavior 帮助预测 mixture loss；
- AutoMixAlign 用 specialist losses 定义 generalist gap，并动态 reweight/sample。

Idea 02 可以吸收的原则：

> atom 不应只描述“数据像什么”，还应该配合独立 capability reference 描述“它对当前模型缺口有什么作用”。

### Phase-transition work

数据 mixture 对能力学习可能存在阈值。对 Idea 02 的直接要求是：

- atom 的重要性不能只在一个模型尺度和一个 mixing ratio 上测；
- 不同模型/训练阶段必须重新 probe；
- 不要默认小模型上的 atom recipe 平滑迁移到更大模型。

---

## 10. 是否还需要“个性化残差 atom”？

保留，但降为 **diagnostic branch**，不是默认主方法。

可以对共享因素 \(Z\) 做加权残差：

\[
R=G-Z(Z^\top WZ)^\dagger Z^\top WG.
\]

或比较：

\[
g(h_i,y_i)
-
\mathbb E_{\tilde h_i}
g(\tilde h_i,y_i).
\]

它们可以诊断：

- template；
- output length；
- SID prefix；
- item popularity；
- history dependence

是否主导 atom。

但：

- residual 不等于因果兴趣；
- 共享 item grounding / 格式能力可能是真正有用，不能先验全部去掉；
- 只有诊断证据支持时，才将 residual atoms 升级为正式方法。

---

## 11. 任务场景

### 场景 A：LC-Rec / MiniOneRec 多角色 SFT

重点检查：

- item grounding 样本是否形成超大高密度方向；
- user-conditioned 样本是否在 source 内部形成少量但稳定的 distinct units；
- full code 是否优于 dominant atom；
- rare capability guard 是否比简单 task quota 更有效。

### 场景 B：RecLM-gen controllable recommendation

重点检查：

- 正类别与负类别约束是否在语义 embedding 上接近、在 gradient space 上分开；
- 同义模板是否在 gradient space 上高度重复；
- 稀有组合约束是否能由少量 atoms 表示并被保护。

### 场景 C：多类目 Amazon expansion

研究 atom 结构随：

- root 数；
- item 多样性；
- task derivations；
- domain 数

变化的 scaling。

---

## 12. 强 baselines

Idea 02 必须直接战胜：

1. raw projected gradient；
2. raw full-gradient target alignment；
3. KMeans on same projected gradients；
4. PDF dominant-atom dedup；
5. full code without Gram correction；
6. Gram-corrected code；
7. semantic embedding clusters；
8. DEALRec；
9. GORACS；
10. LESS；
11. DoGraph-style gradient groups。

若 atoms 只比 dominant-atom rule 好，但不比 raw gradient / KMeans 好，不能称核心方法贡献。

---

## 13. Atom-specific experiments

### E02-A：representation fidelity

比较 projection dim 256/512/1024，atom count 32/64/128，sparsity 2/4/8：

- reconstruction error；
- pairwise cosine rank preservation；
- atom tail distribution；
- multi-seed dictionary stability；
- task/domain purity（仅诊断，不作为优化目标）。

### E02-B：effect prediction

用 atom group 的 response 预测 held-out 真实短训效果，和：

- per-example raw gradient；
- KMeans group mean；
- semantic cluster mean

比较：

- sign accuracy；
- MSE/correlation；
- calibration；
- cross-window decay。

### E02-C：rare capability protection

人工 quota、rare atom、held-out capability-positive atom 三种策略对比。

如果“rare atom”本身不如 capability-validated guard，则论文必须写成后者。

### E02-D：cost

必须报告：

\[
C_{\text{probe}}
+
C_{\text{projection}}
+
C_{\text{dictionary}}
+
C_{\text{selection}}
+
C_{\text{train}}.
\]

固定随机投影降低存储，不会自动消除逐样本 backward 成本。

---

## 14. Killer tests

Idea 02 降级，如果：

1. raw gradient ≈ atom；
2. KMeans ≈ atom；
3. dominant atom 和 full code 无实质差异；
4. atom response 在独立短训上无法预测作用；
5. dictionary seed / checkpoint 变化导致结构完全不稳定；
6. rare atom protection 不如简单 task quota；
7. 等总 compute 的 uniform / stratified training 追平；
8. 只在 probe prompt 不匹配生产训练时有效。

---

## 15. 当前状态与历史证据

旧仓库证据必须继续保留：

- E006 atoms 没有超过 fixed / ordinary cluster；
- atoms 额外表示和选择开销没有换来收益；
- 单个 seed 的负结果不能证明整个方向永远无效；
- 旧实现使用 partial readout sketch 学组、full trainable gradient 估响应，不能等同于当前 production-aligned full-code 新假设。

当前允许继续的原因不是“旧实验其实成功”，而是我们已经明确了更强的可证伪问题：

> **atoms 是否在 data-rich、fixed-budget、多角色 LLM4Rec 中作为 learned mixing units 提供 raw gradient / KMeans / source mixing 无法替代的估计或训练价值？**

答案目前仍然未知。
