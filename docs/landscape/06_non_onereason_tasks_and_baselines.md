# 非 OneReason 的 LLM4Rec 任务与 Baseline 地图（同步版）

## 主实验优先级

### A. LC-Rec / MiniOneRec 式多角色生成推荐

推荐作为第一主场景。

训练角色包括：
- sequential / next-item generation；
- item text ↔ identifier alignment；
- cross-representation prediction；
- user history → preference/intention supervision。

研究问题：
当 item grounding、热门轨迹、简单推荐样本越来越丰富时，固定训练 token 下是否挤占了更稀有的用户条件监督？

### B. RecLM-gen 式 controllable recommendation

推荐作为第二主场景。

任务包括：
- 普通 sequential recommendation；
- positive category control；
- negative category control；
- personalized control；
- category ratio/count constraint。

优势：
文本表面相近的数据可能具有相反训练作用，适合检验“语义去重 ≠ 训练作用去重”。

### C. Amazon Reviews 2023 多类目扩池

作用：
- 增加独立 root；
- 增加 item / text coverage；
- 构造更大的 multi-role candidate pool。

边界：
review 不是完整曝光日志；严格 next-item 输入不能读取目标发生后的评论内容。

### D. KuaiSAR 结构补充

作用：
同一批用户存在 search/recommend 两种服务记录，适合跨服务结构。

边界：
query keyword / caption 匿名化，不作为自然语言理解主 benchmark。

---

## 模型不绑定 OneReason

建议：

- Qwen3-1.7B：主开发；
- Llama-3.2-3B-Instruct：跨架构确认；
- 原生 LC-Rec / MiniOneRec / RecLM-gen backbone：用于严格原协议对照时保留。

跨 backbone：
- 重新提取 gradient；
- 重新 fit dictionary；
- 不能按 atom ID 比较；
- 选择记录 ID 的跨模型 transfer 是额外实验，不是默认成立。

---

## 必比的数据选择 baseline

### 基础预算控制

- full pool；
- uniform fixed-budget；
- task/domain/root/length stratified random；
- provenance duplicate-only removal；
- equal-total-compute longer uniform training。

### PDF 方法自身

- dominant-atom dedup；
- full-code selection；
- Gram-corrected code；
- random refill；
- targeted refill。

### 几何 / gradient

- raw projected gradient；
- full-gradient target alignment；
- KMeans；
- GRAD-MATCH；
- LESS。

### 推荐专用

- DEALRec；
- GORACS；
- MoRec。

### learned-domain / mixture

- DoGraph；
- source-level mixing；
- AutoMixAlign-style task-gap sampling（适配时明确不是原始 DPO 场景严格复现）。

### RL（后置）

- MiniRec；
- 只有 SFT regime interaction 成立后才进入。

---

## 必须匹配的预算

所有方法至少统一：

- backbone / initialization；
- candidate pool；
- train response tokens；
- optimizer steps；
- item catalog / legal decoder；
- reference access；
- selection-query budget；
- evaluation users / candidates；
- seeds。

另外单列：

\[
C_{\text{total}}
=
C_{\text{probe}}
+
C_{\text{selection}}
+
C_{\text{train}}.
\]

不允许只报筛完之后的训练时间。

---

## 论文主要指标

### 推荐

- Recall / HR；
- NDCG；
- legal item rate；
- duplicate output rate。

### controllable rec

- ranking quality；
- constraint satisfaction；
- list length/count；
- positive/negative category control。

### 能力保护

- item ↔ identifier grounding；
- user-condition task；
- rare control combination；
- worst task/domain。

### 方法诊断

- JLO gap；
- predicted vs realized short-window effect；
- atom reconstruction；
- dictionary stability；
- exposure TV/KL；
- unique roots / occurrences；
- response-token exposure。

---

## 最危险的审稿人问题

1. 为什么不是 task mixing？
2. 为什么不是 raw gradient？
3. 为什么不是 KMeans？
4. 为什么不是 DEALRec/GORACS/LESS？
5. 为什么不是简单 semantic dedup？
6. 多场景真的必要吗，还是只是数据量更大？
7. 数据少时全用更好，是否说明方法只是一种 budget artifact？
8. 每条样本都算 gradient，selection 计算是不是比训练省下来的还贵？
9. atoms 在不同 checkpoint / backbone 稳定吗？
10. rare atom 是否只是异常/噪声？

每个问题都必须有预注册实验，而不是只靠文字解释。
