# 11｜Budget-Conditioned Behavioral Selection for Multi-Task LLM Recommendation

> Working title: **When More Data Stops Helping: Budget-Conditioned Behavioral Selection for Multi-Task LLM Recommendation**  
> Working method name: **BAGE — Budget-Aware Gradient Exposure**  
> Status: **primary paper candidate, not yet validated on a real LLM4Rec benchmark**.

## 0. One-sentence thesis

This project is **not a data-compression project**. It studies how to allocate a fixed amount of LLM training exposure when a recommendation SFT pool becomes richer than the available optimization budget:

> When every useful example can still be sufficiently exposed, using the full pool should remain a strong default; once heterogeneous multi-role supervision competes for a fixed token/step budget, training examples should be selected according to their **behavioral update structure**, while protecting rare but necessary recommendation capabilities.

The central empirical question is therefore not “how many rows can we remove?”, but:

\[
\boxed{
\text{When does additional supervision still add learnable information, and when does it mainly compete for training exposure?}
}
\]

The source gradient-atom protocol defines the representation pipeline as:

\[
g_i
\rightarrow
z_i
\rightarrow
x_i
\approx c_iD,
\]

where \(g_i\) is the per-example LoRA gradient, \(z_i\) is a fixed signed-hash projection, \(x_i=z_i/\|z_i\|\), \(D\) is a learned dictionary, and \(c_i\) is the sparse signed code. The source material explicitly warns that dominant atom is only a summary, that rare atoms cannot be deleted automatically, that magnitude should be preserved separately, and that training conclusions require actual ablations rather than atom visualization alone.

Source basis:
- `gradient atom(1).pdf`, especially pp. 1–16.
- `DataAttribution&Mixing.pptx`, especially the “learned domain / data mixture” and “trajectory effect / null hypothesis” sections.
- Frozen negative evidence in `YWDDLiang/rec/exp/01_02/`: the current atom recipe did not establish additional recommendation benefit.

---

## 1. Research question

Given a multi-task LLM recommendation training pool

\[
\mathcal P=\{(x_i,y_i,b_i,r_i,c_i)\}_{i=1}^{N},
\]

where:

- \(x_i,y_i\): production-aligned SFT input and response;
- \(b_i=(\text{task},\text{domain},\text{source})\): human-defined source bucket;
- \(r_i\): root interaction / root query identifier;
- \(c_i\): response-token training cost;

and a training exposure budget \(B\), learn a training distribution or schedule \(q\) that improves recommendation objectives while preserving required auxiliary capabilities.

The key comparison is:

\[
\textbf{full-pool exposure}
\quad\text{vs.}\quad
\textbf{uniform fixed-budget exposure}
\quad\text{vs.}\quad
\textbf{behavior-aware fixed-budget exposure}.
\]

The paper must distinguish three regimes rather than presuppose that selection is always useful.

### Regime A — data-limited / under-saturated

\[
C(\mathcal P)=\sum_i c_i \lesssim B.
\]

Most eligible data can receive meaningful exposure. Unless there is label noise or negative influence evidence, the full pool is a strong default.

### Regime B — budget-saturated

\[
C(\mathcal P) > B.
\]

Not every example can be exposed equally. Allocation matters even if no row is intrinsically “bad”.

### Regime C — behavior-rich / redundancy-rich

The pool is not only large, but contains repeated update directions alongside rare or cross-task directions. This is the regime where gradient atoms may provide value beyond task/domain mixing.

**Important:** \(C(\mathcal P)/B>1\) is not sufficient to prove that atom selection helps. The method must also show exploitable within-bucket heterogeneity.

---

## 2. Why LLM4Rec is a suitable domain

A modern generative recommender can simultaneously learn:

1. sequential recommendation;
2. controllable / conditional recommendation;
3. item-description ↔ item-identifier grounding;
4. user-history → preference / intent understanding;
5. cross-domain or cross-service recommendation;
6. explanation / reasoning or other auxiliary instruction tasks.

These roles can share model parameters but need not have identical training effects. Human task labels are therefore useful constraints but may be too coarse as the unit of training allocation.

This motivates a hierarchy:

\[
\text{source/task labels}
\rightarrow
\text{learned behavioral update units}
\rightarrow
\text{training exposure allocation}.
\]

The learned unit is **not claimed to be a semantic or causal “interest type”**. It is a reusable update direction defined at a specific probe checkpoint, training objective, tokenizer/template, LoRA initialization, parameter ordering and projection map.

---

## 3. Candidate benchmark matrix

### Main setting A — multi-role generative recommendation

Use an LC-Rec / MiniOneRec-style training pool that contains next-item recommendation plus item/user alignment roles.

Purpose:
- test whether abundant item-grounding or easy recommendation supervision crowds out rarer user-conditioned behaviors;
- retain a fully generative LLM4Rec setting;
- avoid binding the paper to OneReason.

Candidate backbones:
- Qwen3-1.7B;
- Llama-3.2-3B-Instruct for cross-architecture confirmation.

### Main setting B — controllable recommendation

Use a RecLM-gen-style task family:

- standard sequential recommendation;
- positive category constraints;
- negative category constraints;
- personal controllable recommendation;
- category-rate / count constraints.

Purpose:
- create a pool where superficially similar instructions can have opposite training roles;
- test rare compositional constraints rather than only next-item accuracy;
- jointly evaluate recommendation quality and instruction satisfaction.

### Expansion setting — data-rich Amazon multi-category pool

Use multiple Amazon Reviews categories to increase independent roots, item coverage and task derivations.

This setting is for **candidate-pool scaling**, not for claiming that reviews are exposure logs.

### Structural supplement — KuaiSAR

Useful for cross-service search/recommendation structure, but anonymized text fields prevent treating it as the primary natural-language understanding benchmark.

---

## 4. Core hypothesis

Let \(p_i\) be the baseline training probability and \(q_i\) the selected probability. Keep the source bucket mass fixed:

\[
\sum_{i\in b}q_i=\sum_{i\in b}p_i=\pi_b.
\]

For objective \(m\), let \(a_{mi}\) be the local signed benefit of training example \(i\). Then:

\[
\mathbb E_q[a_m]-\mathbb E_p[a_m]
=
\sum_b \pi_b
\operatorname{Cov}_{p(\cdot\mid b)}
\left(
\frac{q_i}{p_i},
a_{mi}
\right).
\]

### Consequence

Fine-grained selection can improve over pure source mixing **only if useful within-bucket variation exists and can be estimated**.

Therefore:

- many scenes are neither necessary nor sufficient;
- a single scene can still benefit if its internal training effects differ;
- many scenes can still show no benefit if each source is internally homogeneous or the estimator is unreliable.

This identity is the first paper-level falsifiability condition.

---

## 5. Behavioral representation

Follow the source PDF closely, but correct the known weaknesses.

### 5.1 Production-aligned per-example loss

Use the same tokenizer, chat template, truncation, loss mask and response-only objective as actual SFT.

Do not inject an analysis-only task string into the probe input if production training does not contain it.

### 5.2 Full trainable-parameter gradient

For sample \(i\):

\[
g_i=\nabla_{\theta_{\text{LoRA}}}\ell_i.
\]

Save a parameter manifest containing:

- checkpoint/revision;
- LoRA configuration and initialization;
- parameter names, shapes and flatten offsets;
- tokenizer/special tokens;
- prompt template;
- truncation;
- dtype;
- projection fingerprint.

### 5.3 Fixed projection with magnitude retained

\[
z_i=Rg_i,\qquad
x_i=\frac{z_i}{\|z_i\|}.
\]

Save:

- `projected_raw`;
- `projected_unit`;
- `full_grad_norm`.

The source PDF explicitly notes that unit directions alone lose update magnitude.

### 5.4 Sparse dictionary

\[
X\approx CD.
\]

Do not use dominant atom as the final selection representation. The source PDF reports that most samples are multi-atom mixtures; full signed sparse codes must be retained.

### 5.5 Correct code geometry

Because dictionary atoms need not be orthogonal,

\[
\langle c_iD,c_jD\rangle
=
c_i(DD^\top)c_j^\top.
\]

Thus raw code cosine is generally not equal to reconstructed-gradient cosine. Any code-space neighbor/refill procedure must either reconstruct gradients or use the Gram matrix \(G=DD^\top\).

---

## 6. Budget-conditioned selection

Define a pool exposure ratio:

\[
\rho=\frac{C(\mathcal P)}{B}.
\]

This ratio is a **regime descriptor**, not a sufficient decision rule.

### Case 1: budget slack

If \(\rho\le 1\), quality filters pass, and there is no independently verified harmful-sample signal, schedule all eligible data at least once.

This is a conservative policy, not a theorem that all data is always optimal.

### Case 2: fixed-budget competition

When \(\rho>1\), choose \(S\) or \(q\) under response-token budget:

\[
\sum_i q_i c_i \le B.
\]

A candidate objective is:

\[
F(S)
=
\alpha
\log\det\left(
I+\lambda^{-1}\sum_{i\in S}z_iz_i^\top
\right)
+
\beta
\sum_{i\in S}v_i
-
\gamma
R(S),
\]

subject to task/domain/root coverage constraints.

Interpretation:

- log-det term: diminishing returns for repeatedly covered update directions;
- \(v_i\): independently estimated signed target usefulness;
- \(R(S)\): duplicate / repeated-root / excessive-occurrence penalty.

The log-det component alone is **not enough** because \(z\) and \(-z\) have the same outer product. Signed validation alignment or another target-sensitive term is required.

### Rare-capability guard

Rare atoms are not automatically valuable. Protect rare **validated capabilities**, not rarity itself.

Require either:

- task/domain minimum exposure;
- held-out capability non-degradation;
- independently positive validation alignment.

---

## 7. Directional refill

If capability \(m\) falls after selection:

1. build an independent validation-gradient target \(v_m\);
2. search only the removed **training** pool;
3. rank candidates using full raw-gradient similarity or Gram-corrected code geometry;
4. match task/domain and response-token cost;
5. swap an equal-cost low-value selected example;
6. compare targeted refill against matched random refill.

No final test label/gradient is allowed in selection or refill.

---

## 8. Why this is not “data compression”

The paper's independent variable is **training opportunity under a fixed optimization budget**, not file size.

A selected dataset can have:

- fewer unique rows but the same training tokens;
- the same unique rows with different occurrence weights;
- the same task proportions but different within-task exposure;
- refill swaps that keep total rows and tokens fixed.

The scientific claim should be about:

\[
\text{supervision richness}
\times
\text{training budget}
\times
\text{behavioral diversity}
\rightarrow
\text{learned recommendation capability}.
\]

---

## 9. Strong baseline matrix

### Mandatory data-selection baselines

- full pool;
- uniform fixed-budget sampling;
- task/domain/length stratified random;
- PDF dominant-atom dedup;
- raw full-gradient target alignment;
- ordinary KMeans in the same projected-gradient space;
- full-code atom selection;
- Gram-corrected code selection;
- random matched refill.

### External research baselines to reproduce

- DEALRec;
- GORACS;
- LESS;
- MoRec / source-level data-mixture control;
- DoGraph-style learned gradient groups when practical.

If RL is added later:
- MiniRec / RL-specific data selection.

Do not relabel simplified in-house approximations as reproductions of these methods.

---

## 10. Scaling experiment that directly tests the thesis

Create nested root pools:

\[
\mathcal P_{12.5\%}
\subset
\mathcal P_{25\%}
\subset
\mathcal P_{50\%}
\subset
\mathcal P_{100\%}.
\]

Independently vary:

1. number of independent roots;
2. number of derived task roles per root;
3. number of scenes/domains;
4. training response-token budget \(B\).

The key plot is not “keep ratio vs score”. It is:

\[
\boxed{
\text{benefit of selection over uniform}
\quad\text{vs.}\quad
\text{candidate-pool richness at fixed training budget}
}
\]

Report:

- recommendation metric;
- controllability / instruction satisfaction;
- worst task/domain;
- long-tail / rare capability metric;
- legal item generation;
- actual response tokens;
- unique roots and occurrences;
- selection/probe/training compute.

---

## 11. Primary hypotheses

### H1 — under-saturated boundary

When the candidate pool is small enough that useful examples can all receive adequate exposure, behavior-aware selection should not be expected to produce a large stable gain over full-pool training.

A clear gain is possible if the pool contains noise or harmful supervision, so this is an empirical boundary hypothesis, not a theorem.

### H2 — data-rich fixed-budget gain

At fixed training tokens, once the candidate pool contains substantial repeated update directions and heterogeneous roles, behavior-aware exposure should beat uniform exposure if the learned representation predicts held-out training effects.

### H3 — atoms must earn their role

Atoms remain a core contribution only if full signed codes improve at least one of:

- held-out effect prediction;
- cross-window stability;
- rare-capability preservation;
- final recommendation performance;
- selection-query cost;

over direct raw-gradient and ordinary clustering baselines.

Otherwise the final method should drop atoms rather than preserve them for story consistency.

### H4 — source ratios are insufficient

Holding task/domain exposure fixed, within-source behavioral selection should still improve at least one target without unacceptable auxiliary degradation. If not, the contribution reduces to ordinary data mixture.

---

## 12. Killer experiments / stop conditions

Stop or downgrade the main claim if any of the following hold:

1. **No regime interaction:** selection does not become more useful as pool richness increases under fixed training budget.
2. **No within-bucket heterogeneity:** source-controlled selection gives no stable gain over source-level mixing.
3. **Raw gradient wins:** direct gradient selection matches or beats atoms under equal probe information and total compute.
4. **Simple stratification explains gain:** task/domain/length/root-matched random performs equivalently.
5. **Rare-direction fallacy:** protecting rare atoms hurts validated capability or test performance.
6. **Selection cost dominates:** equal-total-compute uniform training matches the method.
7. **Probe mismatch:** gradient scores do not predict independent 8/16/32-step training changes.
8. **Template dependence:** gains disappear when probe inputs exactly match production training.

Negative outcomes must remain in the repository.

---

## 13. Paper contribution candidates

Only claim contributions whose experiments pass.

### Contribution A — problem / evaluation

Define **budget-conditioned supervision allocation** for multi-role LLM recommendation, separating candidate-pool richness from actual training budget.

### Contribution B — representation / mechanism

Use full signed gradient-atom codes as learned behavioral mixing units while preserving magnitude and correcting non-orthogonal code geometry.

### Contribution C — capability-preserving selection

Optimize behavioral coverage under task/domain/root/token constraints with independently evaluated rare-capability guards and targeted refill.

### Contribution D — regime law

Empirically characterize where full-pool training is preferable and where data-rich fixed-budget selection becomes useful.

Contribution D is especially valuable if the transition is reproducible across at least two LLM4Rec task families and two backbones. It should be described as an empirical regime transition unless evidence supports a stronger mathematical “phase transition” claim.

---

## 14. Current decision

**Promote this idea to the primary SFT data-selection direction.**

Recommended first experiment:

1. non-OneReason backbone;
2. LC-Rec/MiniOneRec-style multi-role generative SFT;
3. nested candidate pools;
4. fixed response-token training budget;
5. full / uniform / stratified / raw-gradient / KMeans / PDF-atoms / full-code-atoms;
6. no RL until the SFT regime interaction is established.

The original 01/02 ideas remain useful as historical mechanisms, but the paper should no longer require strict A+B complementarity as its central story.
