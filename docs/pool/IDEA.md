# Current primary idea — Budget-Conditioned Behavioral Selection

This file is the **implementation-facing version** of `docs/ideas/11_budget_conditioned_behavioral_selection.md`.

## Intended paper claim

Do not frame this repository as data compression.

The executable research question is:

> Under a fixed response-token / optimizer-step budget, when a multi-role LLM recommendation SFT candidate pool becomes richer than the available training exposure, can full signed gradient-atom structure allocate training opportunities better than full-pool/uniform/source-level baselines while preserving rare validated capabilities?

## Required pipeline

```text
production-aligned SFT records
  -> per-example LoRA gradients at one frozen probe point
  -> fixed signed-hash projection
  -> save raw / unit / full-gradient norm
  -> sparse dictionary, full signed codes
  -> budget / source / root / capability constraints
  -> selection or occurrence schedule
  -> optional matched directional refill
  -> actual SFT consuming that exact schedule
  -> independent recommendation + capability evaluation
```

## Non-negotiable implementation rules

1. Probe encoding must match actual training tokenizer/template/loss mask/truncation.
2. Fixed projection and parameter manifest must be reusable and fingerprinted.
3. Dominant atom is diagnostic only; final selection uses full code or raw gradients.
4. Code-space similarity must account for \(DD^\top\), or reconstruct projected gradients.
5. Save magnitude; unit direction alone is insufficient.
6. Rare atom != rare valuable capability.
7. Final test data never enters selection/refill.
8. Selection intensity must be measured with response-token exposure, occurrences and TV/KL, not only unique-row count.
9. If the token budget can expose all clean data, full-pool is an explicit baseline/default regime.
10. Atoms remain a paper contribution only if they beat raw-gradient and ordinary-clustering controls under equal information/compute.

## Priority experiment grid

Axes:
- nested independent root pool: 12.5 / 25 / 50 / 100%;
- derived task richness: low / medium / high;
- scenes: 1 / 2 / multi;
- fixed training response-token budget;
- backbone: Qwen3-1.7B first, Llama-3.2-3B-Instruct confirmation.

Methods:
- full pool;
- uniform fixed-budget;
- task/domain/root/length stratified random;
- PDF dominant-atom rule;
- raw gradient selection;
- KMeans projected-gradient groups;
- full signed code;
- Gram-corrected code;
- targeted refill vs matched random refill.

Primary gate:
selection must outperform the strongest simple baseline **as candidate-pool richness grows at fixed training budget**, not merely at one arbitrary keep ratio.

## Existing negative evidence that must remain visible

`exp/01_02` showed that the previous atom configuration did not establish additional benefit and was materially worse than fixed mixing on one Industrial development run. Do not delete or rewrite those results. The new method is a new hypothesis motivated by the failure analysis, not a reinterpretation that makes the old run positive.

## Where the code lives

- new pool-selection implementation: `src/rec_pool/`
- configs: `configs/pool/`
- experiments: `experiments/pool/`
- detailed protocol: `docs/pool/`
- conceptual paper idea: `docs/ideas/11_budget_conditioned_behavioral_selection.md`

## Kill rule

If raw gradients or matched stratified random fully explain the gain, remove gradient atoms from the claimed core method.
