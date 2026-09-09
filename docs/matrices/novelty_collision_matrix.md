# Novelty collision matrix — 2026-09-09 sync

| Candidate contribution | Closest collision | What is already covered | What must be demonstrated to keep our claim |
|---|---|---|---|
| Idea 01: joint learning opportunity | MoRec, AutoMixAlign, MGDA, Filter-then-Weight | multi-objective mixing / adaptive weighting / common descent | fine-grained opportunity beyond source-level mixture, validated by independent real training effects |
| Idea 02: gradient atoms as mixing units | Gradient Atoms, DoGraph, GORACS, CLIMB | sparse gradient behaviors, gradient domains, group selection, learned semantic domains | atoms beat raw gradient / KMeans / semantic units in effect prediction, stability, cost, or capability preservation |
| Idea 11: data-rich fixed-budget regime | pruning/scaling work, data-constrained LM, phase-transition mixing | selection depends on data/compute regime; mixture effects can be non-smooth | LLM4Rec-specific candidate-richness × train-budget interaction across task families/backbones |
| Rare capability protection | AutoMixAlign, task quotas | weakest-task / task-gap emphasis exists | within-task learned units preserve validated rare capabilities beyond manual quotas |
| Directional refill | attribution / retrieval / gradient matching | target-aligned retrieval is established | matched-cost refill repairs a specific capability better than random/raw-gradient alternatives |
| Full signed multi-atom codes | Gradient Atoms | sparse signed gradient decomposition exists | full-code/Gram geometry matters downstream beyond dominant atom and ordinary grouping |
| Learned domain claim | CLIMB, DoGraph | learned domains are not novel | recommendation-specific behavioral units with independent training-value evidence |
| Dynamic limited-budget mixture | Dynamic Data Mixing for MoE | redundancy-aware dynamic mixing exists | recommendation-specific multi-role constraints + user/item/control metrics + behavior units |
| Scalable attribution | RISE / readout sketching | cheap attribution representations exist | our cost story must compare against cheap sketches, not only full LoRA gradients |

## Hard rule

If a simpler baseline explains the gain, narrow the paper claim rather than adding modules:

- source mixing explains gain -> Idea 01 becomes mixture study;
- raw gradients explain gain -> drop atoms;
- KMeans explains gain -> drop dictionary-learning novelty;
- stratified random explains gain -> describe distribution rebalancing;
- full-data equal-compute catches up -> no efficiency/selection advantage;
- only proxy margin improves -> no recommendation-effect claim.
