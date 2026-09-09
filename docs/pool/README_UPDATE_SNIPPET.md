## 2026-09-09 当前主方向

当前优先研究方向为 [预算条件下的行为训练选择](../ideas/11_budget_conditioned_behavioral_selection.md)。

核心问题不是“压缩数据”，而是：当多角色LLM4Rec候选池的监督丰富度超过固定训练token/step预算时，如何利用完整带符号gradient-atom结构分配训练暴露，并保护稀有但已验证的能力。

实现入口：[docs/pool/IDEA.md](IDEA.md)。
