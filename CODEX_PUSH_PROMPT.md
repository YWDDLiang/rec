# Codex 简易交接 Prompt

你现在要把我下载好的两个增量包合并并推送到两个 GitHub 仓库。**不要重写历史实验，不要删除现有文件，不要修改 Research/crystal。**

## 目标仓库

1. 实施仓库：`https://github.com/YWDDLiang/rec`
2. 调研仓库：`https://github.com/YWDDLiang/Research`
   - 只更新 `llm4rec/`
   - `crystal/` 必须保持不变

## 本地文件

我会把下面两个 ZIP 下载并解压到本地：

- `rec_pool_selection_update_v3_with_idea.zip`
- `Research_llm4rec_pool_update_v3_with_idea.zip`

## 你要做的事情

### A. 更新 YWDDLiang/rec

1. clone / pull `YWDDLiang/rec` 的 `main`。
2. 把 `rec_pool_selection_update_v3_with_idea.zip` 解压内容合并到 **rec 仓库根目录**。
3. 重点新增/更新：
   - `src/rec_pool/`
   - `tests/pool/`
   - `experiments/pool/`
   - `scripts/pool/`
   - `configs/pool/`
   - `docs/pool/`
   - `docs/ideas/11_budget_conditioned_behavioral_selection.md`
   - `docs/ideas/README.md`
   - `references/pool/`
   - `results/pool_20260909/`
4. **不要改写或删除 `exp/01_02/` 冻结实验。**
5. 若 README 需要导航，只做最小修改：加入当前主方向链接
   `docs/ideas/11_budget_conditioned_behavioral_selection.md`
   和实现入口 `docs/pool/IDEA.md`。
6. 运行：
   ```bash
   python -m pytest -q
   python -m pytest tests/pool -q
   python scripts/validate_repo.py
   ```
   如果环境缺可选 HF/CUDA 依赖，区分“可选依赖未安装”和真实测试失败，不要伪造通过。
7. 审计 `git diff`：
   - 不应有模型权重、原始私有数据、token、密码；
   - 不应误删旧负结果；
   - 不应把 synthetic/CPU validation 写成真实 LLM4Rec 效果。
8. commit：
   `research: add budget-conditioned behavioral selection for LLM4Rec`
9. push 到 `main`（如果 main 有新提交，先 rebase/merge，解决冲突后重跑测试再 push）。

### B. 更新 YWDDLiang/Research

1. clone / pull `YWDDLiang/Research` 的 `main`。
2. 把 `Research_llm4rec_pool_update_v3_with_idea.zip` 解压内容合并到 **Research 仓库根目录**。
3. 只允许更新：
   - `llm4rec/revisions/2026-09-09-data-rich/`
   - `llm4rec/docs/ideas/11_budget_conditioned_behavioral_selection.md`
   - `llm4rec/docs/ideas/README.md`
   - 必要时对 `llm4rec/README.md` 做最小导航更新
4. **不要修改 `crystal/`、根目录模板或其他研究专题。**
5. 检查文档内部链接；如仓库校验器可用，运行：
   ```bash
   python llm4rec/scripts/validate_repo.py
   ```
6. 保留所有证据边界：
   - Gradient Atoms 是局部更新方向表示，不直接预测榜单；
   - “数据丰富时选择更重要”是待验证的 regime hypothesis；
   - 当前真实 LLM4Rec SOTA 提升尚未完成；
   - 旧 E006 负结果不能删除。
7. commit：
   `docs(llm4rec): promote budget-conditioned behavioral selection idea`
8. push 到 `main`。

## 最后给我汇报

只需要告诉我：
- 两个仓库各自最终 commit SHA；
- 新增/修改了哪些关键文件；
- 测试是否通过；
- 是否有冲突或未完成项；
- 不要顺便改方法，不要自行启动大规模 GPU 训练。
