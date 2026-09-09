# Codex 简易同步 Prompt（Idea 01 / 02 / 11 + 新调研背景）

我有两个已经准备好的增量包：

1. `rec_idea01_02_background_sync_v4.zip`
   -> 目标仓库：`https://github.com/YWDDLiang/rec`

2. `Research_llm4rec_idea01_02_background_sync_v4.zip`
   -> 目标仓库：`https://github.com/YWDDLiang/Research`

请严格按下面步骤做，**不要自行重写研究主线，不要删除旧负结果，不要启动大规模GPU训练。**

## A. 更新 YWDDLiang/rec

1. pull 最新 `main`。
2. 将 `rec_idea01_02_background_sync_v4.zip` 解压内容合并到仓库根目录。
3. 允许新增/更新的重点文件：
   - `docs/ideas/01_joint_learning_and_complementarity.md`
   - `docs/ideas/02_personalized_atoms.md`
   - `docs/ideas/11_budget_conditioned_behavioral_selection.md`
   - `docs/ideas/README.md`
   - `docs/landscape/05_data_rich_selection_and_data_mixture.md`
   - `docs/landscape/06_non_onereason_tasks_and_baselines.md`
   - `docs/matrices/novelty_collision_matrix.md`
   - `docs/pool/`
   - `references/pool/`
   - `src/rec_pool/`
   - `tests/pool/`
   - `experiments/pool/`
   - `scripts/pool/`
   - `configs/pool/`
4. **严禁删除、重写或“修正为正结果” `exp/01_02/`。**
5. root `README.md` 只做最小导航更新（如果当前README尚未有新方向入口）：
   - Idea 01
   - Idea 02
   - Idea 11
   - `docs/landscape/05_data_rich_selection_and_data_mixture.md`
6. 检查新研究定位：
   - Idea 01 = coarse source mixing vs fine-grained Joint Learning Opportunity；
   - Idea 02 = Behavioral Gradient Atoms learned mixing units；
   - Idea 11 = candidate richness × fixed training budget regime；
   - 三者是层次关系，不是只保留11。
7. 运行：
   ```bash
   python -m pytest -q
   python -m pytest tests/pool -q
   python scripts/validate_repo.py
   ```
   可选HF/CUDA依赖缺失要单独注明，不能伪造通过。
8. 检查：
   ```bash
   git diff --stat
   git diff
   ```
   确保：
   - 无模型权重；
   - 无私有推荐数据；
   - 无密钥/token；
   - 没有把synthetic/CPU验证写成真实LLM4Rec效果；
   - 新增外部研究只写为调研/未独立复现，不虚构baseline成绩。
9. commit：
   `research: synchronize ideas 01 02 11 and data-rich LLM4Rec landscape`
10. push `main`。如远端有新提交，先同步再解决冲突，之后重跑测试。

## B. 更新 YWDDLiang/Research

1. pull 最新 `main`。
2. 解压 `Research_llm4rec_idea01_02_background_sync_v4.zip` 到仓库根目录。
3. **只允许修改 `llm4rec/`。不要修改 `crystal/`。**
4. 重点更新：
   - `llm4rec/docs/ideas/01_joint_learning_and_complementarity.md`
   - `llm4rec/docs/ideas/02_personalized_atoms.md`
   - `llm4rec/docs/ideas/11_budget_conditioned_behavioral_selection.md`
   - `llm4rec/docs/ideas/README.md`
   - `llm4rec/docs/landscape/05_data_rich_selection_and_data_mixture.md`
   - `llm4rec/docs/landscape/06_non_onereason_tasks_and_baselines.md`
   - `llm4rec/docs/matrices/novelty_collision_matrix.md`
   - `llm4rec/revisions/2026-09-09-data-rich/`
5. 若需要，只对 `llm4rec/README.md` 做最小导航更新；不要改根Research README和`crystal/`。
6. 运行：
   ```bash
   python llm4rec/scripts/validate_repo.py
   ```
7. 保留边界：
   - “数据丰富时selection更重要”仍是待验证的regime hypothesis；
   - learned domain已有CLIMB/DoGraph等先例；
   - task-gap / structured mixture已有MDE/AutoMixAlign等先例；
   - phase transition只作为外部背景，不能预先声称LLM4Rec已经出现；
   - Gradient Atoms是局部更新表示，不能直接等价于榜单收益；
   - 旧E006负结果继续保留。
8. commit：
   `docs(llm4rec): sync ideas 01 02 11 and updated selection landscape`
9. push `main`。

## 最后只向我汇报

- `YWDDLiang/rec` 最终 commit SHA；
- `YWDDLiang/Research` 最终 commit SHA；
- 两个仓库关键新增/更新文件；
- tests / validation 是否通过；
- 是否发生冲突；
- 哪些外部 baseline 仍是 not-run。

不要自行启动正式GPU实验。
