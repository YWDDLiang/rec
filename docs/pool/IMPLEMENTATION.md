# 实现、运行、接口与未完成范围

## 1. 安装与目录

这是对现有 `rec` 的新增目录包，新增Python namespace是 `rec_pool`，不改旧 `rec_lab`、旧实验和旧manifest。合并后可沿用原repo的editable安装，额外安装 `requirements-pool.txt`。也可在本包根目录设置 `PYTHONPATH=src` 独立运行。

```bash
python -m pip install -r requirements-pool.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/pool/run_tests.py
python experiments/pool/run_regimes.py
python experiments/pool/run_causal_pipeline.py
```

上面三条运行只需CPU，不会下载模型或数据。真实HF训练另需安装transformers、peft等并提供本地模型；本轮未测试真实HF依赖组合，不能将推荐版本范围当作已验证lockfile。

## 2. 文件说明

| 文件 | 功能 |
|---|---|
| `src/rec_pool/contracts.py` | 同一份生产编码Record、root/cell/内容hash、跨分区泄漏检查 |
| `sft.py` | response-only causal shift、完整可训练梯度流式投影、缓存、训练计划与梯度累积 |
| `geometry.py` | PDF方向字典、signed OMP、非正交字典Gram、完整code与raw对齐 |
| `selection.py` | token预算、来源上下限、保护来源、provenance cap、剩余配额可行greedy |
| `baselines.py` | PDF文字规则、KMeans、嵌套root reservoir；不冒充外部论文实现 |
| `refill.py` | 同来源同response长度的有限target/random交换 |
| `metrics.py` | 对外部生成ranking的严格交叉核验，非法/重复项保留原排名位置 |
| `data.py` | Amazon23流式字段标准化，以及小/中型规范事件到诊断SFT的构建 |
| `__main__.py` | reservoir/probe/fit/select/align/refill/plan/train命令 |

## 3. 原生模型与任务编码

首选由原始LC-Rec、MiniOneRec或RecLM-gen生产编码器导出Record，而不是用一个“看起来差不多”的prompt。Record的字段是：

```json
{
  "record_id":"unique-row-id",
  "root_id":"original-query-or-item-root",
  "task":"next_item",
  "domain":"office",
  "source":"native-frozen-generator",
  "split":"train",
  "input_ids":[1,5,9,2],
  "labels":[-100,-100,9,2],
  "recipe_id":"tokenizer-template-truncation-mask-fingerprint",
  "user_id":"original-user-id",
  "label_origin":"observed"
}
```

第一位置必须mask；所有监督位置标签必须等于原输入token。以上数字只解释schema，不是实际SID映射。一个root的不同派生任务共用root_id，record_id不同。不得为绕开泄漏检查把每个派生行伪装成独立root。

`export_native.py`接收本地 `module:function` 工厂，工厂输出生产Record。**工厂接口已实现，不等于已经改好所有第三方trainer。** 提供 `encode_jsonl.py` 作为普通HF chat路线，严格检查prefix一致且超长即报错；它不是原生LC-Rec等训练协议复现。probe和训练消费同一编码，但是否等于上游编码仍需外部parity确认。

SID路线要求本地checkpoint已经包含正确的新增词表、resize后的embedding以及对应目录。默认LoRA loader不会自己生成SID，不会随意训练新token embedding。上游使用modules_to_save的adapter需保持同配置。Llama模型授权由用户依法完成，本包不绕过授权或自动下载。

## 4. 一个离线流程

先从实际生产编码器得到 `data/train.jsonl` 和独立 `data/select.jsonl`。下面MODEL/ADAPTER等路径由用户实际环境决定。所有选择方法从相同warm模型开始；建议probe也在该warm模型上，而非一个随机LoRA尚未适配的新点。

```bash
MODEL=/path/to/local/warmed/base
ADAPTER=/path/to/local/warmed/adapter

python -m rec_pool probe --records data/train.jsonl \
  --model "$MODEL" --adapter "$ADAPTER" --device cuda \
  --base-revision PINNED_BASE_REVISION --dimension 512 --out cache/train

python -m rec_pool fit --records data/train.jsonl --cache cache/train \
  --atoms 64 --sparsity 4 --fit-limit 8192 --out cache/atoms

python -m rec_pool select --records data/train.jsonl --fit cache/atoms \
  --config configs/pool/selection.example.json --out runs/admission.json

python -m rec_pool plan --records data/train.jsonl --selected runs/admission.json \
  --budget-tokens 2000000 --out runs/training_plan.json

python -m rec_pool train --records data/train.jsonl --plan runs/training_plan.json \
  --model "$MODEL" --adapter "$ADAPTER" --device cuda --seed 42 \
  --batch-size 16 --microbatch 1 --lr 0.0001 --out runs/model
```

200万是示例预算，不是预先验证的最佳配方。selection.example默认也只是示例，不默认某类用户任务必须丢弃；真正运行前应从实际任务键生成保护与上下限。旧四任务名字匹配不到新任务时应失败，而不是静默忽略。

HF入口使用全新AdamW，所有方法必须同样重置。它目前不恢复上游scheduler/optimizer，也不实现DDP、packing、混合任务自定义loss；严格复现这些训练设置时，只使用本包选出的record IDs/occurrence计划，将其接回原native trainer，不声称这个轻量入口等价于其完整训练。

## 5. 回填

```bash
python -m rec_pool probe --records data/select.jsonl \
  --model "$MODEL" --adapter "$ADAPTER" --device cuda \
  --base-revision PINNED_BASE_REVISION --dimension 512 --out cache/select

python -m rec_pool align --records data/train.jsonl --reference data/select.jsonl \
  --cache cache/train --reference-cache cache/select --fit cache/atoms \
  --task YOUR_TARGET_TASK --out runs/target_scores.npy

python -m rec_pool refill --records data/train.jsonl --selected runs/admission.json \
  --scores runs/target_scores.npy --swaps 500 --out runs/refilled.json

python -m rec_pool refill --records data/train.jsonl --selected runs/admission.json \
  --scores runs/target_scores.npy --swaps ACTUAL_TARGET_SWAP_COUNT \
  --random-control --out runs/random_refilled.json
```

所有参考必须与候选同checkpoint、参数manifest、投影、recipe。`align`算的是重构投影方向cosine，不是LESS，不是精确全参数influence。当前参考中心为指定task下select记录的平均梯度；多domain的加权方式应显式分开构造，不可让大domain自然吞掉小domain。

不得直接拿最终audit掉点来反复确定target任务；掉点观察若已经参与决策，应将其标记development，最终冻结后另测。回填后需重新生成训练计划，不能只替换selected JSON而沿用旧plan。

## 6. 规模与成本

`reservoir_jsonl.py`使用两遍流式读取、O(K) root内存，先在原始JSONL层形成嵌套池。保留一个root的所有派生记录，所以root上限不等于row上限。需要各任务保护时先按实际来源制定抽样框；不把全局hash采样叫分层采样。

probe保存N×d memmap，不保存N×P全梯度矩阵；但仍需每条样本的一次反传，并保存一个P维bucket/sign映射。默认拟合8192条不代表其他候选免费：全部候选已经提取梯度，OMP编码也需要计时。加载token记录与code仍有O(N)内存，不能宣称当前实现已经适配十亿记录。

精确greedy的代价随N、准入量和code维度增长；shortlist是成本控制的启发式，不提供全局最优保证。复杂配额下每步还做剩余可行性检查。本轮仅运行小型机制和线路测试，百万级吞吐尚未benchmark。

主选择器约束总response-token预算和每来源row上下限，**没有实现任意每来源token等式约束**。正式实验应在原生数据准备中匹配长度桶，并报告实际来源token差异；回填阶段则严格交换同长度样本。不要把row配额当成完全相同的梯度权重。

## 7. 实现与未完成范围

已实现并运行：本包56项测试、梯度/字典/选样/回填/实际训练计划、80次构造线性训练、小型随机初始化causal model的完整线路。

已提供但未跑真实权重：本地HF/PEFT入口、普通chat编码器、原生factory接口。

尚未完成：DEALRec/GORACS/LESS/MoRec等正式复现，LC-Rec全部派生任务与RQ-VAE，RecLM-gen teacher服务，真实KuaiSAR字段实测，上游packing/DDP/RL及官方大目录解码，真实预训练模型的排名实验。

## 8. 合并与push

把本包中的文件合并到现有rec根目录，不删除任何旧文件。本包没有 `.git/`，没有改远端，也不覆盖 `exp/01_02`。ROOT README可手动加一段链接到 `POOL_SELECTION_README.md`。独立的 `POOL_SELECTION_MANIFEST.json` 只描述本次文件，不重写旧实验的校验清单。

```bash
git status --short
# 先查看新增代码、文档与生成结果，确认没有模型/原始数据/私密路径。
git add src/rec_pool tests/pool docs/pool scripts/pool experiments/pool \
  configs/pool references/pool results/pool_20260909 data/pool_examples \
  POOL_SELECTION_README.md POOL_SELECTION_MANIFEST.json requirements-pool.txt
git commit -m "feat: add PDF-aligned budget-conditioned LLM4Rec selection"
git push origin main
```

Research包直接合并 `llm4rec/revisions/2026-09-09-data-rich/`，不覆盖crystal或旧llm4rec资料。没有自动推送脚本。
