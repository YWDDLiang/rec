# PDF-aligned LLM4Rec data selection under finite training budgets

本轮更新不以数据体积缩减为目标，不依赖OneReason，也不宣称大数据场景必然应当筛选。默认采用用户PDF的离线流程：同checkpoint的response-only梯度 → 固定投影 → 稀疏atoms → 任务/稀有能力保护下的选择 → 定向/随机回填 → 相同预算SFT。

**优先主场景：LC-Rec/MiniOneRec式多角色生成推荐，RecLM-gen式可控推荐。** KuaiSAR作为跨服务结构补充，不能把其匿名query当原始文本。

研究总览：[docs/pool/REPORT_ZH.md](docs/pool/REPORT_ZH.md)。任务与强基线：[TASKS_AND_BASELINES](docs/pool/TASKS_AND_BASELINES.md)。数学与反例：[MATHEMATICS](docs/pool/MATHEMATICS.md)。逐命令运行：[IMPLEMENTATION](docs/pool/IMPLEMENTATION.md)。真实实验方案：[EXPERIMENT_PROTOCOL](docs/pool/EXPERIMENT_PROTOCOL.md)。证据：[VALIDATION](docs/pool/VALIDATION.md)；审计：[AUDIT](docs/pool/AUDIT.md)。

新增namespace为 `rec_pool`，与原 `rec_lab` 并存；不修改任何E001–E006冻结结果。直接合并到现有仓库后安装requirements，或设置PYTHONPATH使用。

```bash
python -m pip install -r requirements-pool.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/pool/run_tests.py
python experiments/pool/run_regimes.py
python experiments/pool/run_causal_pipeline.py
```

本轮CPU机制与线路测试不等于真实LLM推荐实验。HF入口需要用户本地模型、原生数据编码和合法目录；第三方baseline只登记了复现协议，没有虚构运行结果。代码包没有模型权重、原始用户日志或任何GitHub写入操作。
