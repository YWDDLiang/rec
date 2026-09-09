# 示例与真实数据的边界

本目录的 `schema_example.jsonl` 仅解释已编码的Record格式，token数字不是任何真实模型的词表。不能直接喂给Qwen或Llama作为实际推荐数据。

可运行人工自回归例子由 `experiments/pool/run_causal_pipeline.py` 自行构造，实际输出及日志保存于 `results/pool_20260909/causal_pipeline/`。真实实验必须从相应原生训练环境导出input_ids、labels与recipe_id。
