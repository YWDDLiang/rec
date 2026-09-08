"""Large-stage baseline review. Development evidence, one query per audit user."""
import argparse
import json
from pathlib import Path

import numpy as np

from rec_lab.checkpointing import atomic_json


def paired(left, right, seed=22056):
    a = {row["record_id"]: row for row in left["per_record"]}
    b = {row["record_id"]: row for row in right["per_record"]}
    if a.keys() != b.keys():
        raise ValueError("Paired ranking records differ")
    ids = sorted(a)
    def ndcg(row):
        rank = row["rank"]
        return 1/np.log2(rank+1) if rank is not None and rank <= 10 else 0.
    differences = np.asarray([ndcg(a[key])-ndcg(b[key]) for key in ids])
    rng = np.random.default_rng(seed)
    samples = differences[rng.integers(len(ids), size=(2000, len(ids)))].mean(1)
    return {"ndcg_difference": float(differences.mean()), "interval_95": np.quantile(samples, [.025,.975]).tolist(),
            "users": len(ids), "scope": "Descriptive paired development-user bootstrap; one training seed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root, out = Path(config["run_root"]), Path(config["result_root"])
    out.mkdir(parents=True, exist_ok=True)
    runs = {method: json.loads((root/f"rec_atom_{method}"/"run.json").read_text()) for method in config["methods"]}
    if not all(run["complete"] for run in runs.values()):
        raise ValueError("Baseline stage is not complete")
    comparisons = {}
    for method in ["joint_uniform", "joint_empirical"]:
        comparisons[method] = {domain: paired(runs[method]["audit"][domain], runs[f"single_{domain}"]["audit"][domain])
                               for domain in config["domains"]}
    learned = {domain: min(stage["nll"][domain]["sid_code_nll"] for stage in runs[f"single_{domain}"]["stages"])
               < runs[f"single_{domain}"]["initial_nll"][domain]["sid_code_nll"] for domain in config["domains"]}
    summary = {"experiment": "E005", "complete": True, "runs": {method: {
        "selected_adapter": run["selected_adapter"], "selected_initial": run["selected_initial"],
        "ranking": {d: {k:v for k,v in r.items() if k!="per_record"} for d,r in run["audit"].items()},
        "elapsed_seconds": run["elapsed_seconds"], "training_step_seconds": run["training_step_seconds"],
        "training_tokens": run["training_tokens"], "sampled_records": run["sampled_records"]} for method,run in runs.items()},
        "joint_minus_own_domain_specialist": comparisons, "single_domain_loss_learned": learned,
        "next_stage_decision": "Baseline training is learnable; review joint tradeoffs and start the matched method stage" if all(learned.values())
                               else "Inspect training failure at the large-stage boundary before the method stage",
        "final_test_evaluated": False, "automatic_new_training_started": False}
    atomic_json(out / "summary.json", summary)
    lines = ["# E005 两域基线阶段", "", "单种子开发结果。每域768名用户、每人一个固定查询；最终test未读取。", "",
             "| 配置 | Office R@10 | Office NDCG@10 | Industrial R@10 | Industrial NDCG@10 | 总分钟 |", "|---|---:|---:|---:|---:|---:|"]
    for method, run in runs.items():
        a,b = run["audit"]["office"],run["audit"]["industrial"]
        lines.append(f"| {method} | {a['recall@10']:.6f} | {a['ndcg@10']:.6f} | {b['recall@10']:.6f} | {b['ndcg@10']:.6f} | {run['elapsed_seconds']/60:.2f} |")
    lines += ["", "单域模型的未训练域列仅作诊断；单域与共享模型应按各自声明的使用方式比较。", "",
              "四组更新数和样本使用量一致，实际tokens及每域训练暴露不同；下降不能单凭此表归因于梯度冲突。",
              "来源为MiniOneRec公开数据，自有Qwen3-1.7B LoRA/合法SID SFT；不是其论文全参数SFT/RL复现。", "",
              "阶段决策：" + summary["next_stage_decision"], "", "配对差异、初始化回退与成本详情见summary.json。"]
    (out / "report.md").write_text("\n".join(lines)+"\n")
    print("BASELINE_STAGE_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
