"""Method-stage report with per-domain effects and full preparation/selection costs."""
import argparse
import json
from pathlib import Path

from rec_lab.checkpointing import atomic_json
from e005_compare import paired


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    root,out=Path(config["run_root"]),Path(config["result_root"])
    out.mkdir(parents=True,exist_ok=True)
    runs={method:json.loads((root/f"rec_atom_{method}"/"run.json").read_text()) for method in config["methods"]}
    if not all(run["complete"] for run in runs.values()):
        raise ValueError("Method stage incomplete")
    representation=json.loads((root/"groups/complete.json").read_text())
    differences={method:{domain:paired(runs["atoms_window"]["audit"][domain],runs[method]["audit"][domain],config["seed"])
                         for domain in config["domains"]} for method in ["fixed","domain_window","cluster_window"]}
    both_domain_point_gains=all(row["ndcg_difference"]>0 for comparison in differences.values() for row in comparison.values())
    summary={"experiment":"E006","complete":True,"runs":{method:{
        "selected_adapter":run["selected_adapter"],"selected_initial":run["selected_initial"],
        "ranking":{domain:{key:value for key,value in row.items() if key!="per_record"} for domain,row in run["audit"].items()},
        "elapsed_seconds":run["elapsed_seconds"],"training_step_seconds":run["training_step_seconds"],
        "selection_seconds":run["selection_seconds"],"training_tokens":run["training_tokens"],"sampled_records":run["sampled_records"],
        "representation_seconds_to_charge":representation["total_seconds"] if method in ["atoms_window","cluster_window"] else 0.}
        for method,run in runs.items()},"atoms_minus_controls":differences,"representation":representation,
        "next_stage_decision":"Positive per-domain point estimates versus all controls: review uncertainty, then prioritize fixed-domain-mass A/B replacement and 02-only controls"
                              if both_domain_point_gains else "Primary joint gain not established: inspect the completed-stage comparison before changing representation, mixture objective or training horizon",
        "final_test_evaluated":False,"automatic_new_training_started":False}
    atomic_json(out/"summary.json",summary)
    lines=["# E006 两域方法对照","","同一800步均衡混训起点，继续1600步；单种子开发证据，最终test未读取。","",
           "| 方法 | Office R@10 | Office NDCG@10 | Industrial R@10 | Industrial NDCG@10 | 训练分钟 | 选择分钟 | 表示准备分钟 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method,run in runs.items():
        a,b=run["audit"]["office"],run["audit"]["industrial"]
        extra=representation["total_seconds"]/60 if method in ["atoms_window","cluster_window"] else 0.
        lines.append(f"| {method} | {a['recall@10']:.6f} | {a['ndcg@10']:.6f} | {b['recall@10']:.6f} | {b['ndcg@10']:.6f} | {run['training_step_seconds']/60:.2f} | {run['selection_seconds']/60:.2f} | {extra:.2f} |")
    lines += ["","原子与普通梯度簇保持精确1:1领域概率；领域基线只改领域概率。实际抽样数量仍有随机波动。",
              "表示准备在本次队列中共享执行一次，但方法成本比较应分别为两种依赖表示的方法计入这项成本。",
              "未证明读出梯度草图对应个性化行为；组数、作用估计、有限候选选择和实际训练效果均有边界。",
              "若多组选择同一起点，其零差异来自同一模型，不构成策略等效证据。","","阶段决策："+summary["next_stage_decision"],
              "","逐域配对差异与描述性区间见summary.json。不能仅凭单个开发种子宣称普遍互补或超过外部论文。"]
    (out/"report.md").write_text("\n".join(lines)+"\n")
    print("E006_METHOD_STAGE_COMPLETE",json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
