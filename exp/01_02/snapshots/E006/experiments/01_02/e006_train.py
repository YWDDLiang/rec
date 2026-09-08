"""Matched early-checkpoint continuation with full-pool domain/behavior mixtures."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from rec_lab.checkpointing import CheckpointManager, atomic_json
from rec_lab.domain_mixture import DomainMixtureSelector
from rec_lab.multidomain_sid import MultiDomainSIDLanguageModel
from rec_lab.sid_data import SIDRecord
from e005_train import train_step, evaluate, compact
from e006_groups import ordered_train


GPUS = {5: "GPU-302a7fe1-795f-752c-75b0-67be61c5e513", 6: "GPU-7a4b9355-083c-c33a-a76b-454f0054cf64"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=["fixed", "domain_window", "atoms_window", "cluster_window"], required=True)
    parser.add_argument("--gpu", type=int, choices=[5,6], required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPUS[args.gpu]:
        raise ValueError("Unapproved GPU")
    root = Path(config["run_root"])
    out = root/f"rec_atom_{args.method}"
    if (out/"run.json").exists():
        raise FileExistsError("Method run already started")
    out.mkdir(exist_ok=True)
    train, domain_ids = ordered_train(config)
    partitions = json.loads((root/"partitions.json").read_text())
    sets = {part:{domain:[SIDRecord(**row) for row in rows] for domain,rows in value.items()} for part,value in partitions.items()}
    base = np.zeros(len(train))
    for index in range(len(config["domains"])):
        mask = domain_ids == index
        base[mask] = .5/mask.sum()
    groups = None
    if args.method in ["atoms_window", "cluster_window"]:
        metadata = json.loads((root/"groups/complete.json").read_text())
        order_hash = hashlib.sha256("\n".join(record.record_id for record in train).encode()).hexdigest()
        if order_hash != metadata["record_order_sha256"]:
            raise ValueError("Group membership and training record order differ")
        with np.load(root/"groups/groups.npz") as data:
            groups = data["atoms" if args.method == "atoms_window" else "clusters"].astype(float)
            groups /= groups.sum(1,keepdims=True)
    started = time.perf_counter()
    report = {"experiment":"E006", "method":args.method, "pid":os.getpid(), "physical_gpu":args.gpu,
        "complete":False, "config":config, "trace":[], "stages":[], "windows":[], "final_test_evaluated":False,
        "group_count":len(groups) if groups is not None else 0}
    atomic_json(out/"run.json", report)
    lm = MultiDomainSIDLanguageModel(config["model"],config["data"],adapter=config["adapter"],rank=config["rank"],
        max_length=config["max_length"],seed=config["model_seed"],proxy_dim=config["proxy_dim"],
        gradient_checkpointing=config["gradient_checkpointing"])
    lora = [p for name,p in zip(lm.parameter_names,lm.parameters) if "trainable_tokens" not in name]
    tokens = [p for name,p in zip(lm.parameter_names,lm.parameters) if "trainable_tokens" in name]
    optimizer = torch.optim.AdamW([{"params":lora,"lr":config["lora_lr"]},{"params":tokens,"lr":config["token_lr"]}],
                                  weight_decay=0,foreach=False)
    state = torch.load(config["optimizer_checkpoint"],map_location="cpu",weights_only=False)
    if state["completed_steps"] != config["initial_steps"] or state["optimizer_name"] != "adamw":
        raise ValueError("Anchor optimizer stage differs from configuration")
    optimizer.load_state_dict(state["optimizer"])
    for group in optimizer.param_groups:
        group["foreach"] = False
    del state
    report["initial_tuning"] = evaluate(lm, sets["tuning"], config)
    print("INITIAL",args.method,json.dumps(compact(report["initial_tuning"])),flush=True)
    initial_score = float(np.mean([row["ndcg@10"] for row in report["initial_tuning"].values()]))
    selector = None if args.method == "fixed" else DomainMixtureSelector(lm,optimizer,train,domain_ids,sets["references"],config,args.method,groups)
    manager = CheckpointManager(out)
    sampled = dict.fromkeys(config["domains"],0)
    token_counts = dict.fromkeys(config["domains"],0)
    seen = set()
    microbatch = config["microbatch"]
    step_seconds = 0.
    for step in range(config["steps"]):
        global_step = config["initial_steps"]+step
        progress = max(0.,(global_step-config["warmup_steps"])/(config["schedule_total_steps"]-config["warmup_steps"]))
        factor = config["lr_floor"]+(1-config["lr_floor"])*.5*(1+math.cos(math.pi*progress))
        optimizer.param_groups[0]["lr"] = config["lora_lr"]*factor
        optimizer.param_groups[1]["lr"] = config["token_lr"]*factor
        if step % config["window_steps"] == 0:
            window = step//config["window_steps"]
            weights, info = (base,{"selection_seconds":0.,"query_records":0,"reference_records":0,"domain_mass":[.5,.5],"name":"fixed"}) if selector is None else selector.choose(window)
            info.update(window=window,start_step=step+1,applies_through_step=min(config["steps"],step+config["window_steps"]))
            report["windows"].append(info)
            np.save(out/f"window_{window:03d}_weights.npy", weights.astype(np.float32))
            atomic_json(out/f"window_{window:03d}.json",info)
            print("WINDOW",args.method,json.dumps({k:v for k,v in info.items() if not k.endswith("record_ids")}),flush=True)
        rng = np.random.default_rng(config["seed"]+10007*global_step)
        ids = rng.choice(len(train),config["batch"],p=weights)
        records = [train[int(index)] for index in ids]
        tic = time.perf_counter()
        loss,norm,microbatch = train_step(lm,optimizer,records,microbatch,config["clip_norm"])
        seconds = time.perf_counter()-tic
        step_seconds += seconds
        for record in records:
            sampled[record.task] += 1
            token_counts[record.task] += len(lm.encode(record)[0])
            seen.add(record.record_id)
        report["trace"].append({"step":step+1,"global_step":global_step+1,"loss":loss,"gradient_norm":norm,"seconds":seconds})
        if (step+1)%100 == 0:
            report.update(completed_steps=step+1,elapsed_seconds=time.perf_counter()-started,training_step_seconds=step_seconds,
                sampled_records=dict(sampled),training_tokens=dict(token_counts),unique_records=len(seen),microbatch=microbatch,
                selection_seconds=sum(window["selection_seconds"] for window in report["windows"]))
            atomic_json(out/"run.json",report)
            print("TRAIN",args.method,step+1,loss,report["elapsed_seconds"],flush=True)
        if (step+1)%config["stage_steps"] == 0 or step+1 == config["steps"]:
            ranking = evaluate(lm,sets["tuning"],config)
            nll = {domain:lm.evaluate(rows) for domain,rows in sets["tuning"].items()}
            score = float(np.mean([row["ndcg@10"] for row in ranking.values()]))
            checkpoint = out/"checkpoints"/f"step_{step+1:06d}"
            lm.save(checkpoint/"adapter")
            torch.save({"optimizer_name":"adamw","optimizer":optimizer.state_dict(),"completed_steps":global_step+1},checkpoint/"trainer_state.pt")
            atomic_json(checkpoint/"ready.json",{"producer":"rec_atom","complete":True,"step":step+1,"ranking":ranking})
            report["checkpoint_policy"] = manager.consider(checkpoint,score,min(row["ndcg@10"] for row in ranking.values()))
            stage = {"step":step+1,"global_step":global_step+1,"ranking":ranking,"nll":nll,"selection_score":score,
                     "decision":"Continue frozen protocol" if step+1<config["steps"] else "Review all methods at the large-stage boundary"}
            report["stages"].append(stage)
            atomic_json(out/f"stage_{step+1:06d}.json",stage)
            atomic_json(out/"run.json",report)
            print("STAGE_COMPLETE",args.method,step+1,json.dumps(compact(ranking)),flush=True)
    improved = manager.best_rec_score > initial_score
    selected = str(manager.best_rec/"adapter") if improved else config["adapter"]
    report.update(selected_adapter=selected,selected_initial=not improved)
    del selector,optimizer,lm,lora,tokens
    gc.collect();torch.cuda.empty_cache()
    lm = MultiDomainSIDLanguageModel(config["model"],config["data"],adapter=selected,rank=config["rank"],
        max_length=config["max_length"],seed=config["model_seed"],gradient_checkpointing=False)
    report["audit"] = evaluate(lm,sets["audit"],config)
    report["audit_nll"] = {domain:lm.evaluate(rows) for domain,rows in sets["audit"].items()}
    report.update(complete=True,completed_steps=config["steps"],elapsed_seconds=time.perf_counter()-started,
        training_step_seconds=step_seconds,sampled_records=dict(sampled),training_tokens=dict(token_counts),
        unique_records=len(seen),selection_seconds=sum(row["selection_seconds"] for row in report["windows"]))
    atomic_json(out/"run.json",report)
    print("RUN_COMPLETE",args.method,json.dumps(compact(report["audit"])),flush=True)


if __name__ == "__main__":
    main()
