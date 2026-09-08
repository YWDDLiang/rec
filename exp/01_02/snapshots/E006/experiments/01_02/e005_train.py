"""Finite, matched-backbone two-domain baselines; decisions occur at stage boundaries."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from rec_lab.checkpointing import CheckpointManager, atomic_json
from rec_lab.multidomain_sid import MultiDomainSIDLanguageModel
from rec_lab.sid_data import SIDRecord, load_sid_records


GPUS = {5: "GPU-302a7fe1-795f-752c-75b0-67be61c5e513", 6: "GPU-7a4b9355-083c-c33a-a76b-454f0054cf64"}


def train_step(lm, optimizer, records, microbatch, clip_norm):
    while True:
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        try:
            lm.model.train()
            for start in range(0, len(records), microbatch):
                loss = lm.losses(records[start:start+microbatch]).sum() / len(records)
                loss.backward()
                total_loss += float(loss.detach())
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            loss = None
            gc.collect()
            torch.cuda.empty_cache()
            if microbatch == 1:
                raise
            microbatch //= 2
            lm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            lm.model.enable_input_require_grads()
            print("OOM_RECOVERY_SAME_RECORDS", microbatch, flush=True)
            continue
        norm = torch.nn.utils.clip_grad_norm_(lm.parameters, clip_norm)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite update norm")
        optimizer.step()
        return total_loss, float(norm), microbatch


def compact(rows):
    return {domain: {key: value for key, value in row.items() if key != "per_record"}
            for domain, row in rows.items()}


def evaluate(lm, sets, config):
    return {domain: lm.rank_domain(records, domain, beams=config["eval_beams"], batch_size=config["eval_batch"])
            for domain, records in sets.items()}


def objective(ranking, method):
    if method.startswith("single_"):
        return ranking[method.removeprefix("single_")]["ndcg@10"]
    return float(np.mean([row["ndcg@10"] for row in ranking.values()]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=["smoke", "single_office", "single_industrial", "joint_uniform", "joint_empirical"], required=True)
    parser.add_argument("--gpu", type=int, choices=[5, 6], required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPUS[args.gpu]:
        raise ValueError("Unapproved GPU")
    root = Path(config["run_root"])
    out = root / f"rec_atom_{args.method}"
    if (out / "run.json").exists():
        raise FileExistsError("Do not restart or overwrite an existing run")
    out.mkdir(exist_ok=True)
    domains = list(config["domains"])
    train = {domain: load_sid_records(Path(config["data"]) / "records.jsonl", "train", domain) for domain in domains}
    partitions = json.loads((root / "partitions.json").read_text())
    sets = {part: {domain: [SIDRecord(**record) for record in rows] for domain, rows in values.items()}
            for part, values in partitions.items()}
    if args.method == "smoke":
        config = {**config, "steps": 24, "stage_steps": 24, "batch": 16, "microbatch": 16, "eval_batch": 4}
        sets = {part: {domain: rows[:16] for domain, rows in values.items()} for part, values in sets.items()}
    if args.method.startswith("single_"):
        probabilities = np.asarray([float(domain == args.method.removeprefix("single_")) for domain in domains])
    elif args.method == "joint_empirical":
        probabilities = np.asarray([len(train[domain]) for domain in domains], dtype=float)
        probabilities /= probabilities.sum()
    else:
        probabilities = np.ones(len(domains)) / len(domains)
    started = time.perf_counter()
    report = {"experiment": "E005", "method": args.method, "pid": os.getpid(), "physical_gpu": args.gpu,
        "config": config, "domain_probabilities": dict(zip(domains, probabilities.tolist())), "complete": False,
        "final_test_evaluated": False, "trace": [], "stages": [], "initialization": config["initialization"]}
    atomic_json(out / "run.json", report)
    lm = MultiDomainSIDLanguageModel(config["model"], config["data"], rank=config["rank"],
        max_length=config["max_length"], seed=config["model_seed"], gradient_checkpointing=config["gradient_checkpointing"])
    lora = [p for name, p in zip(lm.parameter_names, lm.parameters) if "trainable_tokens" not in name]
    tokens = [p for name, p in zip(lm.parameter_names, lm.parameters) if "trainable_tokens" in name]
    optimizer = torch.optim.AdamW([{"params": lora, "lr": config["lora_lr"]},
        {"params": tokens, "lr": config["token_lr"]}], weight_decay=0, foreach=False)
    report["gradient_elements"] = lm.gradient_elements
    initial_adapter = out / "initial_adapter"
    lm.save(initial_adapter)
    report["initial_tuning"] = evaluate(lm, sets["tuning"], config)
    report["initial_nll"] = {domain: lm.evaluate(rows) for domain, rows in sets["tuning"].items()}
    print("INITIAL", args.method, json.dumps(compact(report["initial_tuning"])), flush=True)
    manager = CheckpointManager(out)
    microbatch = config["microbatch"]
    sampled = dict.fromkeys(domains, 0)
    token_counts = dict.fromkeys(domains, 0)
    seen = {domain: set() for domain in domains}
    step_seconds = 0.
    fixed_smoke = [train[domain][i] for domain in domains for i in range(8)]
    smoke_before = lm.evaluate(fixed_smoke)["sid_code_nll"] if args.method == "smoke" else None
    for step in range(config["steps"]):
        progress = max(0., (step-config["warmup_steps"]) / max(1, config["steps"]-config["warmup_steps"]))
        factor = min(1., (step+1)/config["warmup_steps"]) if step < config["warmup_steps"] else config["lr_floor"] + (1-config["lr_floor"])*.5*(1+math.cos(math.pi*progress))
        optimizer.param_groups[0]["lr"] = config["lora_lr"]*factor
        optimizer.param_groups[1]["lr"] = config["token_lr"]*factor
        rng = np.random.default_rng(config["seed"] + step*10007)
        source_ids = rng.choice(len(domains), config["batch"], p=probabilities)
        records = [train[domains[int(index)]][int(rng.integers(len(train[domains[int(index)]])))] for index in source_ids]
        if args.method == "smoke":
            records = fixed_smoke
        tic = time.perf_counter()
        loss, norm, microbatch = train_step(lm, optimizer, records, microbatch, config["clip_norm"])
        duration = time.perf_counter()-tic
        step_seconds += duration
        for record in records:
            sampled[record.task] += 1
            token_counts[record.task] += len(lm.encode(record)[0])
            seen[record.task].add(record.record_id)
        report["trace"].append({"step": step+1, "loss": loss, "gradient_norm": norm, "seconds": duration})
        if (step+1) % 100 == 0:
            report.update(completed_steps=step+1, elapsed_seconds=time.perf_counter()-started,
                training_step_seconds=step_seconds, sampled_records=dict(sampled), training_tokens=dict(token_counts),
                unique_records={domain: len(values) for domain, values in seen.items()}, microbatch=microbatch,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
            atomic_json(out / "run.json", report)
            print("TRAIN", args.method, step+1, loss, report["elapsed_seconds"], flush=True)
        if (step+1) % config["stage_steps"] == 0 or step+1 == config["steps"]:
            ranking = evaluate(lm, sets["tuning"], config)
            nll = {domain: lm.evaluate(rows) for domain, rows in sets["tuning"].items()}
            checkpoint = out / "checkpoints" / f"step_{step+1:06d}"
            lm.save(checkpoint / "adapter")
            torch.save({"optimizer_name": "adamw", "optimizer": optimizer.state_dict(), "completed_steps": step+1}, checkpoint / "trainer_state.pt")
            atomic_json(checkpoint / "ready.json", {"producer": "rec_atom", "complete": True, "step": step+1, "ranking": ranking})
            score = objective(ranking, args.method)
            report["checkpoint_policy"] = manager.consider(checkpoint, score, min(row["ndcg@10"] for row in ranking.values()))
            stage = {"step": step+1, "ranking": ranking, "nll": nll, "selection_score": score,
                "sampled_records": dict(sampled), "training_tokens": dict(token_counts),
                "decision": "Continue frozen protocol to the next stage" if step+1 < config["steps"] else "Baseline run complete; compare at the large-stage boundary"}
            report["stages"].append(stage)
            atomic_json(out / f"stage_{step+1:06d}.json", stage)
            atomic_json(out / "run.json", report)
            print("STAGE_COMPLETE", args.method, step+1, json.dumps(compact(ranking)), flush=True)
    if args.method == "smoke":
        smoke_after = lm.evaluate(fixed_smoke)["sid_code_nll"]
        report["smoke"] = {"fixed_batch_nll_before": smoke_before, "fixed_batch_nll_after": smoke_after,
            "training_learned": smoke_after < smoke_before, "domains": domains}
        if not smoke_after < smoke_before:
            raise RuntimeError("Two-domain fixed-batch learning check failed")
    improved = manager.best_rec_score > objective(report["initial_tuning"], args.method)
    selected_adapter = str(manager.best_rec / "adapter") if improved else str(initial_adapter)
    report.update(selected_adapter=selected_adapter, selected_initial=not improved)
    del optimizer, lm, lora, tokens
    gc.collect()
    torch.cuda.empty_cache()
    lm = MultiDomainSIDLanguageModel(config["model"], config["data"], adapter=selected_adapter,
        rank=config["rank"], max_length=config["max_length"], seed=config["model_seed"], gradient_checkpointing=False)
    report["audit"] = evaluate(lm, sets["audit"], config)
    report["audit_nll"] = {domain: lm.evaluate(rows) for domain, rows in sets["audit"].items()}
    report.update(complete=True, completed_steps=config["steps"], elapsed_seconds=time.perf_counter()-started,
        training_step_seconds=step_seconds, sampled_records=dict(sampled), training_tokens=dict(token_counts),
        unique_records={domain: len(values) for domain, values in seen.items()}, microbatch=microbatch)
    atomic_json(out / "run.json", report)
    print("RUN_COMPLETE", args.method, json.dumps(compact(report["audit"])), flush=True)


if __name__ == "__main__":
    main()
