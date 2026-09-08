"""One shared, training-only feature pass and frozen executable behavior groups."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from rec_lab.atoms import learn_dictionary, nuisance_residual, omp_encode
from rec_lab.checkpointing import atomic_json
from rec_lab.domain_mixture import kmeans_groups
from rec_lab.multidomain_sid import MultiDomainSIDLanguageModel
from rec_lab.sid_data import load_sid_records


def ordered_train(config):
    records, ids = [], []
    for domain_index, domain in enumerate(config["domains"]):
        rows = load_sid_records(Path(config["data"])/"records.jsonl", "train", domain)
        records.extend(rows)
        ids.extend([domain_index]*len(rows))
    return records, np.asarray(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, choices=[5,6])
    parser.add_argument("--phase", choices=["features", "fit"], required=True)
    args = parser.parse_args()
    allowed = {5: "GPU-302a7fe1-795f-752c-75b0-67be61c5e513", 6: "GPU-7a4b9355-083c-c33a-a76b-454f0054cf64"}
    if args.phase == "features" and os.environ.get("CUDA_VISIBLE_DEVICES") != allowed.get(args.gpu):
        raise ValueError("Unapproved GPU")
    config = json.loads(Path(args.config).read_text())
    out = Path(config["run_root"])/"groups"
    out.mkdir(exist_ok=True)
    if (out/"complete.json").exists():
        raise FileExistsError("Groups already frozen")
    started = time.perf_counter()
    train, domain_ids = ordered_train(config)
    if args.phase == "features":
        if (out/"features_complete.json").exists():
            raise FileExistsError("Feature pass already complete")
        lm = MultiDomainSIDLanguageModel(config["model"], config["data"], adapter=config["adapter"],
            max_length=config["max_length"], rank=config["rank"], seed=config["model_seed"], proxy_dim=config["proxy_dim"],
            gradient_checkpointing=False)
        features = []
        for start in range(0, len(train), 2048):
            values = lm.proxies(train[start:start+2048], batch_size=config["feature_batch"])
            features.append(values)
            atomic_json(out/"progress.json", {"pid": os.getpid(), "physical_gpu": args.gpu, "records": min(start+2048,len(train)),
                "total": len(train), "elapsed_seconds": time.perf_counter()-started})
            print("FEATURES", min(start+2048,len(train)), len(train), flush=True)
        features = np.concatenate(features)
        features /= np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-10)
        np.save(out/"readout_gradient_features.npy", features.astype(np.float32))
        atomic_json(out/"features_complete.json", {"complete": True, "records": len(train), "seconds": time.perf_counter()-started})
        print("FEATURE_PASS_COMPLETE_RELEASING_GPU", flush=True)
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Dictionary fitting must be a CPU-only phase")
    features = np.load(out/"readout_gradient_features.npy")
    feature_seconds = json.loads((out/"features_complete.json").read_text())["seconds"]
    rng = np.random.default_rng(config["seed"])
    fit_ids = np.concatenate([rng.choice(np.flatnonzero(domain_ids == index), config["dictionary_fit_per_domain"], replace=False)
                              for index in range(len(config["domains"]))])
    nuisance = np.asarray([record.nuisance for record in train])
    _, coefficient = nuisance_residual(features[fit_ids], nuisance[fit_ids])
    residual = features-nuisance@coefficient
    scale = max(float(np.sqrt(np.mean(residual[fit_ids]**2))),1e-10)
    residual /= scale
    learned = learn_dictionary(residual[fit_ids], config["atoms"], config["sparsity"], config["dictionary_iterations"], config["seed"])
    codes = omp_encode(residual, learned["dictionary"], config["sparsity"])
    groups, labels = [], []
    for index in range(codes.shape[1]):
        for sign in [1,-1]:
            mass = np.maximum(sign*codes[:,index],0)
            if np.count_nonzero(mass > 1e-12) >= config["minimum_group_support"] and mass.sum() > 1e-10:
                groups.append(mass/mass.sum())
                labels.append(f"atom_{index}_{sign}")
    if len(groups) < 2:
        raise RuntimeError("Fewer than two supported learned groups; inspect at the stage boundary")
    atom_groups = np.asarray(groups, dtype=np.float32)
    cluster_groups, centers = kmeans_groups(residual, fit_ids, len(groups), config["seed"])
    np.savez_compressed(out/"groups.npz", atoms=atom_groups, clusters=cluster_groups, dictionary=learned["dictionary"],
        centers=centers, nuisance_coefficient=coefficient, domain_ids=domain_ids)
    report = {"complete": True, "records": len(train), "atom_groups": len(atom_groups), "cluster_groups": len(cluster_groups),
        "labels": labels, "feature_seconds": feature_seconds, "total_seconds": feature_seconds+time.perf_counter()-started,
        "dictionary_fit_records": len(fit_ids), "dictionary_mse": learned["mse"],
        "record_order_sha256": hashlib.sha256("\n".join(record.record_id for record in train).encode()).hexdigest(),
        "atom_support": [int(np.count_nonzero(row)) for row in atom_groups],
        "atom_ess": [float(1/np.sum(row.astype(float)**2)) for row in atom_groups],
        "representation": "Normalized legal-readout partial-gradient sketch; training-only nuisance fit and signed sparse dictionary. Not an identified user-interest representation.",
        "cost_scope": "Shared feature preparation cost is charged to both gradient-cluster and atom methods; fixed/domain baselines do not require it",
        "scope": "Frozen full-training-pool memberships. Every window re-estimates current full trainable-gradient effects using the same importance-sampled query budget."}
    atomic_json(out/"complete.json", report)
    print("GROUPS_READY", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
