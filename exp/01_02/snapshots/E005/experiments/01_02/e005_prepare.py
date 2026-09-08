"""Prepare a two-domain study from pinned train/valid files; final test stays sealed."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from rec_lab.checkpointing import atomic_json
from rec_lab.multidomain_sid import namespace_catalog
from rec_lab.sid_data import SIDRecord, SID_PATTERN, ALL_TOKEN_PATTERN, stable_bucket


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out, run = Path(config["data"]), Path(config["run_root"])
    if out.exists() or run.exists():
        raise FileExistsError("E005 inputs already exist; do not overwrite a study")
    out.mkdir(parents=True)
    run.mkdir(parents=True)
    (run / "logs").mkdir()
    source = Path(config["source"])
    catalog, reports, records, partitions = {}, {}, [], {part: {} for part in ["references", "tuning", "audit"]}
    for domain_index, (domain, upstream) in enumerate(config["domains"].items()):
        index = json.loads((source / "index" / f"{upstream}.index.json").read_text())
        metadata = json.loads((source / "index" / f"{upstream}.item.json").read_text())
        domain_catalog, repairs = namespace_catalog(index, metadata, domain, domain_index)
        catalog.update(domain_catalog)
        atomic_json(out / f"catalog_{domain}.json", domain_catalog)
        rows = {}
        for split in ["train", "valid"]:
            with (source / split / f"{upstream}_5_2016-10-2018-11.csv").open() as stream:
                rows[split] = list(csv.DictReader(stream))
        popularity = Counter(f"{domain}:{row['item_id']}" for row in rows["train"])
        scale = max(math.log1p(value) for value in popularity.values())

        def convert(row, row_index, split):
            history = [f"{domain}:{item}" for item in ast.literal_eval(row["history_item_id"])]
            history = history[-config["max_history"]:]
            target = f"{domain}:{row['item_id']}"
            if not history or any(item not in domain_catalog for item in history + [target]):
                raise ValueError("Empty history or unknown catalog item")
            prompt = (f"Domain: {upstream.replace('_', ' ')}. Recommend the next item for this user. "
                      "Return exactly one item SID.\nEarlier interactions, in chronological order:\n"
                      + " ".join(domain_catalog[item]["sid"] for item in history))
            first = SID_PATTERN.findall(domain_catalog[target]["sid"])[0]
            bucket = stable_bucket(first, 8)
            nuisance = [1., float(domain_index), len(history) / config["max_history"],
                        math.log1p(popularity[target]) / max(scale, 1e-12)]
            nuisance += [float(index == bucket) for index in range(8)]
            return SIDRecord(f"{domain}:{split}:{row_index}", domain, split,
                f"{domain}:{row['user_id']}", target, history, prompt, domain_catalog[target]["sid"], nuisance)

        train = [convert(row, index, "train") for index, row in enumerate(rows["train"])]
        valid_by_user = defaultdict(list)
        for index, row in enumerate(rows["valid"]):
            valid_by_user[row["user_id"]].append(convert(row, index, "selection"))
        # One fixed validation query per user avoids letting active users dominate this pilot.
        valid = [min(values, key=lambda r: hashlib.sha256(f"{config['seed']}:{r.record_id}".encode()).hexdigest())
                 for values in valid_by_user.values()]
        valid.sort(key=lambda r: hashlib.sha256(f"partition:{config['seed']}:{r.user_id}".encode()).hexdigest())
        required = config["reference_users"] + config["tune_users"] + config["audit_users"]
        if len(valid) < required:
            raise ValueError(f"Insufficient independent development users: {domain}")
        cursor = 0
        for part, count in [("tuning", config["tune_users"]), ("references", config["reference_users"]),
                            ("audit", config["audit_users"])]:
            partitions[part][domain] = [asdict(record) for record in valid[cursor:cursor+count]]
            cursor += count
        train_keys = {(r.user_id, tuple(r.history), r.item_id) for r in train}
        valid_keys = {(r.user_id, tuple(r.history), r.item_id) for r in valid}
        if train_keys & valid_keys:
            raise ValueError("Identical training and validation queries overlap")
        records.extend(train)
        records.extend(valid)
        reports[domain] = {"upstream": upstream, "items": len(domain_catalog), "train_records": len(train),
            "train_users": len({r.user_id for r in train}), "valid_rows": len(rows["valid"]),
            "valid_users": len(valid), "collision_repairs": repairs,
            "train_valid_user_overlap": len({r.user_id for r in train} & {r.user_id for r in valid})}
    partition_users = [{row["user_id"] for rows in part.values() for row in rows} for part in partitions.values()]
    if any(left & right for i, left in enumerate(partition_users) for right in partition_users[i+1:]):
        raise AssertionError("Reference, tuning and audit users overlap")
    with (out / "records.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    atomic_json(out / "catalog.json", catalog)
    tokens = sorted({token for item in catalog.values() for token in ALL_TOKEN_PATTERN.findall(item["sid"])})
    manifest = {"domains": list(config["domains"]), "new_tokens": tokens, "counts": reports,
        "source_commit": json.loads((source / "provenance.json").read_text())["commit"],
        "history_limit": config["max_history"], "final_test_downloaded": False, "final_test_evaluated": False,
        "split_protocol": "Preserve upstream train/valid boundaries; one deterministic valid query per user; disjoint tuning/reference/audit users",
        "code_protocol": "Preserve each domain's three semantic codes, domain namespace all code levels, append a deterministic fourth collision code",
        "identity_scope": "Source user/item namespaces are domain-local; no same-person cross-domain claim",
        "temporal_scope": "Upstream chronology retained; exported CSV lacks timestamps, so no independent global timestamp verification",
        "loss": "Four-code mean negative log probability with domain-specific legal-prefix normalization; deterministic suffix contributes zero"}
    atomic_json(out / "manifest.json", manifest)
    atomic_json(run / "partitions.json", partitions)
    atomic_json(run / "config.json", config)
    atomic_json(run / "plan.json", {"experiment": "E005", "stage": "data_ready", "counts": reports,
        "records_sha256": digest(out / "records.jsonl"), "catalog_sha256": digest(out / "catalog.json"),
        "partitions_sha256": digest(run / "partitions.json"), "config_sha256": digest(args.config),
        "complete": False, "test_evaluated": False})
    print("E005_DATA_READY", json.dumps({d: {k: v for k, v in value.items() if k != "collision_repairs"}
                                          for d, value in reports.items()}), flush=True)


if __name__ == "__main__":
    main()
