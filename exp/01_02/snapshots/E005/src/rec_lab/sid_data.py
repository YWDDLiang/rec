"""Frozen-catalog generative SID data; official per-user chronology is retained."""
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter
import hashlib,json,re

SID_PATTERN=re.compile(r"<s_[a-d]_\d+>")
ALL_TOKEN_PATTERN=re.compile(r"<[^<>]+>")

@dataclass
class SIDRecord:
    record_id: str
    task: str
    split: str
    user_id: str
    item_id: str
    history: list
    prompt: str
    answer: str
    nuisance: list

def text_field(value):
    if isinstance(value,list):
        return "; ".join(text_field(v) for v in value)
    return str(value or "").strip()

def stable_bucket(value,buckets=1000):
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:12],16)%buckets

def rec_prompt(history,catalog):
    return ("Recommend the next item for this user. Return exactly one item SID.\n"
            "Earlier interactions, in chronological order:\n"
            + " ".join(catalog[i]["sid"] for i in history))

def grounding_prompt(item):
    return ("Identify the item from its description. Return exactly one item SID.\n"
            f"Title: {text_field(item.get('title'))}\n"
            f"Categories: {text_field(item.get('categories'))}\n"
            f"Description: {text_field(item.get('description'))[:320]}")

def repair_sid_collisions(catalog):
    """Deterministic catalog-only tie breaking; never consult interactions."""
    groups={}
    used={}
    for item_id,item in catalog.items():
        tokens=SID_PATTERN.findall(item["sid"])
        prefix=tuple(tokens[:3])
        used.setdefault(prefix,set()).add(int(tokens[3].split("_")[-1][:-1]))
        groups.setdefault(item["sid"],[]).append(item_id)
    changes=[]
    for sid,ids in sorted(groups.items()):
        if len(ids)<2:
            continue
        tokens=SID_PATTERN.findall(sid);prefix=tuple(tokens[:3])
        for item_id in sorted(ids)[1:]:
            available=[j for j in range(256) if j not in used[prefix]]
            if not available:
                raise ValueError("No free fourth-level code for collision repair")
            replacement=f"<s_d_{available[0]}>";used[prefix].add(available[0])
            new_sid=sid.replace(tokens[3],replacement)
            catalog[item_id]["sid"]=new_sid
            changes.append({"item_id":item_id,"original_sid":sid,"repaired_sid":new_sid})
    return changes

def build_beauty(source,output,max_history=20):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((source/"Beauty.pretrain.json").read_text(encoding="utf-8"))
    histories=[]
    for line in (source/"sequential_data_processed.txt").read_text().splitlines():
        fields=line.split()
        if len(fields)>=6:
            histories.append((fields[0],fields[1:]))
    missing={i for _,h in histories for i in h if i not in catalog}
    if missing:
        raise ValueError(f"Unknown item IDs: {len(missing)}")
    for i,item in catalog.items():
        if len(SID_PATTERN.findall(item["sid"]))!=4:
            raise ValueError(f"Non-four-level SID for item {i}")
    sid_count=Counter(item["sid"] for item in catalog.values())
    collision_count=sum(v-1 for v in sid_count.values())
    repairs=repair_sid_collisions(catalog)
    if len({item["sid"] for item in catalog.values()})!=len(catalog):
        raise ValueError("Collision repair failed")
    (output/"sid_collision_repairs.json").write_text(json.dumps(repairs,indent=2),encoding="utf-8")
    popularity=Counter(i for _,h in histories for i in h[:-2])
    pop_scale=max(1.,max(__import__("math").log1p(v) for v in popularity.values()))
    def nuisance(task,item_id,history):
        import math
        first=SID_PATTERN.findall(catalog[item_id]["sid"])[0]
        bucket=stable_bucket(first,8)
        return [1.,float(task=="rec"),len(history)/max_history,
                math.log1p(popularity[item_id])/pop_scale]+[float(j==bucket) for j in range(8)]
    records=[]
    for user,h in histories:
        for split,index in [("train",len(h)-3),
                            ("selection" if stable_bucket(user,2)==0 else "calibration",len(h)-2),
                            ("test",len(h)-1)]:
            item=h[index];past=h[max(0,index-max_history):index]
            records.append(SIDRecord(f"rec:{user}:{index}","rec",split,user,item,past,
                rec_prompt(past,catalog),catalog[item]["sid"],nuisance("rec",item,past)))
    for item_id,item in sorted(catalog.items()):
        bucket=stable_bucket("semantic:"+item_id)
        split="train" if bucket<700 else "selection" if bucket<800 else "calibration" if bucket<900 else "test"
        prompt=grounding_prompt(item)
        if item["sid"] in prompt or SID_PATTERN.search(prompt):
            raise ValueError("SID leaked into metadata prompt")
        records.append(SIDRecord("ground:"+item_id,"ground",split,"",item_id,[],
            prompt,item["sid"],nuisance("ground",item_id,[])))
    counts=Counter((r.task,r.split) for r in records)
    for task in ["rec","ground"]:
        for split in ["train","selection","calibration","test"]:
            if not counts[(task,split)]:raise ValueError("Empty task/split")
    with (output/"records.jsonl").open("w",encoding="utf-8") as f:
        for r in records:f.write(json.dumps(asdict(r),ensure_ascii=False)+"\n")
    (output/"catalog.json").write_text(json.dumps(catalog,ensure_ascii=False),encoding="utf-8")
    tokens=sorted({token for item in catalog.values() for token in ALL_TOKEN_PATTERN.findall(item["sid"])})
    report={"source_commit":"4c573ac6dd6e7d8a1ee4152083e4e07928de1d92",
            "items":len(catalog),"users":len(histories),"sid_collisions_original":collision_count,"sid_collisions_after_repair":0,"repaired_item_count":len(repairs),
            "new_tokens":tokens,"counts":{f"{a}/{b}":v for (a,b),v in counts.items()},
            "history_limit":max_history,
            "split_protocol":"official per-user last-two chronology; validation users split into selection/calibration; semantic split by item hash",
            "not_claimed":["global temporal split without raw timestamps","verified purchases/exposures","unseen-item zero-shot generalization"],
            "semantic_protocol":"held-out item descriptions on a known item catalog; recommendation training may contain those item IDs",
            "test_evaluated":False}
    (output/"manifest.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

def load_sid_records(path,split,task=None):
    if split not in {"train","selection","calibration","test"}:
        raise ValueError("invalid split")
    records=[]
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            obj=json.loads(line)
            if obj["split"]==split and (task is None or obj["task"]==task):
                records.append(SIDRecord(**obj))
    return records
