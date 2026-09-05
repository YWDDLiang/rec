"""Canonical timestamp-safe multi-feedback records; no fabricated negatives."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json, math, csv

@dataclass
class Record:
    record_id: str
    user_id: str
    history: list[str]
    history_times: list[float]
    timestamp: float
    item_id: str
    scene: str
    labels: list[float | None]
    split: str
    prompt: str = ""
    propensity: float | None = None
    source: str = "unspecified"

    def validate(self):
        if self.split not in {'train','selection','test'}: raise ValueError('split must be train/selection/test')
        if not self.record_id or not self.user_id or not self.item_id: raise ValueError('missing identifiers')
        if not math.isfinite(self.timestamp): raise ValueError('non-finite time')
        if len(self.history)!=len(self.history_times): raise ValueError('history/time mismatch')
        if any(not math.isfinite(t) or t>=self.timestamp for t in self.history_times):
            raise ValueError('history must strictly precede target; same timestamp is not ordered')
        if self.history_times!=sorted(self.history_times): raise ValueError('history not sorted')
        if not self.labels or all(v is None for v in self.labels): raise ValueError('record has no observed objective')
        if any(v is not None and (not math.isfinite(v) or not 0<=v<=1) for v in self.labels):
            raise ValueError('binary-feedback labels must lie in [0,1]; unknown is null')
        if self.propensity is not None and (not math.isfinite(self.propensity) or not 0<self.propensity<=1):
            raise ValueError('invalid logged propensity')
        return self

def read_records(path):
    out=[]
    with Path(path).open(encoding='utf-8') as f:
        for lineno,line in enumerate(f,1):
            if not line.strip(): continue
            try: out.append(Record(**json.loads(line)).validate())
            except Exception as e: raise ValueError(f'{path}:{lineno}: {e}') from e
    validate_splits(out)
    return out

def write_records(path,records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for r in records: f.write(json.dumps(asdict(r.validate()),ensure_ascii=False)+'\n')

def validate_splits(records,require_all=True):
    if not records: raise ValueError('empty data')
    for record in records: record.validate()
    ids=[r.record_id for r in records]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate record across splits')
    dims={len(r.labels) for r in records}
    if len(dims)!=1: raise ValueError('objective schema changed')
    by={k:[r for r in records if r.split==k] for k in ('train','selection','test')}
    if require_all and any(not v for v in by.values()): raise ValueError('all three splits required')
    for a,b in [('train','selection'),('selection','test'),('train','test')]:
        if by[a] and by[b] and max(r.timestamp for r in by[a])>=min(r.timestamp for r in by[b]):
            raise ValueError(f'global temporal boundary overlaps: {a}/{b}')
    return by

def feedback_prompt(record, item_text=None):
    """Construct prompt from allowed context, never labels or logged propensity.

    Any caller-supplied external text needs its own availability-time audit.
    An arbitrary `Record.prompt` is ignored by default to prevent answer leakage.
    """
    txt={} if item_text is None else item_text
    history='; '.join(txt.get(i,i) for i in record.history)
    candidate=txt.get(record.item_id,record.item_id)
    return f'Scene: {record.scene}\nEarlier items: {history}\nCandidate: {candidate}\nPredict user feedback.'

def prepare_kuairand(csv_paths,output,train_end,selection_end,max_history=20,max_rows=200000):
    """Local, memory-bounded official log adapter. IDs stay strings.

    Only is_click and is_like are read as objectives; is_click has UI-dependent
    meaning. No monthly statistics are loaded. Random flag is NOT a propensity.
    Files should cover complete retained histories; Pure has incomplete sequences.
    """
    if train_end>=selection_end or max_history<1: raise ValueError('invalid boundaries')
    rows=[]
    for path in csv_paths:
        with Path(path).open(newline='',encoding='utf-8') as f:
            reader=csv.DictReader(f)
            required={'user_id','video_id','time_ms','tab','is_click','is_like'}
            if not required.issubset(reader.fieldnames or []): raise ValueError(f'missing columns in {path}')
            for row in reader:
                rows.append(row)
                if len(rows)>max_rows: raise ValueError('max_rows exceeded; partition explicitly, do not silently truncate')
    rows.sort(key=lambda r:float(r['time_ms']))
    from itertools import groupby
    histories={};out=[];seen=set()
    # All events at the same timestamp observe the same strictly earlier
    # history. Trimming after each tied event would erase older valid context.
    for t,tied_rows in groupby(rows,key=lambda r:float(r['time_ms'])):
        if not math.isfinite(t): raise ValueError('non-finite source timestamp')
        block=list(tied_rows);pending={}
        for row in block:
            u=row['user_id'];i=row['video_id'];s=row['tab']
            key=(u,i,t,s)
            if key in seen: raise ValueError('duplicate source event; resolve provenance before conversion')
            seen.add(key)
            past=histories.get(u,[])[-max_history:]
            split='train' if t<train_end else 'selection' if t<selection_end else 'test'
            labels=[None if row[k].strip() in {'','-1','nan','None'} else float(row[k]) for k in ('is_click','is_like')]
            if any(x is not None for x in labels):
                out.append(Record(str(len(out)),u,[x[0] for x in past],[x[1] for x in past],t,i,s,labels,split,
                    source='KuaiRand local official logs; propensities not supplied'))
            pending.setdefault(u,[]).append((i,t))
        for u,events in pending.items():
            histories[u]=(histories.get(u,[])+events)[-max_history:]
    validate_splits(out);write_records(output,out)
    return {'records':len(out),'output':str(output),'objectives':['UI-dependent click/valid-play','like'],
            'warning':'Observational training only; no inferred propensities, no causal retention claim.'}
