from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                   separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class Record:
    record_id: str
    root_id: str
    task: str
    domain: str
    source: str
    split: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    recipe_id: str
    user_id: str | None = None
    label_origin: str = 'observed'

    def __post_init__(self):
        for key in ['record_id','root_id','task','domain','source','recipe_id','label_origin']:
            if not isinstance(getattr(self,key),str) or not getattr(self,key).strip():
                raise ValueError(f'empty or nonstring {key}')
        if self.split not in {'train','select','tune','audit','test'}:
            raise ValueError('unknown split')
        if self.user_id is not None and (not isinstance(self.user_id,str) or not self.user_id):
            raise ValueError('invalid user ID')
        if len(self.input_ids)<2 or len(self.labels)!=len(self.input_ids):
            raise ValueError('token/label length mismatch')
        if any(type(t) is not int or t<0 for t in self.input_ids):
            raise ValueError('invalid input token')
        if any(type(y) is not int or (y!=-100 and y!=x) for x,y in zip(self.input_ids,self.labels)):
            raise ValueError('labels must equal supervised input tokens or -100')
        if self.labels[0]!=-100 or self.cost==0:
            raise ValueError('first token must be masked; need a nonempty causal response')

    @property
    def cost(self): return sum(t!=-100 for t in self.labels[1:])
    @property
    def cell(self): return json.dumps([self.task,self.domain], ensure_ascii=False,separators=(',',':'))
    @property
    def content_id(self): return digest([self.input_ids,self.labels,self.recipe_id])
    @property
    def provenance_group(self): return digest([self.root_id,self.task,self.domain])


def read_records(path, split=None):
    rows=[]
    with Path(path).open(encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            if not line.strip(): continue
            try:
                obj=json.loads(line); obj['input_ids']=tuple(obj['input_ids']);obj['labels']=tuple(obj['labels'])
                row=Record(**obj)
                if split and row.split!=split: raise ValueError(f'expected {split}')
                rows.append(row)
            except Exception as exc: raise ValueError(f'{path}:{i}: {exc}') from exc
    validate_rows(rows)
    return rows


def validate_rows(rows):
    if not rows or len({r.record_id for r in rows})!=len(rows):
        raise ValueError('empty data or duplicate record IDs')
    if len({r.recipe_id for r in rows})!=1: raise ValueError('mixed tokenizer/template recipes')


def validate_partitions(*parts):
    ids=set();roots=set();hashes=set();recipes=set()
    for rows in parts:
        validate_rows(rows)
        if len({r.split for r in rows})!=1: raise ValueError('mixed partition')
        a={r.record_id for r in rows};b={r.root_id for r in rows};c={r.content_id for r in rows}
        if ids&a or roots&b or hashes&c: raise ValueError('cross-partition identity or exact-content leakage')
        ids|=a;roots|=b;hashes|=c;recipes|={r.recipe_id for r in rows}
    if len(recipes)!=1: raise ValueError('different recipes across partitions')
    # Shared known users across TRAIN and temporal evaluation are allowed.
    # Repeated adaptive use of select/tune requires independent user blocks.
    seen={}
    for rows in parts:
        for r in rows:
            if r.split!='train' and r.user_id:
                if seen.setdefault(r.user_id,r.split)!=r.split:
                    raise ValueError('nontrain user overlap: supply an explicit independent split')


def write_records(rows,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(asdict(r),ensure_ascii=False)+'\n')


def rows_hash(rows):
    return digest([(r.record_id,r.root_id,r.cell,r.split,r.content_id) for r in rows])
