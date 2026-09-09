"""Bridge a LOCAL, user-audited production encoder to the Record contract.

Factory signature: factory(config: dict) -> iterable[Record or Record-field dict].
The factory runs the SAME tokenizer, prompt, truncation, and loss mask as training.
No arbitrary inference of root identity is made. This is an interface, not an
implementation of every third-party dataset class. Do not load untrusted pickle.
"""
import argparse,importlib,json
from rec_pool.contracts import Record,write_records,validate_rows
p=argparse.ArgumentParser();p.add_argument('--factory',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
module,name=a.factory.split(':',1);factory=getattr(importlib.import_module(module),name)
rows=[]
for item in factory(json.load(open(a.config,encoding='utf-8'))):
    if isinstance(item,Record):rows.append(item)
    else:
        item=dict(item);item['input_ids']=tuple(item['input_ids']);item['labels']=tuple(item['labels']);rows.append(Record(**item))
validate_rows(rows);write_records(rows,a.out)
