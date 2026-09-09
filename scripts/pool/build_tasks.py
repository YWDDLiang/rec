import argparse,json
from pathlib import Path
from rec_pool.data import build_task_rows
p=argparse.ArgumentParser();p.add_argument('--events',required=True);p.add_argument('--catalog',required=True)
p.add_argument('--train-end',type=int,required=True);p.add_argument('--select-end',type=int,required=True);p.add_argument('--out',required=True)
a=p.parse_args();events=[json.loads(l) for l in Path(a.events).read_text().splitlines() if l.strip()]
catalog_rows=[json.loads(l) for l in Path(a.catalog).read_text().splitlines() if l.strip()]
catalog={(r['domain'],r['item_id']):r for r in catalog_rows}
rows=build_task_rows(events,catalog,train_end=a.train_end,select_end=a.select_end)
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
for split in ['train','select','test']:
    with (out/f'{split}.jsonl').open('w',encoding='utf-8') as f:
        for r in rows:
            if r['split']==split:f.write(json.dumps(r,ensure_ascii=False)+'\n')
