import argparse,json
from pathlib import Path
from rec_pool.data import normalize_amazon23
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--domain',required=True)
p.add_argument('--timestamp-field',default='timestamp');p.add_argument('--output',required=True)
a=p.parse_args();Path(a.output).parent.mkdir(parents=True,exist_ok=True)
with open(a.output,'w',encoding='utf-8') as f:
    for r in normalize_amazon23(a.input,a.domain,a.timestamp_field):f.write(json.dumps(r)+'\n')
