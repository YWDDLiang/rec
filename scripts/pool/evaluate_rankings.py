import argparse,json
from pathlib import Path
from rec_pool.metrics import evaluate_rankings
p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--catalogs',required=True);p.add_argument('--k',type=int,default=10);p.add_argument('--scope',choices=['full-catalog','declared-candidate-set'],required=True);p.add_argument('--out',required=True);a=p.parse_args()
rows=[json.loads(l) for l in Path(a.predictions).read_text().splitlines() if l.strip()]
out=evaluate_rankings(rows,json.loads(Path(a.catalogs).read_text()),a.k,a.scope)
Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(out,ensure_ascii=False,indent=2))
