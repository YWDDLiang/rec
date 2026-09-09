"""Two-pass, O(K)-root-memory reservoir BEFORE tokenization/probing.

Accepts raw or encoded JSONL with an explicit root_id. All variants of selected
roots are emitted; hence a root bound is not necessarily a row bound. This never
samples test/audit rows. Stratify externally by scientifically required cells.
"""
import argparse,heapq,hashlib,json
from pathlib import Path

def subset(path,out,k,seed=42):
    if type(k) is not int or k<1:raise ValueError('invalid root budget')
    if Path(path).resolve()==Path(out).resolve():raise ValueError('input and output must differ')
    heap=[];retained=set();input_rows=0
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            d=json.loads(line);root=d['root_id'];input_rows+=1
            if not isinstance(root,str) or not root or d.get('split')!='train':raise ValueError('need explicit TRAIN root IDs')
            if root in retained:continue
            h=int.from_bytes(hashlib.sha256(f'{seed}|{root}'.encode()).digest(),'big')
            if len(heap)<k:heapq.heappush(heap,(-h,root));retained.add(root)
            elif h < -heap[0][0]:
                _,old=heapq.heapreplace(heap,(-h,root));retained.remove(old);retained.add(root)
    Path(out).parent.mkdir(parents=True,exist_ok=True);output_rows=0
    with Path(path).open(encoding='utf-8') as src,Path(out).open('w',encoding='utf-8') as dst:
        for line in src:
            if line.strip() and json.loads(line)['root_id'] in retained:dst.write(line if line.endswith('\n') else line+'\n');output_rows+=1
    report={'input_rows':input_rows,'retained_roots':len(retained),'output_rows':output_rows,'seed':seed,
            'scope':'bounded roots, all their variants; not guaranteed a bounded row count'}
    Path(str(out)+'.meta.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);p.add_argument('--roots',type=int,required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args();print(json.dumps(subset(a.input,a.out,a.roots,a.seed)))
