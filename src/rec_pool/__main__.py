from __future__ import annotations
import time
import argparse
import json
from pathlib import Path
import numpy as np
from .contracts import read_records,write_records,rows_hash,digest,validate_partitions


def write_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description='PDF-aligned budgeted data selection for causal-LLM recommendation')
    subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('reservoir');p.add_argument('--records',required=True);p.add_argument('--roots',type=int,required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--out',required=True)
    p=subs.add_parser('probe');p.add_argument('--records',required=True);p.add_argument('--model',required=True);p.add_argument('--base-revision',required=True);p.add_argument('--adapter');p.add_argument('--device',default='cpu');p.add_argument('--dimension',type=int,default=512);p.add_argument('--seed',type=int,default=42);p.add_argument('--out',required=True)
    p=subs.add_parser('fit');p.add_argument('--records',required=True);p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--atoms',type=int,default=64);p.add_argument('--sparsity',type=int,default=4);p.add_argument('--fit-limit',type=int,default=8192);p.add_argument('--max-iter',type=int,default=1000);p.add_argument('--seed',type=int,default=42)
    p=subs.add_parser('select');p.add_argument('--records',required=True);p.add_argument('--fit',required=True);p.add_argument('--config',required=True);p.add_argument('--utility');p.add_argument('--mode',choices=['full_code','random','utility'],default='full_code');p.add_argument('--out',required=True)
    p=subs.add_parser('refill');p.add_argument('--records',required=True);p.add_argument('--selected',required=True);p.add_argument('--scores',required=True);p.add_argument('--swaps',type=int,default=500);p.add_argument('--random-control',action='store_true');p.add_argument('--seed',type=int,default=42);p.add_argument('--out',required=True)
    p=subs.add_parser('align');p.add_argument('--records',required=True);p.add_argument('--reference',required=True);p.add_argument('--cache',required=True);p.add_argument('--ref-cache','--reference-cache',dest='ref_cache',required=True);p.add_argument('--fit',required=True);p.add_argument('--task',required=True);p.add_argument('--out',required=True)
    p=subs.add_parser('plan');p.add_argument('--records',required=True);p.add_argument('--selected',required=True);p.add_argument('--budget-tokens',type=int,required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--out',required=True)
    p=subs.add_parser('train');p.add_argument('--records',required=True);p.add_argument('--plan',required=True);p.add_argument('--model',required=True);p.add_argument('--adapter');p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--batch','--batch-size',dest='batch',type=int,default=8);p.add_argument('--microbatch',type=int,default=1);p.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    rows=read_records(args.records)
    if args.command not in {'probe'} and any(r.split!='train' for r in rows):raise ValueError('selector/trainer only accepts TRAIN candidates')
    if args.command=='reservoir':
        from .baselines import nested_root_reservoir
        ids=nested_root_reservoir([r.root_id for r in rows],args.roots,args.seed)
        write_records([rows[i] for i in ids],args.out);return
    if args.command=='probe':
        from .sft import load_hf,extract
        model=load_hf(args.model,device=args.device,adapter_path=args.adapter,seed=args.seed)
        print(json.dumps(extract(model,rows,args.out,dimension=args.dimension,seed=args.seed,base_revision=args.base_revision),ensure_ascii=False));return
    if args.command=='fit':
        from .sft import validate_cache
        from .geometry import fit_dictionary,diagnostics
        meta,raw,_,_,_=validate_cache(rows,args.cache)
        path=Path(args.out);path.mkdir(parents=True,exist_ok=True)
        started=time.perf_counter()
        d,c,fit_ids=fit_dictionary(raw,args.atoms,args.sparsity,args.fit_limit,args.seed,args.max_iter)
        np.save(path/'dictionary.npy',d);np.save(path/'codes.npy',c);np.save(path/'fit_ids.npy',fit_ids)
        write_json(path/'meta.json',{'rows_hash':rows_hash(rows),'cache_meta':meta,'fit_seconds':time.perf_counter()-started,'config':vars(args),
                                   'dictionary_hash':digest(d.tolist()),'codes_hash':digest(c.tolist()),'diagnostics':diagnostics(raw,c,d)})
        return
    if args.command in {'select','align'}:
        path=Path(args.fit);fmeta=json.loads((path/'meta.json').read_text())
        if fmeta['rows_hash']!=rows_hash(rows):raise ValueError('fit/record mismatch')
        d=np.load(path/'dictionary.npy');c=np.load(path/'codes.npy')
        if digest(d.tolist())!=fmeta['dictionary_hash'] or digest(c.tolist())!=fmeta['codes_hash']:raise ValueError('dictionary/codes changed')
    if args.command=='select':
        from .geometry import coordinates
        from .selection import Config,select
        config=Config(**json.loads(Path(args.config).read_text()))
        util=None if not args.utility else np.load(args.utility)
        started=time.perf_counter()
        ids,report=select(coordinates(c,d),np.array([r.cost for r in rows]),[r.cell for r in rows],
                          [r.provenance_group for r in rows],config,util,args.mode)
        report['selection_seconds']=time.perf_counter()-started
        write_json(args.out,{'rows_hash':rows_hash(rows),'indices':ids.tolist(),'record_ids':[rows[i].record_id for i in ids],'report':report,'selection_config':vars(config)});return
    if args.command=='align':
        from .sft import validate_cache,compatible
        from .geometry import alignment
        refs=read_records(args.reference,'select');validate_partitions(rows,refs)
        meta,*_=validate_cache(rows,args.cache);rm,rr,*_=validate_cache(refs,args.ref_cache);compatible(meta,rm)
        if meta!=fmeta['cache_meta']:raise ValueError('fit/cache mismatch')
        ids=[i for i,r in enumerate(refs) if r.task==args.task]
        if not ids:raise ValueError('target task absent from select reference')
        centroid=np.mean(rr[ids],axis=0,keepdims=True)
        if np.linalg.norm(centroid)<1e-12:raise ValueError('near-zero target gradient centroid: partition task before alignment')
        score=alignment(c,d,centroid,True).ravel();Path(args.out).parent.mkdir(parents=True,exist_ok=True);np.save(args.out,score)
        write_json(str(args.out)+'.meta.json',{'task':args.task,'reference_rows':len(ids),'scope':'reconstructed projected cosine, not exact influence/causal effect','reference_rows_hash':rows_hash(refs)})
        return
    if args.command in {'refill','plan'}:
        selected=json.loads(Path(args.selected).read_text())
        if selected['rows_hash']!=rows_hash(rows):raise ValueError('selected IDs belong to another pool')
    if args.command=='refill':
        from .refill import refill
        cfg=selected.get('selection_config',{})
        ids,report=refill(selected['indices'],np.load(args.scores),np.array([r.cost for r in rows]),[r.cell for r in rows],max_swaps=args.swaps,seed=args.seed,random_control=args.random_control,protected_cells=cfg.get('protected',()),groups=[r.provenance_group for r in rows],provenance_cap=cfg.get('provenance_cap'))
        write_json(args.out,{'rows_hash':rows_hash(rows),'indices':ids.tolist(),'record_ids':[rows[i].record_id for i in ids],'report':report,'selection_config':cfg});return
    if args.command=='plan':
        from .sft import schedule
        plan,report=schedule(rows,selected['indices'],args.budget_tokens,args.seed)
        write_json(args.out,{'rows_hash':rows_hash(rows),'plan':plan,'report':report});return
    if args.command=='train':
        import torch
        from .sft import load_hf,train_plan
        obj=json.loads(Path(args.plan).read_text())
        if obj['rows_hash']!=rows_hash(rows):raise ValueError('training plan belongs to another pool')
        out=Path(args.out)
        if out.exists() and any(out.iterdir()):raise FileExistsError('refuse to overwrite training results')
        out.mkdir(parents=True,exist_ok=True)
        model=load_hf(args.model,device=args.device,adapter_path=args.adapter,seed=args.seed)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
        report=train_plan(model,rows,obj['plan'],optimizer,batch_size=args.batch,microbatch_size=args.microbatch)
        model.save_pretrained(out/'adapter');torch.save(optimizer.state_dict(),out/'optimizer.pt')
        write_json(out/'report.json',dict(report,config=vars(args),plan_report=obj['report']))
        return


if __name__=='__main__':main()
