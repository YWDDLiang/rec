"""Controlled linear-regression falsification suite, NOT an LLM benchmark.

Four regimes are predeclared. Known sparse coordinates are used, so this checks
selection mechanics, not whether learned atoms outperform raw gradients.
"""
import argparse,json,platform,time
from pathlib import Path
import numpy as np
from rec_pool.selection import Config,select


def run(seed,regime):
    rng=np.random.default_rng(seed);dim=12;budget=72;draws=288
    if regime=='scarce_balanced':axes=np.tile(np.arange(dim),2)
    elif regime=='rich_balanced':axes=np.tile(np.arange(dim),200)
    else:axes=np.concatenate([np.tile(np.arange(2),1080),np.tile(np.arange(2,dim),24)])
    rng.shuffle(axes);n=len(axes);x=np.eye(dim)[axes];y=np.ones(n)
    if regime=='rich_rare_wrong_labels':y[axes>=2]=-1.
    # Gradient at theta=0 for square loss. Identity sparse dictionary is KNOWN.
    z=-y[:,None]*x
    result=[]
    for method in ['all_pool_uniform','random_admission','full_code_coverage','signed_target_coverage']:
        start=time.perf_counter()
        if method=='all_pool_uniform':ids=np.arange(n);status='all_support_available'
        else:
            utility=y.copy() # exact toy clean-reference alignment, not available for free in real data
            cfg=Config(budget,coverage_weight=1.,utility_weight=(2. if method=='signed_target_coverage' else 0.),seed=seed,shortlist=0)
            ids,report=select(z,np.ones(n,dtype=int),['same-task']*n,list(map(str,range(n))),cfg,utility,
                             'random' if method=='random_admission' else 'full_code')
            status=report['status']
        # Same number of gradient updates, same initial model, independent seeded shuffles.
        prng=np.random.default_rng(seed+1000);schedule=[]
        while len(schedule)<draws:schedule.extend(prng.permutation(ids).tolist())
        schedule=schedule[:draws];theta=np.zeros(dim)
        for i in schedule:theta-=.1*(theta@x[i]-y[i])*x[i]
        mse=float(np.mean((theta-np.ones(dim))**2))
        result.append({'seed':seed,'regime':regime,'method':method,'pool_rows':n,
            'admitted_rows':len(ids),'available_direction_count':len(set(axes[ids])),
            'training_occurrences':draws,'unique_trained_rows':len(set(schedule)),
            'test_mse':mse,'status':status,'seconds':time.perf_counter()-start})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/pool_20260909/regimes.json');args=p.parse_args()
    regimes=['scarce_balanced','rich_balanced','rich_redundant','rich_rare_wrong_labels']
    runs=[r for g in regimes for seed in range(5) for r in run(seed,g)]
    summary=[]
    for g in regimes:
        for m in sorted({r['method'] for r in runs}):
            vals=[r['test_mse'] for r in runs if r['regime']==g and r['method']==m]
            summary.append({'regime':g,'method':m,'mean_mse':float(np.mean(vals)),'std_across_seeds':float(np.std(vals,ddof=1))})
    obj={'scope':'constructed linear regression with known sparse directions; NOT pretrained LLM or recommendation evidence',
         'root_pool_budget':{'admission_rows':72,'training_occurrences':288},'runs':runs,'summary':summary,
         'python':platform.python_version(),'numpy':np.__version__}
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2))
    for r in summary:print(r)

if __name__=='__main__':main()
