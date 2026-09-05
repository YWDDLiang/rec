"""Summarize actual JSON results; never replace missing runs with fictional data."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    s=json.loads((ROOT/'results/tiny_sft.json').read_text())['results']
    r=json.loads((ROOT/'results/tiny_rl.json').read_text())['runs']
    out={'scope':'synthetic CPU experiments only','sft':{},'rl':{}}
    for m in sorted({x['method'] for x in s}):
        runs=[x for x in s if x['method']==m];a=np.array([x['test']['bce'] for x in runs])
        out['sft'][m]={'seeds':[x['seed'] for x in runs],'mean_test_bce':a.mean(0).tolist(),
            'std_test_bce':a.std(0,ddof=1).tolist(),'mean_elapsed_seconds':float(np.mean([x['elapsed_seconds'] for x in runs]))}
    for m in sorted({x['method'] for x in r}):
        runs=[x for x in r if x['method']==m];a=np.array([x['final_expected_reward'] for x in runs])
        out['rl'][m]={'seeds':[x['seed'] for x in runs],'mean_reward':a.mean(0).tolist(),
            'std_reward':a.std(0,ddof=1).tolist(),'rollouts_per_run':runs[0]['rollouts']}
    f=out['sft']['frontier']['mean_test_bce'];u=out['sft']['uniform']['mean_test_bce']
    out['decision']={'sft_frontier_better_all_objectives':bool(np.all(np.array(f)<u)),
        'real_llm_tested':False,'real_recommendation_benchmark_tested':False,
        'atoms_effect':'same solution as full LP in delivered experiment; additional overhead',
        'conclusion':'do not approve empirical effectiveness from these experiments'}
    (ROOT/'results/summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
