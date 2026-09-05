import argparse,json,time
from pathlib import Path
import torch
from rec_lab.synthetic import synthetic_feedback
from rec_lab.training import TinyFeedbackModel,train_feedback
from rec_lab.data import write_records

def run(output,seeds=(0,1,2),steps=25):
    torch.set_num_threads(1);out=Path(output);out.mkdir(parents=True,exist_ok=True);results=[]
    for seed in seeds:
        data=synthetic_feedback(seed)
        for method in ['uniform','scalar','frontier','atoms_frontier']:
            torch.manual_seed(seed)
            model=TinyFeedbackModel([str(i) for i in range(16)],['0','1'])
            res=train_feedback(model,data,method,steps=steps,pool_size=16,reference_size=48,lr=.15,seed=seed)
            res['evidence']='synthetic, randomly initialized neural recommender, NOT pretrained LLM / real users'
            results.append(res)
            print(seed,method,res['test']['bce'],round(res['elapsed_seconds'],2),flush=True)
    payload={'experiment':'tiny multi-feedback SFT','results':results,'test_used_for_selection':False}
    (out/'tiny_sft.json').write_text(json.dumps(payload,indent=2))
    write_records(out/'synthetic_records.jsonl',synthetic_feedback(0))
    return payload
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results');p.add_argument('--steps',type=int,default=25)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);a=p.parse_args();run(a.output,a.seeds,a.steps)
