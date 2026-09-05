"""Single-process real-model integration entry point; optional deps/model required."""
import argparse,json,torch
from pathlib import Path
from rec_lab.data import read_records
from rec_lab.hf_adapter import HFFeedbackRanker
from rec_lab.training import train_feedback

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',required=True)
    p.add_argument('--output',required=True);p.add_argument('--method',choices=['uniform','scalar','frontier','atoms_frontier'],default='frontier')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--steps',type=int,default=20)
    p.add_argument('--pool-size',type=int,default=8);p.add_argument('--reference-size',type=int,default=16)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--gradient-filter',default='feedback_head');p.add_argument('--max-gradient-elements',type=int,default=2000000)
    p.add_argument('--lora-rank',type=int,default=8);p.add_argument('--max-length',type=int,default=512)
    p.add_argument('--optimizer',choices=['sgd','adamw'],default='sgd');a=p.parse_args()
    if int(__import__('os').environ.get('WORLD_SIZE','1'))!=1:
        raise RuntimeError('This reference trainer is single-process. Do not launch with torchrun; see scaling protocol.')
    torch.manual_seed(a.seed);data=read_records(a.data)
    model=HFFeedbackRanker(a.model,len(data[0].labels),a.max_length,a.lora_rank,a.device)
    res=train_feedback(model,data,a.method,a.steps,a.pool_size,a.reference_size,a.lr,a.seed,
        None if a.gradient_filter=='ALL' else a.gradient_filter,a.max_gradient_elements,optimizer_name=a.optimizer)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);model.save(out)
    (out/'run.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
if __name__=='__main__':main()
