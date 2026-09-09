"""Real autograd, sparse fitting, selection and training on a random tiny causal LM.
No pretrained weights or benchmark performance claims. This is a plumbing test.
"""
from pathlib import Path
from types import SimpleNamespace
import json,copy,time
import numpy as np
import torch
from rec_pool.contracts import Record,write_records
from rec_pool.sft import extract,schedule,train_plan,trainable_hash
from rec_pool.__main__ import main as cli

class TinyCausal(torch.nn.Module):
    def __init__(self):
        super().__init__();self.emb=torch.nn.Embedding(32,12);self.proj=torch.nn.Linear(12,32)
    def forward(self,input_ids,attention_mask=None):
        h=self.emb(input_ids);h=h*(attention_mask.unsqueeze(-1) if attention_mask is not None else 1)
        h=h.cumsum(1)/torch.arange(1,h.shape[1]+1,device=h.device)[None,:,None]
        return SimpleNamespace(logits=self.proj(h))

def main():
    root=Path('results/pool_20260909/causal_pipeline');root.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(13);model=TinyCausal();initial=trainable_hash(model)
    rows=[]
    for i in range(48):
        inp=(1,3+i%12,4+(i//12));ans=(18+i%4,2)
        rows.append(Record(str(i),f'root-{i}','next_item' if i%2 else 'item_grounding','a','toy','train',inp+ans,(-100,)*3+ans,'tiny-common-recipe'))
    write_records(rows,root/'train.jsonl');extract(model,rows,root/'cache',dimension=16,base_revision='random-tiny-not-pretrained',overwrite=True)
    cli(['fit','--records',str(root/'train.jsonl'),'--cache',str(root/'cache'),'--out',str(root/'fit'),'--atoms','8','--sparsity','4','--max-iter','20'])
    (root/'config.json').write_text(json.dumps({'budget_tokens':32,'shortlist':0}))
    cli(['select','--records',str(root/'train.jsonl'),'--fit',str(root/'fit'),'--config',str(root/'config.json'),'--out',str(root/'selected.json')])
    cli(['plan','--records',str(root/'train.jsonl'),'--selected',str(root/'selected.json'),'--budget-tokens','128','--out',str(root/'plan.json')])
    selection=json.loads((root/'selected.json').read_text());obj=json.loads((root/'plan.json').read_text())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    report=train_plan(model,rows,obj['plan'],optimizer,batch_size=8,microbatch_size=2)
    assert set(report['consumed_ids'])==set(selection['record_ids'])
    assert initial!=trainable_hash(model)
    report.update(scope='random tiny autoregressive model; no recommendation metric',initial_state=initial,
                  final_state=trainable_hash(model),selection=selection['report'],plan=obj['report'])
    (root/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
