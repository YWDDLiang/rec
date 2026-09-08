"""Causal-LM label masking utilities, unit-tested without downloading weights."""
import torch
import torch.nn.functional as F

def supervised_tokens(prompt_ids,response_ids,pad_id=0):
    if len(prompt_ids)!=len(response_ids) or not prompt_ids: raise ValueError('batch mismatch')
    if any(not p or not r for p,r in zip(prompt_ids,response_ids)): raise ValueError('prompt and response must be nonempty')
    n=max(len(p)+len(r) for p,r in zip(prompt_ids,response_ids))
    ids=torch.full((len(prompt_ids),n),pad_id,dtype=torch.long);labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids)
    for i,(p,r) in enumerate(zip(prompt_ids,response_ids)):
        ids[i,:len(p)+len(r)]=torch.tensor(p+r);labels[i,len(p):len(p)+len(r)]=torch.tensor(r);mask[i,:len(p)+len(r)]=1
    return {'input_ids':ids,'labels':labels,'attention_mask':mask}

def shifted_sequence_nll(logits,labels):
    if logits.shape[:2]!=labels.shape: raise ValueError('shape mismatch')
    y=labels[:,1:];x=logits[:,:-1,:];valid=y!=-100
    if not valid.any(1).all(): raise ValueError('sequence has no supervised next token')
    raw=F.cross_entropy(x.reshape(-1,x.shape[-1]),y.reshape(-1),ignore_index=-100,reduction='none').reshape_as(y)
    return raw.sum(1)/valid.sum(1)
