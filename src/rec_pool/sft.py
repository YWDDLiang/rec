"""Shared causal SFT loss for probe and training; accepts production-encoded rows.

No task prefix is added. One tokenizer/template recipe must be used on all paths.
This is not a reproduction of any upstream training/packing/RL implementation.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from collections import Counter
import hashlib
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from .contracts import validate_rows,rows_hash,digest


def collate(rows,pad_id=0,device='cpu'):
    length=max(len(r.input_ids) for r in rows)
    ids=torch.full((len(rows),length),pad_id,dtype=torch.long,device=device)
    labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids)
    for i,r in enumerate(rows):
        n=len(r.input_ids);ids[i,:n]=torch.tensor(r.input_ids,device=device)
        labels[i,:n]=torch.tensor(r.labels,device=device);mask[i,:n]=1
    return {'input_ids':ids,'attention_mask':mask},labels


def loss_sums(model,rows,pad_id=0):
    batch,labels=collate(rows,pad_id,next(model.parameters()).device)
    output=model(**batch);logits=output.logits if hasattr(output,'logits') else output
    if logits.ndim!=3 or logits.shape[:2]!=labels.shape:raise ValueError('not causal LM logits')
    target=labels[:,1:];pred=logits[:,:-1].float()
    losses=F.cross_entropy(pred.reshape(-1,pred.shape[-1]),target.reshape(-1),ignore_index=-100,reduction='none').reshape_as(target)
    return losses.sum(1),(target!=-100).sum(1)


def loss(model,rows,reduction='token_mean',pad_id=0):
    sums,counts=loss_sums(model,rows,pad_id)
    if torch.any(counts==0):raise ValueError('empty supervision')
    if reduction=='token_mean':return sums.sum()/counts.sum()
    if reduction=='example_mean':return (sums/counts).mean()
    raise ValueError('unknown reduction')


@contextmanager
def stable_probe(model):
    modes=[(m,m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.enable_grad():yield
    finally:
        for m,mode in modes:m.training=mode


def manifest(model):
    out=[];offset=0
    for name,p in model.named_parameters():
        if p.requires_grad:
            out.append({'name':name,'shape':list(p.shape),'start':offset,'end':offset+p.numel(),'dtype':str(p.dtype)})
            offset+=p.numel()
    if not out:raise ValueError('no trainable parameters')
    return out


def trainable_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode());h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def extract(model,rows,path,*,dimension=512,seed=42,base_revision,overwrite=False):
    """Stream full TRAINABLE gradients into signed-hash memmaps; no optimizer steps.

    Saves raw sketch, unit direction, full norm, losses, projection and manifest.
    Per-row gradient uses response-token mean. The extraction cost is fully paid
    before selection. Test/audit/tune rows are never accepted as probe input.
    """
    validate_rows(rows)
    if len({r.split for r in rows})!=1 or rows[0].split not in {'train','select'}:raise ValueError('probe accepts train OR select, never test/audit/tune')
    if type(dimension) is not int or dimension<1 or not isinstance(base_revision,str) or not base_revision:raise ValueError('invalid projection/base revision')
    path=Path(path)
    if path.exists() and any(path.iterdir()) and not overwrite:raise FileExistsError(path)
    path.mkdir(parents=True,exist_ok=True)
    m=manifest(model);pcount=m[-1]['end'];params=[p for p in model.parameters() if p.requires_grad]
    rng=np.random.default_rng(seed);bucket=rng.integers(0,dimension,pcount,dtype=np.int64)
    signs=rng.choice(np.array([-1,1],np.int8),pcount)
    raw=np.lib.format.open_memmap(path/'raw.npy',mode='w+',dtype='float32',shape=(len(rows),dimension))
    unit=np.lib.format.open_memmap(path/'unit.npy',mode='w+',dtype='float32',shape=(len(rows),dimension))
    norms=np.zeros(len(rows));losses=np.zeros(len(rows));before=trainable_hash(model)
    started=time.perf_counter()
    with stable_probe(model):
        for i,row in enumerate(rows):
            value=loss(model,[row]);grads=torch.autograd.grad(value,params,allow_unused=True)
            projected=np.zeros(dimension,np.float64);sq=0.
            for item,param,g in zip(m,params,grads):
                if g is None:continue
                a=g.detach().float().cpu().numpy().reshape(-1).astype(np.float64)
                if not np.isfinite(a).all():raise FloatingPointError('nonfinite gradient')
                lo,hi=item['start'],item['end']
                sq+=float(a@a);projected+=np.bincount(bucket[lo:hi],weights=a*signs[lo:hi],minlength=dimension)
            raw[i]=projected;unit[i]=projected/max(np.linalg.norm(projected),1e-30)
            norms[i]=np.sqrt(sq);losses[i]=float(value.detach())
    raw.flush();unit.flush()
    after=trainable_hash(model)
    if before!=after:raise RuntimeError('probe mutated parameters')
    np.save(path/'norms.npy',norms);np.save(path/'losses.npy',losses)
    np.save(path/'buckets.npy',bucket);np.save(path/'signs.npy',signs)
    projection_hash=hashlib.sha256(bucket.tobytes()+signs.tobytes()).hexdigest()
    meta={'record_ids':[r.record_id for r in rows],'rows_hash':rows_hash(rows),'recipe_id':rows[0].recipe_id,
          'base_revision':base_revision,'trainable_state_hash':before,'manifest':m,'projection_hash':projection_hash,
          'projection_dimension':dimension,'seed':seed,'split':rows[0].split,'seconds':time.perf_counter()-started,
          'probe_mode':'eval with autograd; fixed checkpoint; response token mean; all requires_grad parameters',
          'scope':'identity-gradient geometry, not AdamW update or ranking gain', 'numpy_version':np.__version__}
    meta['file_hashes']={name:file_sha(path/name) for name in ['raw.npy','unit.npy','norms.npy','losses.npy','buckets.npy','signs.npy']}
    (path/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return meta


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def validate_cache(rows,path):
    path=Path(path);meta=json.loads((path/'meta.json').read_text())
    if meta['rows_hash']!=rows_hash(rows):raise ValueError('cache record order/content/partition mismatch')
    for name,expected in meta.get('file_hashes',{}).items():
        if file_sha(path/name)!=expected:raise ValueError('gradient cache bytes changed')
    if not meta.get('file_hashes'):raise ValueError('unsigned cache: re-export with file hashes')
    raw=np.load(path/'raw.npy',mmap_mode='r');unit=np.load(path/'unit.npy',mmap_mode='r')
    norms=np.load(path/'norms.npy');losses=np.load(path/'losses.npy')
    if raw.shape!=unit.shape or raw.shape!=(len(rows),meta['projection_dimension']):raise ValueError('invalid cache shape')
    if norms.shape!=(len(rows),) or losses.shape!=(len(rows),):raise ValueError('invalid norm/loss cache')
    if not all(np.isfinite(a).all() for a in [raw,unit,norms,losses]):raise ValueError('nonfinite cache')
    return meta,raw,unit,norms,losses


def compatible(a,b):
    keys=['recipe_id','base_revision','trainable_state_hash','manifest','projection_hash']
    if any(a[k]!=b[k] for k in keys):raise ValueError('cannot compare different gradient coordinate systems')


def schedule(rows,selected,budget_tokens,seed=42,max_epochs=100):
    """Whole examples, shuffled without replacement each epoch; all repeats logged.

    Does not truncate a target to hit an exact token count. Reports unused budget.
    Admission token budget and optimizer exposure budget are separate controls.
    """
    ids=np.asarray(selected)
    if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or len(ids)==0 or len(set(ids.tolist()))!=len(ids):raise ValueError('invalid selected IDs')
    if np.any(ids<0) or np.any(ids>=len(rows)) or type(budget_tokens) is not int or budget_tokens<1:raise ValueError('invalid schedule bounds')
    if type(max_epochs) is not int or max_epochs<1:raise ValueError('invalid max_epochs')
    rng=np.random.default_rng(seed);plan=[];spent=0;counts=Counter()
    for epoch in range(max_epochs):
        made=False
        for i in rng.permutation(ids):
            r=rows[int(i)]
            if spent+r.cost>budget_tokens:continue
            counts[r.record_id]+=1
            plan.append({'record_index':int(i),'record_id':r.record_id,'content_id':r.content_id,
                         'occurrence_id':f'{r.record_id}#{counts[r.record_id]}','epoch':epoch,
                         'response_tokens':r.cost,'input_tokens':len(r.input_ids),'cell':r.cell})
            spent+=r.cost;made=True
        if not made or spent==budget_tokens:break
    if not plan:raise ValueError('budget smaller than any selected example')
    return plan,{'tokens':spent,'unused_tokens':budget_tokens-spent,'occurrences':len(plan),
                'unique_records':len(counts),'duplicate_occurrence_fraction':1-len(counts)/len(plan),
                'input_tokens':sum(p['input_tokens'] for p in plan),'max_epochs':max_epochs}


def train_plan(model,rows,plan,optimizer,*,batch_size=8,microbatch_size=1,reduction='token_mean',clip_norm=1.):
    if type(batch_size) is not int or type(microbatch_size) is not int or min(batch_size,microbatch_size)<1:raise ValueError('invalid batching')
    if not np.isfinite(clip_norm) or clip_norm<=0:raise ValueError('invalid clip norm')
    if not plan:raise ValueError('empty plan')
    validate_rows(rows)
    seen_occurrences=set()
    for ent in plan:
        i=ent.get('record_index')
        if type(i) is not int or not 0<=i<len(rows):raise ValueError('plan index out of range')
        r=rows[i]
        if (ent.get('response_tokens')!=r.cost or ent.get('input_tokens')!=len(r.input_ids)
            or ent.get('cell')!=r.cell):raise ValueError('forged plan cost/cell accounting')
        occ=ent.get('occurrence_id')
        if not isinstance(occ,str) or not occ or occ in seen_occurrences:raise ValueError('duplicate/invalid occurrence ID')
        seen_occurrences.add(occ)
    consumed=[];step_losses=[];started=time.perf_counter();model.train()
    for start in range(0,len(plan),batch_size):
        entries=plan[start:start+batch_size];batch=[]
        for ent in entries:
            i=ent['record_index']
            if type(i) is not int or not 0<=i<len(rows):raise ValueError('plan index out of range')
            r=rows[i]
            if r.split!='train' or ent['record_id']!=r.record_id or ent['content_id']!=r.content_id:raise ValueError('plan/training data mismatch')
            batch.append(r)
        optimizer.zero_grad(set_to_none=True);value=0.
        denominator=sum(r.cost for r in batch) if reduction=='token_mean' else len(batch)
        if reduction not in {'token_mean','example_mean'}:raise ValueError('unknown loss reduction')
        for lo in range(0,len(batch),microbatch_size):
            part=batch[lo:lo+microbatch_size];sums,counts=loss_sums(model,part)
            scaled=sums.sum()/denominator if reduction=='token_mean' else (sums/counts).sum()/denominator
            scaled.backward();value+=float(scaled.detach())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],clip_norm,error_if_nonfinite=True)
        optimizer.step();step_losses.append(value);consumed.extend(entries)
    return {'steps':len(step_losses),'occurrences':len(consumed),'tokens':sum(e['response_tokens'] for e in consumed),
            'input_tokens':sum(e['input_tokens'] for e in consumed),'seconds':time.perf_counter()-started,
            'losses':step_losses,'consumed_ids':[e['record_id'] for e in consumed]}


def load_hf(model_path,*,device='cpu',lora_rank=8,adapter_path=None,seed=42):
    """Local-only model loading; no silent download, no OneReason-specific tokens."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig,get_peft_model,PeftModel
    torch.manual_seed(seed)
    dtype=torch.float32 if device=='cpu' else torch.bfloat16
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=dtype,trust_remote_code=False)
    model.config.use_cache=False
    if adapter_path:
        model=PeftModel.from_pretrained(model,adapter_path,is_trainable=True,local_files_only=True)
    else:
        model=get_peft_model(model,LoraConfig(r=lora_rank,lora_alpha=2*lora_rank,lora_dropout=0.,
                            target_modules='all-linear',task_type='CAUSAL_LM'))
    return model.to(device)
