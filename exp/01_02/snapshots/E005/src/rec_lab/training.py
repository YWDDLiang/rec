"""Shared supervised multi-feedback trainer: tiny model and optional HF ranker.

Exact gradients in an explicitly selected parameter subspace. The selector is a
local acquisition/mixture controller, not an approximation of the original loss.
Model testing is held out until training ends. Single-process reference code.
"""
from __future__ import annotations
from dataclasses import dataclass
import time, json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import validate_splits
from .frontier import response_matrix,solve_frontier,column_generation,response_with_floor
from .atoms import nuisance_residual,learn_dictionary,nominate_columns
from .numerics import softmax

class TinyFeedbackModel(nn.Module):
    """Randomly initialized sequential neural recommender, NOT a pretrained LLM."""
    def __init__(self,items,scenes,dim=8,objectives=2):
        super().__init__();self.item_map={x:i+1 for i,x in enumerate(items)}
        self.scene_map={x:i for i,x in enumerate(scenes)}
        self.item_embedding=nn.Embedding(len(items)+1,dim,padding_idx=0)
        self.scene_embedding=nn.Embedding(len(scenes),dim)
        self.feedback_head=nn.Sequential(nn.Linear(dim*4,dim),nn.Tanh(),nn.Linear(dim,objectives))
    def forward(self,records):
        dev=next(self.parameters()).device
        maxlen=max(1,max(len(r.history) for r in records))
        hist=torch.zeros(len(records),maxlen,dtype=torch.long,device=dev)
        for k,r in enumerate(records):
            vals=[self.item_map.get(x,0) for x in r.history]
            if vals: hist[k,:len(vals)]=torch.tensor(vals,device=dev)
        mask=(hist!=0).unsqueeze(-1);h=(self.item_embedding(hist)*mask).sum(1)/mask.sum(1).clamp_min(1)
        it=self.item_embedding(torch.tensor([self.item_map.get(r.item_id,0) for r in records],device=dev))
        sc=self.scene_embedding(torch.tensor([self.scene_map[r.scene] for r in records],device=dev))
        return self.feedback_head(torch.cat([h,it,h*it,sc],-1))

def feedback_losses(model,records):
    logits=model(records)
    y=torch.tensor([[0. if v is None else v for v in r.labels] for r in records],dtype=logits.dtype,device=logits.device)
    mask=torch.tensor([[v is not None for v in r.labels] for r in records],dtype=logits.dtype,device=logits.device)
    if logits.shape!=y.shape or not torch.isfinite(logits).all(): raise ValueError('bad model output')
    raw=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
    return raw*mask,mask

def flatten_gradient(loss,parameters,retain_graph=False):
    grad=torch.autograd.grad(loss,parameters,allow_unused=True,retain_graph=retain_graph)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(parameters,grad)]).detach().double().cpu().numpy()

def select_parameters(model,name_filter=None,max_elements=2000000):
    selected=[p for name,p in model.named_parameters() if p.requires_grad and (name_filter is None or name_filter in name)]
    n=sum(p.numel() for p in selected)
    if not selected: raise ValueError('no selected trainable parameters')
    if n>max_elements: raise ValueError(f'gradient subspace has {n} elements; limit {max_elements}. Use a smaller probe or explicitly increase the cap.')
    return selected,n

def gradient_snapshot(model,pool,reference,parameters):
    """Per-example masked mean loss update and per-objective validation gradients."""
    model.eval() # dropouts would otherwise invalidate consistent same-state response
    losses,mask=feedback_losses(model,pool)
    per=losses.sum(1)/mask.sum(1).clamp_min(1)
    G=np.stack([flatten_gradient(per[i],parameters,retain_graph=True) for i in range(len(pool))])
    val,vm=feedback_losses(model,reference)
    if (vm.sum(0)==0).any(): raise ValueError('each target needs observed selection labels')
    means=val.sum(0)/vm.sum(0)
    V=np.stack([flatten_gradient(means[j],parameters,retain_graph=True) for j in range(means.numel())])
    return G,V

def choose_weights(G,V,pool,method,exploration_floor=.05,seed=0):
    A=response_matrix(V,G);n=len(G);p=np.ones(n)/n
    detail={'scope':'selected parameter subspace only','local_linear_margin':float((A@p).min())}
    if method=='uniform': return p,detail
    if method=='scalar': return softmax(A.mean(0)/(np.std(A.mean(0))+1e-12)),detail
    if method not in {'frontier','atoms_frontier'}: raise ValueError('unknown method')
    Af=response_with_floor(A,p,exploration_floor)
    if method=='frontier': res=solve_frontier(Af)
    else:
        scenes=sorted({r.scene for r in pool})
        Z=np.c_[np.ones(n),[len(r.history) for r in pool],[[float(r.scene==s) for s in scenes] for r in pool]]
        residual,_=nuisance_residual(G,Z)
        # A randomized low-dimensional direction sketch only nominates columns.
        # The LP response and full-pool pricing below use the UNSKETCHED gradients.
        dim=min(24,G.shape[1]);rng=np.random.default_rng(seed)
        sketch=residual@rng.normal(size=(G.shape[1],dim))/np.sqrt(dim)
        fit=learn_dictionary(sketch,n_atoms=min(6,n),sparsity=min(2,n),iterations=3,seed=seed)
        codes=fit["codes"]
        initial=nominate_columns(codes,Af,per_atom=1)
        res,cols=column_generation(Af,initial)
        detail.update(nominated_columns=len(initial),priced_columns=len(cols),atom_reconstruction_error=float(fit['mse']))
    q=exploration_floor*p+(1-exploration_floor)*res.weights
    detail.update(local_linear_margin=float((A@q).min()),dual_gap=float(res.gap),matrix_certificate=res.certified,
                  witness=res.witness.tolist(),active=int((q>exploration_floor/n+1e-10).sum()))
    return q,detail

@torch.no_grad()
def evaluate_feedback(model,records,batch_size=64):
    model.eval();all_loss=[];all_mask=[];all_prob=[];all_y=[]
    for a in range(0,len(records),batch_size):
        batch=records[a:a+batch_size];logits=model(batch)
        loss,mask=feedback_losses(model,batch)
        all_loss.append(loss.cpu());all_mask.append(mask.cpu());all_prob.append(logits.sigmoid().cpu())
        all_y.append(torch.tensor([[0 if v is None else v for v in r.labels] for r in batch]))
    l=torch.cat(all_loss);m=torch.cat(all_mask);p=torch.cat(all_prob);y=torch.cat(all_y)
    if (m.sum(0)==0).any(): raise ValueError('evaluation objective missing')
    return {'bce':(l.sum(0)/m.sum(0)).tolist(),'brier':(((p-y)**2*m).sum(0)/m.sum(0)).tolist(),
            'n':len(records),'observed_per_objective':m.sum(0).tolist()}

def train_feedback(model,records,method='frontier',steps=20,pool_size=16,reference_size=48,
                   lr=.1,seed=0,parameter_filter=None,max_gradient_elements=2000000,
                   exploration_floor=.05,optimizer_name='sgd'):
    by=validate_splits(records);rng=np.random.default_rng(seed)
    if min(steps,pool_size,reference_size)<1 or lr<=0: raise ValueError('invalid training config')
    parameters,n=select_parameters(model,parameter_filter,max_gradient_elements)
    if optimizer_name=='sgd': optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=lr)
    elif optimizer_name=='adamw': optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr)
    else: raise ValueError('unsupported optimizer')
    # Reference is from selection partition only. Test records never enter q.
    ref=[by['selection'][i] for i in rng.choice(len(by['selection']),min(reference_size,len(by['selection'])),replace=False)]
    log=[];started=time.perf_counter()
    before=evaluate_feedback(model,by['selection'])
    for step in range(steps):
        ids=rng.choice(len(by['train']),min(pool_size,len(by['train'])),replace=False)
        pool=[by['train'][i] for i in ids];tic=time.perf_counter()
        if method=='uniform':
            q=np.ones(len(pool))/len(pool);detail={'scope':'uniform baseline; no gradient valuation'}
        else:
            G,V=gradient_snapshot(model,pool,ref,parameters)
            q,detail=choose_weights(G,V,pool,method,exploration_floor,seed+step)
        valuation_time=time.perf_counter()-tic
        optimizer.zero_grad(set_to_none=True)
        # Keep dropout disabled for a consistent deterministic weighted update.
        model.eval();loss,mask=feedback_losses(model,pool)
        per=loss.sum(1)/mask.sum(1).clamp_min(1)
        objective=(per*torch.as_tensor(q,dtype=per.dtype,device=per.device)).sum()
        objective.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()): raise FloatingPointError('non-finite gradients')
        # Only abstain for a full-gradient, plain-SGD nonpositive local response.
        # Even positive response is NOT a finite-step smoothness certificate here.
        abstain=bool(parameter_filter is None and optimizer_name=='sgd' and method in {'frontier','atoms_frontier'} and detail['local_linear_margin']<=0)
        if not abstain: optimizer.step()
        log.append({'step':step,'weighted_training_loss':float(objective.detach()),'valuation_seconds':valuation_time,
                    'abstained':abstain,**detail})
    elapsed=time.perf_counter()-started
    return {'method':method,'seed':seed,'steps':steps,'gradient_elements':n,'parameter_filter':parameter_filter,
        'optimizer':optimizer_name,'selection_before':before,'selection_after':evaluate_feedback(model,by['selection']),
        'test':evaluate_feedback(model,by['test']),'elapsed_seconds':elapsed,'trace':log,
        'finite_step_certified':False,'warning':'Matrix LP certificate is not a neural finite-step/test-performance guarantee.'}
