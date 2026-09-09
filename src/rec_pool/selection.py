"""Budgeted selection; logdet is a STANDARD design proxy, not a novel guarantee.

Residual-quota-feasible cost-aware greedy is a heuristic under combined constraints.
No 1-1/e approximation guarantee is claimed for this constrained implementation.
"""
from __future__ import annotations
from dataclasses import dataclass,field
from collections import Counter
import numpy as np
from .geometry import matrix


@dataclass(frozen=True)
class Config:
    budget_tokens: int
    ridge: float=1.
    coverage_weight: float=1.
    utility_weight: float=0.
    seed: int=42
    min_rows: dict[str,int]=field(default_factory=dict)
    max_rows: dict[str,int]=field(default_factory=dict)
    protected: tuple[str,...]=()
    provenance_cap: int | None=None
    shortlist: int=256
    keep_all_when_fits: bool=True
    stop_nonpositive: bool=True

    def __post_init__(self):
        if type(self.budget_tokens) is not int or self.budget_tokens<1:raise ValueError('invalid token budget')
        if not np.isfinite([self.ridge,self.coverage_weight,self.utility_weight]).all():raise ValueError('nonfinite weights')
        if self.ridge<=0 or self.coverage_weight<0 or self.utility_weight<0:raise ValueError('invalid objective weights')
        if type(self.shortlist) is not int or self.shortlist<0:raise ValueError('invalid shortlist')
        if self.provenance_cap is not None and (type(self.provenance_cap) is not int or self.provenance_cap<1):raise ValueError('invalid provenance cap')
        for d in [self.min_rows,self.max_rows]:
            if any(not isinstance(k,str) or type(v) is not int or v<0 for k,v in d.items()):raise ValueError('invalid quotas')


def validate(z,costs,cells,groups,utility):
    z=matrix(z);n=len(z);costs=np.asarray(costs)
    if costs.shape!=(n,) or not np.issubdtype(costs.dtype,np.integer) or np.any(costs<1):raise ValueError('invalid integer costs')
    cells=np.asarray(cells,dtype=str);groups=np.asarray(groups,dtype=str)
    if cells.shape!=(n,) or groups.shape!=(n,):raise ValueError('metadata shape mismatch')
    u=np.zeros(n) if utility is None else np.asarray(utility,dtype=float)
    if u.shape!=(n,) or not np.isfinite(u).all():raise ValueError('invalid utility')
    return z,costs,cells,groups,u


def objective(z,ids,ridge=1.,utility=None,coverage_weight=1.,utility_weight=0.):
    z=matrix(z);ids=np.asarray(ids,dtype=int)
    m=ridge*np.eye(z.shape[1])+z[ids].T@z[ids]
    sign,ld=np.linalg.slogdet(m)
    if sign<=0:raise ArithmeticError('nonpositive information matrix')
    value=coverage_weight*(ld-z.shape[1]*np.log(ridge))
    if utility is not None:value+=utility_weight*float(np.asarray(utility)[ids].sum())
    return float(value)


def select(z,costs,cells,groups,config:Config,utility=None,mode='full_code'):
    z,costs,cells,groups,u=validate(z,costs,cells,groups,utility);n,k=z.shape
    if mode not in {'full_code','random','utility'}:raise ValueError('unknown mode')
    names=set(cells)
    if (set(config.min_rows)|set(config.max_rows)|set(config.protected))-names:raise ValueError('unknown quota cell; no silent task loss')
    counts=Counter(cells);lower={s:config.min_rows.get(s,0) for s in names}
    upper={s:config.max_rows.get(s,counts[s]) for s in names}
    for s in config.protected:lower[s]=counts[s]
    for s in names:
        if not 0<=lower[s]<=upper[s]<=counts[s]:raise ValueError('inconsistent or infeasible cell quota')
    if config.provenance_cap:
        # Provenance groups may not straddle cells; root x task x domain is the intended key.
        for g in set(groups):
            if len(set(cells[groups==g]))!=1:raise ValueError('provenance group crosses cells')
    cap=config.provenance_cap or n
    can_all=(costs.sum()<=config.budget_tokens and all(upper[s]==counts[s] for s in names)
             and max(Counter(groups).values())<=cap)
    if config.keep_all_when_fits and can_all:
        ids=np.arange(n,dtype=int)
        return ids,{'status':'keep_all','selected_rows':n,'tokens':int(costs.sum()),'unused_tokens':int(config.budget_tokens-costs.sum()),
                    'cell_counts':dict(counts),'scope':'no forced pruning; not a theorem that all-data is optimal'}
    rng=np.random.default_rng(config.seed);chosen=[];active=np.ones(n,bool)
    use=Counter();guse=Counter();spent=0;inverse=np.eye(k)/config.ridge;trace=[]
    # Reserve FEASIBILITY, not the actual minimum-cost rows. Preselecting all
    # lower-quota anchors makes min=max ablations degenerate to cheapest rows.
    order_by_cell={}
    for name in sorted(names):
        ix=np.flatnonzero(cells==name)
        order_by_cell[name]=ix[np.lexsort((rng.random(len(ix)),costs[ix]))]

    def completion(extra=None):
        tu=use.copy();tg=guse.copy()
        if extra is not None:tu[cells[extra]]+=1;tg[groups[extra]]+=1
        reserve=[]
        for name in sorted(names):
            need=max(0,lower[name]-tu[name])
            if not need:continue
            for j in order_by_cell[name]:
                if not active[j] or j==extra or tg[groups[j]]>=cap:continue
                reserve.append(int(j));tg[groups[j]]+=1;need-=1
                if not need:break
            if need:return None,None
        return reserve,int(costs[reserve].sum())

    reserve,reserved_cost=completion()
    if reserve is None:raise ValueError('provenance cap makes minimum quota impossible')
    if reserved_cost>config.budget_tokens:raise ValueError('minimum quota cost exceeds budget')

    def add(i,reason):
        nonlocal spent,inverse
        a=inverse@z[i];denom=1.+float(z[i]@a)
        if denom<=0:raise ArithmeticError('invalid Sherman-Morrison denominator')
        inverse-=np.outer(a,a)/denom
        inverse=(inverse+inverse.T)/2
        chosen.append(int(i));active[i]=False;use[cells[i]]+=1;guse[groups[i]]+=1;spent+=int(costs[i])
        if len(trace)<100:trace.append({'index':int(i),'reason':reason,'spent_tokens':spent})
    while True:
        feasible=active & (costs<=config.budget_tokens-spent)
        feasible &= np.array([use[s]<upper[s] for s in cells])
        feasible &= np.array([guse[g]<cap for g in groups])
        candidates=np.flatnonzero(feasible)
        if len(candidates)==0:break
        reserve,reserved_cost=completion()
        if reserve is None:raise ArithmeticError('lost quota feasibility')
        if config.shortlist and len(candidates)>config.shortlist:
            candidates=np.unique(np.concatenate([rng.choice(candidates,config.shortlist,replace=False),reserve])).astype(int)
        if mode=='random':
            ranked=rng.permutation(candidates);density=None
        else:
            gain=np.maximum(np.einsum('ij,jk,ik->i',z[candidates],inverse,z[candidates]),0.)
            score=config.utility_weight*u[candidates]
            if mode=='full_code':score=score+config.coverage_weight*np.log1p(gain)
            if mode=='utility':score=u[candidates]
            density=score/costs[candidates]
            if config.stop_nonpositive and not reserve and float(density.max())<=0:break
            ranked=candidates[np.argsort(-density,kind='stable')]
        picked=None
        for i in ranked:
            _,remaining=completion(int(i))
            if remaining is not None and spent+int(costs[i])+remaining<=config.budget_tokens:
                picked=int(i);break
        if picked is None:break
        add(picked,mode)
    ids=np.asarray(chosen,dtype=int)
    if any(use[s]<lower[s] or use[s]>upper[s] for s in names):raise ArithmeticError('quota violation')
    return ids,{'status':'selected','selected_rows':len(ids),'tokens':spent,'unused_tokens':config.budget_tokens-spent,
        'pool_rows':n,'pool_tokens':int(costs.sum()),'cell_counts':{s:use[s] for s in sorted(names)},
        'objective':objective(z,ids,config.ridge,u,config.coverage_weight,config.utility_weight),
        'trace_first100':trace,'scope':'one-pass admission budget; training occurrences accounted separately',
        'algorithm':'residual-quota-feasible (possibly stochastic) cost-aware greedy; no constrained approximation certificate'}


def concentration(counts):
    counts=np.asarray(counts,float)
    if counts.ndim!=1 or np.any(counts<0) or not np.isfinite(counts).all() or counts.sum()==0:raise ValueError('invalid exposure counts')
    p=counts/counts.sum();n=len(p)
    return {'ess':float(1/(p@p)),'tv_from_uniform':float(np.abs(p-1/n).sum()/2),
            'max_probability':float(p.max()),'unique_exposed':int(np.count_nonzero(p))}
