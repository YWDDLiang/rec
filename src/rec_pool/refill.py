"""Quota-preserving targeted refill from the REMOVED TRAINING pool, PDF §7.7."""
from __future__ import annotations
from collections import Counter
import numpy as np


def refill(selected,scores,costs,cells,*,max_swaps=500,seed=42,random_control=False,
           exact_token_match=True,protected_cells=(),groups=None,provenance_cap=None):
    raw=np.asarray(selected)
    if raw.ndim!=1 or not np.issubdtype(raw.dtype,np.integer) or len(set(raw.tolist()))!=len(raw):raise ValueError('invalid selected indices')
    scores=np.asarray(scores,float);costs=np.asarray(costs);cells=np.asarray(cells,str)
    n=len(scores)
    if scores.ndim!=1 or not np.isfinite(scores).all() or costs.shape!=(n,) or cells.shape!=(n,):raise ValueError('score/cost/cell mismatch')
    if not np.issubdtype(costs.dtype,np.integer) or np.any(costs<1) or np.any(raw<0) or np.any(raw>=n):raise ValueError('invalid costs or index')
    if type(max_swaps) is not int or max_swaps<0:raise ValueError('invalid swap budget')
    groups=np.arange(n).astype(str) if groups is None else np.asarray(groups,str)
    if groups.shape!=(n,):raise ValueError('group mismatch')
    if provenance_cap is not None and (type(provenance_cap) is not int or provenance_cap<1):raise ValueError('invalid provenance cap')
    rng=np.random.default_rng(seed);chosen=set(map(int,raw));counts=Counter(groups[raw]);swaps=[]
    before_cells=Counter(cells[raw]);before_tokens=int(costs[raw].sum())
    used_in=set();used_out=set()
    incoming=[int(i) for i in range(n) if i not in chosen and cells[i] not in protected_cells]
    if random_control:rng.shuffle(incoming)
    else:incoming.sort(key=lambda i:(-scores[i],i))
    for i in incoming:
        if len(swaps)>=max_swaps:break
        outgoing=[j for j in chosen if cells[j]==cells[i] and j not in used_in
                  and (not exact_token_match or costs[j]==costs[i])]
        if provenance_cap:
            outgoing=[j for j in outgoing if counts[groups[i]]-(groups[j]==groups[i])<provenance_cap]
        if not outgoing:continue
        j=int(rng.choice(outgoing)) if random_control else min(outgoing,key=lambda j:(scores[j],j))
        if not random_control and scores[i]<=scores[j]:continue
        chosen.remove(j);chosen.add(i);used_in.add(i);used_out.add(j)
        counts[groups[j]]-=1;counts[groups[i]]+=1
        swaps.append({'out':j,'in':i,'score_gain':float(scores[i]-scores[j]),'token_delta':int(costs[i]-costs[j])})
    result=np.array(sorted(chosen),dtype=int)
    if Counter(cells[result])!=before_cells:raise ArithmeticError('refill changed task/domain counts')
    if exact_token_match and costs[result].sum()!=before_tokens:raise ArithmeticError('refill changed token budget')
    return result,{'swaps':swaps,'requested_swaps':max_swaps,'actual_swaps':len(swaps),
                   'before_tokens':before_tokens,'after_tokens':int(costs[result].sum()),
                   'random_control':random_control,'scope':'training-pool exchange, not label synthesis or test-set retrieval'}
