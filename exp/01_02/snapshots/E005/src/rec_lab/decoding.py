"""IDEA-08. Exact item-to-token transport and one-step speculative identity.

Finite-catalog reference implementation, NOT a GPU latency optimization.
"""
from __future__ import annotations
import numpy as np
from .numerics import probability,simplex

class CatalogTrie:
    """Injective semantic IDs; terminal marker avoids prefix ambiguity."""
    def __init__(self,sequences,eos=-1):
        self.eos=eos
        self.sequences=[tuple(s) for s in sequences]
        if not self.sequences or any(not s or eos in s for s in self.sequences): raise ValueError("bad SID/EOS")
        if len(set(self.sequences))!=len(self.sequences): raise ValueError("SID collision: item mapping must be injective")
        self.paths=[s+(eos,) for s in self.sequences]
    def next_distribution(self,prefix,item_probabilities):
        q=simplex(item_probabilities)
        if len(q)!=len(self.paths): raise ValueError("catalog mismatch")
        prefix=tuple(prefix);mass={}
        for path,p in zip(self.paths,q):
            if path[:len(prefix)]==prefix and len(path)>len(prefix):
                tok=path[len(prefix)];mass[tok]=mass.get(tok,0.)+float(p)
        total=sum(mass.values())
        if total<=0: raise ValueError("unreachable or terminal prefix")
        return {k:v/total for k,v in mass.items() if v>0}
    def item_path_probability(self,index,q):
        q=simplex(q)
        if len(q)!=len(self.paths): raise ValueError("catalog mismatch")
        if not isinstance(index,(int,np.integer)) or index<0 or index>=len(self.paths):
            raise ValueError("invalid item index")
        path=self.paths[index];p=1.
        for k,tok in enumerate(path):
            try: p*=self.next_distribution(path[:k],q).get(tok,0.)
            except ValueError: return 0.
        return p
    def sample(self,q,rng):
        prefix=()
        while True:
            d=self.next_distribution(prefix,q);keys=list(d)
            token=keys[int(rng.choice(len(keys),p=list(d.values())))]
            if token==self.eos: return self.sequences.index(prefix)
            prefix=prefix+(token,)

def exact_speculative_step(target,draft,rng):
    """One categorical proposal, target verification, residual correction.

    Exact for one-item sampling. Does NOT guarantee beam-search/top-K equality.
    """
    p=simplex(target);q=simplex(draft)
    if p.shape!=q.shape: raise ValueError("distribution mismatch")
    j=int(rng.choice(len(q),p=q))
    if rng.random()<min(1.,p[j]/q[j]): return j,True
    residual=np.maximum(p-q,0)
    if residual.sum()<=1e-14:  # rejection is unreachable when p=q in exact arithmetic
        return int(rng.choice(len(p),p=p)),False
    return int(rng.choice(len(p),p=probability(residual))),False

def speculative_identity(target,draft):
    """Analytic output distribution and expected acceptance, for exact tests."""
    p=simplex(target);q=simplex(draft)
    if p.shape!=q.shape: raise ValueError("distribution mismatch")
    accepted=np.minimum(p,q);res=np.maximum(p-q,0)
    mass=1-accepted.sum()
    output=accepted if mass<1e-14 else accepted+mass*res/res.sum()
    return output,float(accepted.sum())
