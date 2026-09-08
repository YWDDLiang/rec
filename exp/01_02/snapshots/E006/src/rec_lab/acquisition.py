"""IDEA-06. Dual-guided audited information acquisition, not fake user labels."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .numerics import array,simplex

@dataclass(frozen=True)
class PreferencePair:
    prompt_id: str
    chosen: str
    rejected: str
    source: str
    confidence: float
    evidence_id: str
    split: str="train"
    def validate(self):
        if self.chosen==self.rejected or not self.prompt_id or not self.evidence_id:
            raise ValueError("nontrivial pair and traceable evidence required")
        if self.source not in {"observed","expert","simulator","synthetic"}: raise ValueError("unknown provenance")
        if not 0<=self.confidence<=1 or self.split not in {"train","validation","test"}: raise ValueError("bad fields")
        return self

def rank_acquisition(witness,predicted_responses,response_uncertainty,cost,current_margin=0.):
    """LCB of dual witness violation per cost; a heuristic acquisition rule.

    A positive violation is necessary, NOT sufficient, to improve pool max-min
    margin. Predictions do not create new verified training data.
    """
    lam=simplex(witness);A=array(predicted_responses,2);U=array(response_uncertainty,2);c=array(cost,1)
    if A.shape!=U.shape or A.shape[0]!=len(lam) or A.shape[1]!=len(c) or np.any(U<0) or np.any(c<=0):
        raise ValueError("invalid acquisition data")
    if not np.isfinite(current_margin): raise ValueError("non-finite margin")
    score=(lam@(A-U)-current_margin)/c
    return np.argsort(-score),score

def accept_pairs(pairs,min_confidence=0.8,allow_synthetic=False):
    if not 0<=min_confidence<=1: raise ValueError("confidence range")
    out=[]
    for p in pairs:
        p.validate()
        if p.split!="train": raise ValueError("selection cannot consume held-out labels")
        if p.confidence<min_confidence: continue
        if p.source in {"synthetic","simulator"} and not allow_synthetic: continue
        out.append(p)
    return out

def noise_corrected_logistic_loss(logits,labels,error_rate):
    """Known symmetric independent label noise epsilon<.5; unbiased risk.

    Variance diverges as epsilon approaches .5. Never estimate epsilon from
    the same self-generated agreement used to declare a pair trustworthy.
    """
    z=array(logits);y=array(labels)
    if z.shape!=y.shape or not np.all(np.isin(y,[0,1])) or not 0<=error_rate<.5:
        raise ValueError("binary labels and noise < .5 required")
    true=np.logaddexp(0,z)-y*z
    flipped=np.logaddexp(0,z)-(1-y)*z
    return ((1-error_rate)*true-error_rate*flipped)/(1-2*error_rate)

def best_acquisition_batch(existing_responses,new_responses,cost,budget,max_combinations=20000):
    """Exact small-pool budgeted complementary acquisition reference.

    Enumerates feasible subsets; exponential, guarded, NOT a scalable claim.
    New response columns are hypothetical until verified outcomes are acquired.
    The set gain is NOT generally submodular, so no greedy 1-1/e is claimed.
    """
    from itertools import combinations
    from .frontier import solve_frontier
    A=array(existing_responses,2);B=array(new_responses,2);c=array(cost,1)
    if A.shape[0]!=B.shape[0] or len(c)!=B.shape[1] or np.any(c<=0) or budget<0:raise ValueError('invalid acquisition budget')
    # Reject before exponential work begins; no silently partial "optimal" result.
    if 2**len(c)>max_combinations:raise ValueError('exact enumeration cap exceeded; use an audited scalable optimizer')
    base=solve_frontier(A);best=base.margin;chosen=[];count=0
    for k in range(1,len(c)+1):
        for S in combinations(range(len(c)),k):
            if c[list(S)].sum()>budget:continue
            count+=1;r=solve_frontier(np.c_[A,B[:,S]])
            if r.margin>best+1e-9:best=r.margin;chosen=list(S)
    return {'selected':chosen,'new_margin':best,'gain':best-base.margin,'subsets_evaluated':count}
