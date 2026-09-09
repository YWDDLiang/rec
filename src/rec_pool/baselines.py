"""Transparent internal controls. Named external papers are NOT reimplemented here."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from .geometry import matrix


def pdf_rule_reference(codes,losses,strata,*,user_mask=None,quality_mask=None,
                       cap=800,min_after_truncation=50,seed=42,middle_quantiles=(.2,.8)):
    """Executable reference for PDF §7.3, NOT the unavailable original script.

    Clarified choices: mid-loss interval uses user-configurable quantiles, and
    round-robin stratification covers task/source/domain/cluster tuple strings.
    Character-length quality filtering must be supplied from RAW text; never
    replace the PDF's character rules with token counts without disclosure.
    """
    c=matrix(codes);n=len(c);losses=np.asarray(losses,float);strata=np.asarray(strata,str)
    user=np.zeros(n,bool) if user_mask is None else np.asarray(user_mask,bool)
    quality=np.ones(n,bool) if quality_mask is None else np.asarray(quality_mask,bool)
    if any(v.shape!=(n,) for v in [losses,strata,user,quality]) or not np.isfinite(losses).all():raise ValueError('invalid rule inputs')
    if type(cap) is not int or cap<1 or not 0<=min_after_truncation<=cap:raise ValueError('invalid atom count bounds')
    qlo,qhi=middle_quantiles
    if not 0<=qlo<qhi<=1:raise ValueError('invalid loss interval')
    rng=np.random.default_rng(seed);dominant=np.abs(c).argmax(1);weight=np.abs(c).max(1);keep=set()
    for atom in range(c.shape[1]):
        ids=np.flatnonzero((dominant==atom)&quality)
        if len(ids)<=cap:keep.update(map(int,ids));continue
        forced=list(map(int,ids[user[ids]]));chosen=set(forced)
        rest=[int(i) for i in ids if int(i) not in chosen];remaining=max(0,cap-len(chosen))
        top_count=int(round(.2*remaining))
        top=sorted(rest,key=lambda i:(-weight[i],i))[:max(top_count*2,top_count)]
        if top_count and top:chosen.update(map(int,rng.choice(top,min(top_count,len(top)),replace=False)))
        lo,hi=np.quantile(losses[ids],[qlo,qhi]);groups=defaultdict(list)
        for i in rest:
            if i not in chosen and lo<=losses[i]<=hi:groups[strata[i]].append(i)
        for g in groups:rng.shuffle(groups[g])
        keys=sorted(groups);rng.shuffle(keys)
        while len(chosen)<max(cap,len(forced)) and any(groups.values()):
            for key in keys:
                if groups[key] and len(chosen)<max(cap,len(forced)):chosen.add(groups[key].pop())
        if len(chosen)<min_after_truncation:
            extra=[int(i) for i in ids if i not in chosen];rng.shuffle(extra)
            chosen.update(extra[:min_after_truncation-len(chosen)])
        keep.update(chosen)
    return np.asarray(sorted(keep),dtype=int)


def full_code_cluster(features,count=64,seed=42):
    from sklearn.cluster import MiniBatchKMeans
    x=matrix(features)
    if not 1<=count<=len(x):raise ValueError('invalid cluster count')
    return MiniBatchKMeans(n_clusters=count,random_state=seed,n_init=3,batch_size=min(256,len(x))).fit_predict(x)


def nested_root_reservoir(root_ids,size,seed=42):
    """Order-invariant nested pool size sweep on ROOTS, not instruction variants.

    This bounds the pool before costly probing. Expanding its size is NOT free;
    every larger probe must be charged in total-cost comparisons.
    """
    import hashlib
    roots=list(map(str,root_ids));unique=sorted(set(roots))
    if type(size) is not int or size<1:raise ValueError('invalid reservoir size')
    rank=sorted(unique,key=lambda s:hashlib.sha256(f'{seed}|{s}'.encode()).digest())
    selected=set(rank[:size]);return np.array([i for i,r in enumerate(roots) if r in selected],dtype=int)
