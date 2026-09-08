"""IDEA-04. Observable estimands, support checks, and doubly robust evaluation.

The functions do NOT infer unlogged propensities or fabricate missing feedback.
"""
from __future__ import annotations
import numpy as np
from .numerics import array

def _propensity(x,shape):
    p=array(x)
    if p.shape!=shape or np.any(p<=0) or np.any(p>1):
        raise ValueError("known propensity in (0,1] required; no silent clipping")
    return p

def missing_at_random_mean(y,observed,propensity,prediction):
    """DR mean E[m(x)+R/e(x)*(Y-m(x))] for one MAR observation process.

    Missing y may be NaN. Returns pseudo outcomes for estimation, not BCE labels.
    """
    y=np.asarray(y,dtype=float);r=np.asarray(observed,dtype=bool);mu=array(prediction)
    if y.shape!=r.shape or y.shape!=mu.shape: raise ValueError("shape mismatch")
    e=_propensity(propensity,y.shape)
    if not np.all(np.isfinite(y[r])): raise ValueError("observed labels must be finite")
    residual=np.zeros_like(mu);residual[r]=(y[r]-mu[r])/e[r]
    pseudo=mu+residual
    return {"estimate":float(pseudo.mean()),"pseudo_outcomes":pseudo}

def dr_policy_value(rewards,actions,behavior_prob,target_prob,predicted_rewards):
    """One-step contextual-bandit DR. No sequential-retention claim.

    behavior_prob contains the ACTUAL logged action propensity (not model score).
    target_prob and predicted_rewards are N x K, cross-fitted externally.
    """
    y=array(rewards,1);raw_a=array(actions,1)
    if not np.all(raw_a==np.floor(raw_a)): raise ValueError("actions must be integer indices")
    a=raw_a.astype(int);pi=array(target_prob,2);mu=array(predicted_rewards,2)
    if pi.shape!=mu.shape or len(pi)!=len(y) or a.shape!=y.shape: raise ValueError("shape mismatch")
    if np.any(a<0) or np.any(a>=pi.shape[1]): raise ValueError("invalid action")
    if np.any(pi<0) or not np.allclose(pi.sum(1),1): raise ValueError("target is not a policy")
    e=_propensity(behavior_prob,y.shape)
    ratio=pi[np.arange(len(y)),a]/e
    pseudo=(pi*mu).sum(1)+ratio*(y-mu[np.arange(len(y)),a])
    return {"estimate":float(pseudo.mean()),"pseudo_outcomes":pseudo,
            "effective_sample_size":float(ratio.sum()**2/(np.square(ratio).sum()+1e-30))}

def mature_mask(event_time,window,observed_until):
    t=array(event_time,1)
    if window<0: raise ValueError("window must be nonnegative")
    return t+window<=observed_until

def validate_feedback_labels(labels,mask):
    """Known values only; unobserved is not a negative label."""
    y=np.asarray(labels,dtype=float);m=np.asarray(mask,dtype=bool)
    if y.shape!=m.shape or not np.all(np.isfinite(y[m])): raise ValueError("invalid feedback")
    return np.where(m,y,0.),m
