"""Full-group RL utilities. No within-group pruning; prompt correction explicit."""
from __future__ import annotations
import torch

def group_advantages(rewards:torch.Tensor,mode="loo"):
    """rewards [prompts, group, objectives].

    'loo' gives a leave-one-out baseline without reward std normalization.
    'grpo' gives standard mean/std-relative surrogates, NOT an unbiased
    estimator of the raw reward gradient. Zero-variance groups stay finite.
    """
    if rewards.ndim!=3 or rewards.shape[1]<2: raise ValueError("need [B,G>=2,M]")
    if not torch.isfinite(rewards).all(): raise ValueError("non-finite rewards")
    r=rewards.detach(); centered=r-r.mean(1,keepdim=True)
    if mode=="loo": return centered*r.shape[1]/(r.shape[1]-1)
    if mode=="grpo": return centered/r.std(1,unbiased=False,keepdim=True).clamp_min(1e-8)
    raise ValueError("mode must be loo or grpo")

def policy_surrogate(logp,old_logp,advantages,objective_weights,prompt_ratio=None,clip=None):
    if logp.ndim!=2 or old_logp.shape!=logp.shape or advantages.shape[:2]!=logp.shape:
        raise ValueError("group shape mismatch")
    if advantages.ndim!=3: raise ValueError("advantages need objective dimension")
    lam=torch.as_tensor(objective_weights,dtype=logp.dtype,device=logp.device)
    if lam.shape!=(advantages.shape[-1],) or (lam<0).any() or not torch.isclose(lam.sum(),lam.new_tensor(1.)):
        raise ValueError("objective weights must be simplex")
    a=(advantages.detach()*lam).sum(-1)
    ratio=torch.exp(logp-old_logp.detach())
    terms=ratio*a
    if clip is not None:
        if not 0<clip<1: raise ValueError("invalid clip")
        terms=torch.minimum(terms,ratio.clamp(1-clip,1+clip)*a)
    per_prompt=-terms.mean(1)
    if prompt_ratio is not None:
        w=torch.as_tensor(prompt_ratio,dtype=logp.dtype,device=logp.device).detach()
        if w.shape!=per_prompt.shape or (w<0).any() or not torch.isfinite(w).all(): raise ValueError("invalid prompt ratio")
        per_prompt=per_prompt*w  # NO self-normalization: preserves fixed-estimand expectation
    return per_prompt.mean()

def categorical_kl(log_probs,reference_log_probs):
    if log_probs.shape!=reference_log_probs.shape: raise ValueError("KL shape mismatch")
    if not torch.isfinite(log_probs).all() or not torch.isfinite(reference_log_probs).all():
        raise ValueError("finite common-support log probabilities required")
    return (log_probs.exp()*(log_probs-reference_log_probs.detach())).sum(-1)
