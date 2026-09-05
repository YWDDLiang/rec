"""IDEA-07. Personalized graph proposals and a separate stopping process."""
from __future__ import annotations
import numpy as np
from .numerics import array,probability,simplex

def pagerank(adjacency,restart,alpha=.85,tol=1e-12,max_iter=10000):
    W=array(adjacency,2);v=simplex(restart)
    if W.shape!=(len(v),len(v)) or np.any(W<0) or not 0<=alpha<1: raise ValueError("invalid graph")
    rows=W.sum(1);P=np.zeros_like(W)
    np.divide(W,rows[:,None],out=P,where=rows[:,None]>0)
    P[rows==0]=v
    r=v.copy()
    for _ in range(max_iter):
        nr=alpha*P.T@r+(1-alpha)*v
        if np.linalg.norm(nr-r,1)<tol: return probability(nr)
        r=nr
    raise RuntimeError("PageRank did not converge")

def absorbing_value(transitions,rewards,discount=1.):
    """Q is substochastic on NONTERMINAL states; lost mass is stopping.

    gamma=1 requires spectral radius<1. An unrelated teleport probability is
    not interpreted as stopping or fatigue.
    """
    Q=array(transitions,2);r=array(rewards,1)
    if Q.shape!=(len(r),len(r)) or np.any(Q<0) or np.any(Q.sum(1)>1+1e-10) or not 0<=discount<=1:
        raise ValueError("invalid substochastic system")
    radius=float(np.max(np.abs(np.linalg.eigvals(discount*Q))))
    if radius>=1-1e-12: raise ValueError("nonabsorbing undiscounted dynamics: inverse not justified")
    return np.linalg.solve(np.eye(len(r))-discount*Q,r)

def safe_mixture(base,explore,lower_rewards,minimum_reward,max_exploration=1.):
    """Largest epsilon satisfying a ONE-STEP LCB expectation constraint.

    Conditional safety is only as valid as externally calibrated lower_rewards.
    No long-horizon retention guarantee is implied.
    """
    p=simplex(base);r=simplex(explore);l=array(lower_rewards,1)
    if p.shape!=r.shape or p.shape!=l.shape or not 0<=max_exploration<=1: raise ValueError("shape/range")
    bp=float(p@l);ep=float(r@l)
    if bp<minimum_reward-1e-12: raise ValueError("baseline itself violates one-step floor")
    eps=max_exploration
    if ep<bp: eps=min(eps,max(0.,(bp-minimum_reward)/(bp-ep)))
    return (1-eps)*p+eps*r,float(eps)

def kl_tilt(base,utility,temperature=1.):
    """Solution of max_q E_q[u]-T KL(q||p) on the base support."""
    p=simplex(base);u=array(utility,1)
    if p.shape!=u.shape or temperature<=0: raise ValueError("invalid tilt")
    valid=p>0;z=np.full(len(p),-np.inf);z[valid]=np.log(p[valid])+u[valid]/temperature
    z-=z[valid].max();out=np.exp(z)
    return out/out.sum()

def coverage_new_probability(interest_probabilities,already_seen):
    p=simplex(interest_probabilities);mask=np.asarray(already_seen,dtype=bool)
    if mask.shape!=p.shape: raise ValueError("coverage shape")
    return float(p[~mask].sum())
