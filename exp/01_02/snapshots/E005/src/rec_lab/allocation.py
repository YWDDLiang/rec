"""IDEA-05. Vector-objective, fixed-estimand prompt sampling allocation."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .numerics import array,simplex

def scalar_variance_optimal(base,second_moment):
    p=simplex(base);a=array(second_moment,1)
    if a.shape!=p.shape or np.any(a<0): raise ValueError("invalid second moments")
    z=p*np.sqrt(a)
    return p.copy() if z.sum()==0 else z/z.sum()

def variance_objectives(base,q,moments):
    p=simplex(base);q=simplex(q);A=array(moments,2)
    if len(A)!=len(p) or len(q)!=len(p) or np.any(A<0): raise ValueError("invalid shapes")
    positive=(p>0)&(np.any(A>0,axis=1))
    if np.any(q[positive]<=0): return np.full(A.shape[1],np.inf)
    terms=np.zeros_like(A);active=q>0
    terms[active]=p[active,None]**2*A[active]/q[active,None]
    return terms.sum(0)

def minimax_allocation(base,moments,floor=0.05):
    """min_q max_m sum_i p_i² E[X_im²]/q_i, with q_i>=floor*p_i.

    Means are fixed, so minimizing second moments bounds variance. Per-objective
    scales must be fixed before calling. The optimizer's success is checked.
    """
    p=simplex(base);A=array(moments,2)
    if len(A)!=len(p) or np.any(A<0) or np.any(p<=0) or not 0<floor<=1:
        raise ValueError("strictly positive base and floor are required")
    n=len(p);scaled=(p[:,None]**2*A)
    scale=max(float((scaled/p[:,None]).sum(0).max()),1e-12)
    C=scaled/scale
    def cons(x): return x[-1]-(C/x[:-1,None]).sum(0)
    def jac_cons(x): return np.c_[(C/x[:-1,None]**2).T,np.ones(A.shape[1])]
    x0=np.r_[p,(C/p[:,None]).sum(0).max()]
    opt=minimize(lambda x:x[-1],x0,jac=lambda x:np.r_[np.zeros(n),1.],
        bounds=[(floor*v,1.) for v in p]+[(0,None)],
        constraints=[{"type":"eq","fun":lambda x:x[:-1].sum()-1,
                      "jac":lambda x:np.r_[np.ones(n),0.]},
                     {"type":"ineq","fun":cons,"jac":jac_cons}],
        method="SLSQP",options={"ftol":1e-11,"maxiter":1000})
    if not opt.success or np.min(cons(opt.x)) < -1e-6: raise RuntimeError(opt.message)
    q=opt.x[:-1];q/=q.sum()
    return {"q":q,"objective_second_moments":variance_objectives(p,q,A),
            "success":True,"floor":floor}

def corrected_estimate(values,sampled_indices,base,q):
    X=array(values);p=simplex(base);q=simplex(q);raw=np.asarray(sampled_indices)
    if raw.ndim!=1 or not np.issubdtype(raw.dtype,np.number) or not np.all(np.isfinite(raw)) or np.any(raw!=np.floor(raw)):
        raise ValueError("indices must be finite integers")
    idx=raw.astype(int)
    if len(X)!=len(p) or len(q)!=len(p) or idx.ndim!=1 or not len(idx): raise ValueError("shape/empty sample")
    if np.any(idx<0) or np.any(idx>=len(p)) or np.any(q[p>0]<=0): raise ValueError("support violation")
    shape=(-1,)+(1,)*(X.ndim-1)
    return (X[idx]*(p[idx]/q[idx]).reshape(shape)).mean(0)

def continuous_cost_allocation(base,variance,cost,total_budget):
    """Independent-stratum continuous n_i solution. No integer or GRPO claim."""
    p=simplex(base);a=array(variance,1);c=array(cost,1)
    if a.shape!=p.shape or c.shape!=p.shape or np.any(a<0) or np.any(c<=0) or total_budget<=0:
        raise ValueError("invalid allocation")
    z=p*np.sqrt(a/c)
    return np.zeros_like(p) if np.dot(c,z)==0 else total_budget*z/np.dot(c,z)
