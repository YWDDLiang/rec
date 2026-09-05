"""IDEA-02. Nuisance-aware gradient representations, not a Gradient Atoms replica.

Uses projected gradients + deterministic orthogonal matching pursuit and
alternating least squares. No EKFAC/preconditioned eigenspace is claimed.
"""
from __future__ import annotations
import numpy as np
from .numerics import array

def random_projection(gradients, dim, seed=0):
    G=array(gradients,2)
    if dim<1: raise ValueError("dim must be positive")
    R=np.random.default_rng(seed).normal(size=(G.shape[1],dim))/np.sqrt(dim)
    return G@R,R

def nuisance_residual(gradients, nuisance, weights=None):
    """Weighted Frisch-Waugh projection, fitted on this TRAINING fold only.

    Invariance to G -> G+ZC holds in exact arithmetic. It is an algebraic
    invariance, not causal identification of user interest.
    """
    G=array(gradients,2);Z=array(nuisance,2)
    if G.shape[0]!=Z.shape[0]: raise ValueError("row mismatch")
    w=np.ones(len(G)) if weights is None else array(weights,1)
    if w.shape!=(len(G),) or np.any(w<=0): raise ValueError("weights must be positive")
    W=np.sqrt(w)[:,None]
    coef=np.linalg.lstsq(W*Z,W*G,rcond=None)[0]
    return G-Z@coef,coef

def crossfit_residual(gradients,nuisance,folds):
    G=array(gradients,2);Z=array(nuisance,2);f=np.asarray(folds)
    if len(G)!=len(Z) or f.shape!=(len(G),) or len(set(f))<2:
        raise ValueError("need at least two nonempty folds")
    out=np.empty_like(G)
    for k in np.unique(f):
        tr=f!=k;te=~tr
        coef=np.linalg.lstsq(Z[tr],G[tr],rcond=None)[0]
        out[te]=G[te]-Z[te]@coef
    return out

def omp_encode(X,D,sparsity):
    X=array(X,2);D=array(D,2)
    if X.shape[1]!=D.shape[1] or not 1<=sparsity<=len(D): raise ValueError("shape/sparsity")
    out=np.zeros((len(X),len(D)))
    for i,x in enumerate(X):
        chosen=[];res=x.copy()
        for _ in range(sparsity):
            corr=np.abs(D@res);corr[chosen]=-np.inf
            j=int(corr.argmax())
            if corr[j]<1e-12: break
            chosen.append(j)
            coef=np.linalg.lstsq(D[chosen].T,x,rcond=None)[0]
            res=x-coef@D[chosen]
        if chosen: out[i,chosen]=coef
    return out

def learn_dictionary(X,n_atoms=8,sparsity=2,iterations=12,seed=0):
    X=array(X,2)
    if not 1<=n_atoms<=len(X): raise ValueError("n_atoms out of range")
    if not 1<=sparsity<=n_atoms or iterations<1: raise ValueError("invalid configuration")
    rng=np.random.default_rng(seed)
    D=X[rng.choice(len(X),n_atoms,replace=False)].copy()
    for k in range(n_atoms):
        if np.linalg.norm(D[k])<1e-12: D[k]=rng.normal(size=X.shape[1])
    D/=np.linalg.norm(D,axis=1,keepdims=True)
    best=None
    for _ in range(iterations):
        A=omp_encode(X,D,sparsity)
        err=float(np.mean((X-A@D)**2))
        if best is None or err<best[0]: best=(err,A.copy(),D.copy())
        D=np.linalg.lstsq(A,X,rcond=None)[0]
        norms=np.linalg.norm(D,axis=1)
        for k in range(n_atoms):
            if norms[k]<1e-10: D[k]=X[rng.integers(len(X))]+1e-4*rng.normal(size=X.shape[1])
        D/=np.maximum(np.linalg.norm(D,axis=1,keepdims=True),1e-12)
    return {"codes":best[1],"dictionary":best[2],"mse":best[0],
            "backend":"OMP + alternating least squares; not upstream Gradient Atoms"}

def nominate_columns(codes, responses, per_atom=2):
    """Both signs matter; nominate on magnitude, then verify complete responses."""
    C=array(codes,2);A=array(responses,2)
    if len(C)!=A.shape[1] or per_atom<1: raise ValueError("invalid inputs")
    idx=set(np.argmax(A,axis=1).tolist())
    for k in range(C.shape[1]):
        idx.update(np.argsort(np.abs(C[:,k]))[-per_atom:].tolist())
    return sorted(idx)
