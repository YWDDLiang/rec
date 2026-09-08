"""IDEA-09/10. Evidence-set recommendation and Gaussian experiment design.

All scores must come with external evidence. Toy features are not chemistry.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .numerics import array

@dataclass(frozen=True)
class ScientificCandidate:
    candidate_id: str
    evidence_id: str
    required_memory_gb: float
    required_equipment: frozenset[str]
    available_at: float
    def eligible(self,memory_gb,equipment,decision_time):
        if not self.candidate_id or not self.evidence_id: return False
        if not np.isfinite(self.required_memory_gb) or self.required_memory_gb<0 or not np.isfinite(self.available_at): return False
        if not np.isfinite(memory_gb) or memory_gb<0 or not np.isfinite(decision_time): return False
        return (self.required_memory_gb<=memory_gb and
                self.required_equipment.issubset(set(equipment)) and
                self.available_at<=decision_time)

def coverage_value(probabilities,weights,selected):
    P=array(probabilities,2);w=array(weights,1);S=list(selected)
    if P.shape[1]!=len(w) or np.any((P<0)|(P>1)) or np.any(w<0): raise ValueError("invalid coverage")
    if len(set(S))!=len(S) or any(i<0 or i>=len(P) for i in S): raise ValueError("invalid set")
    if not S: return 0.
    # Independent per-capability success model; diminishing returns algebra
    # remains true for this defined objective even if its probabilities misfit.
    return float(w@(1-np.prod(1-P[S],axis=0)))

def greedy_coverage(probabilities,weights,k,eligible=None):
    P=array(probabilities,2)
    if not 0<=k<=len(P): raise ValueError("invalid cardinality")
    E=np.ones(len(P),dtype=bool) if eligible is None else np.asarray(eligible,dtype=bool)
    if E.shape!=(len(P),): raise ValueError("eligibility shape")
    S=[]
    for _ in range(min(k,int(E.sum()))):
        available=[i for i in range(len(P)) if E[i] and i not in S]
        i=max(available,key=lambda j:coverage_value(P,weights,S+[j]))
        S.append(i)
    return S,coverage_value(P,weights,S)

def gaussian_information(features,covariance,noise_variance,selected):
    X=array(features,2);Sigma=array(covariance,2);S=list(selected)
    d=X.shape[1]
    if Sigma.shape!=(d,d) or noise_variance<=0 or not np.allclose(Sigma,Sigma.T): raise ValueError("invalid Gaussian model")
    eig,U=np.linalg.eigh(Sigma)
    if np.min(eig)<-1e-10: raise ValueError("covariance not PSD")
    if len(set(S))!=len(S) or any(i<0 or i>=len(X) for i in S): raise ValueError("bad design set")
    if not S: return 0.
    root=(U*np.sqrt(np.maximum(eig,0)))@U.T
    M=np.eye(d)+root@X[S].T@X[S]@root/noise_variance
    sign,value=np.linalg.slogdet(M)
    if sign<=0: raise RuntimeError("nonpositive information determinant")
    return .5*float(value)

def greedy_information(features,covariance,noise_variance,k,eligible=None):
    X=array(features,2)
    if not 0<=k<=len(X): raise ValueError("invalid cardinality")
    E=np.ones(len(X),bool) if eligible is None else np.asarray(eligible,bool)
    if E.shape!=(len(X),): raise ValueError("eligibility shape")
    S=[]
    for _ in range(min(k,int(E.sum()))):
        cand=[i for i in range(len(X)) if E[i] and i not in S]
        S.append(max(cand,key=lambda i:gaussian_information(X,covariance,noise_variance,S+[i])))
    return S,gaussian_information(X,covariance,noise_variance,S)

def posterior_update(mean,covariance,x,y,noise_variance):
    mu=array(mean,1);S=array(covariance,2);x=array(x,1)
    if x.shape!=mu.shape or S.shape!=(len(mu),len(mu)) or noise_variance<=0: raise ValueError("shape/noise")
    if not np.isfinite(y) or not np.allclose(S,S.T) or np.linalg.eigvalsh(S).min()<-1e-10: raise ValueError("invalid observation/covariance")
    k=S@x/(noise_variance+x@S@x)
    return mu+k*(y-x@mu),S-np.outer(k,x@S)


@dataclass(frozen=True)
class MethodDatasetBundle:
    """Evidence-bearing executable pair. No LLM-generated compatibility accepted."""
    bundle_id: str
    method: ScientificCandidate
    dataset: ScientificCandidate
    compatibility_evidence_id: str
    verifier: str
    verified_at: float
    compatible: bool

    def eligible(self,memory_gb,equipment,decision_time):
        if self.verifier not in {"execution", "expert", "documented_interface"}:
            return False
        if not self.bundle_id or not self.compatibility_evidence_id or not self.compatible:
            return False
        if not np.isfinite(self.verified_at) or self.verified_at>decision_time:
            return False
        # Conservative sequential-pipeline convention: max memory. Concurrent
        # execution needs a different resource model; this method does not infer it.
        return (self.method.eligible(memory_gb,equipment,decision_time) and
                self.dataset.eligible(memory_gb,equipment,decision_time))
