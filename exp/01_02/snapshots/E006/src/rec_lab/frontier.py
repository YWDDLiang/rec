"""IDEA-01. Data-realizable joint improvement; exact finite-pool LP certificates.

No gradient matching to the original training mixture is used. A[m,i] is a
signed *target-loss descent response*, not a positive-only importance score.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from .numerics import array, simplex

@dataclass
class FrontierResult:
    weights: np.ndarray
    margin: float
    target_gains: np.ndarray
    witness: np.ndarray
    dual_upper: float
    gap: float
    certified: bool
    scope: str = "provided response matrix at one parameter state"

def solve_frontier(responses, tolerance=1e-7) -> FrontierResult:
    """max_{q in simplex} min_m A[m]q, with independently solved dual.

    All objectives are losses to minimize. Negative margin is allowed; the
    caller MUST reject an update that violates its guardrails. Append a zero
    response column to explicitly represent abstaining. No missing-data or
    causal assumptions are silently supplied here.
    """
    A = array(responses, 2, "responses")
    m,n=A.shape
    if not m or not n: raise ValueError("empty response matrix")
    primal=linprog(np.r_[np.zeros(n),-1.], A_ub=np.c_[-A,np.ones(m)],
        b_ub=np.zeros(m), A_eq=np.array([np.r_[np.ones(n),0.]]),
        b_eq=[1.], bounds=[(0,None)]*n+[(None,None)],method="highs")
    dual=linprog(np.r_[np.zeros(m),1.], A_ub=np.c_[A.T,-np.ones(n)],
        b_ub=np.zeros(n), A_eq=np.array([np.r_[np.ones(m),0.]]),
        b_eq=[1.],bounds=[(0,None)]*m+[(None,None)],method="highs")
    if not primal.success or not dual.success:
        raise RuntimeError(f"LP failed: {primal.message}; {dual.message}")
    q=np.maximum(primal.x[:n],0);q/=q.sum()
    lam=np.maximum(dual.x[:m],0);lam/=lam.sum()
    gains=A@q; low=float(gains.min()); up=float((lam@A).max())
    gap=up-low
    return FrontierResult(q,low,gains,lam,up,gap,
        bool(gap>=-tolerance and gap<=tolerance*(1+abs(up))))

def parameter_feasibility(target_gradients):
    """Common-descent feasibility in a bounded full update space ||d||inf<=1.

    Positive margin means a parameter direction exists, NOT that this pool
    contains it or that finite-step test performance improves.
    """
    V=array(target_gradients,2,"target_gradients")
    m,d=V.shape
    if not m or not d: raise ValueError("empty gradients")
    opt=linprog(np.r_[np.zeros(d),-1.],A_ub=np.c_[-V,np.ones(m)],
        b_ub=np.zeros(m),bounds=[(-1,1)]*d+[(None,None)],method="highs")
    if not opt.success: raise RuntimeError(opt.message)
    return {"direction":opt.x[:d],"margin":float(opt.x[-1])}

def response_matrix(target_gradients, update_directions, scales=None):
    V=array(target_gradients,2);G=array(update_directions,2)
    if V.shape[1]!=G.shape[1]: raise ValueError("parameter coordinate mismatch")
    s=np.ones(V.shape[0]) if scales is None else array(scales,1)
    if s.shape!=(V.shape[0],) or np.any(s<=0): raise ValueError("invalid scales")
    return (V@G.T)/s[:,None]

def response_with_floor(A, base, floor):
    """Reparameterize q=floor*base+(1-floor)*r exactly, not post-hoc."""
    A=array(A,2);p=simplex(base)
    if len(p)!=A.shape[1] or not 0<=floor<=1: raise ValueError("invalid floor")
    return floor*(A@p)[:,None]+(1-floor)*A

def column_generation(A, initial, tolerance=1e-7, max_rounds=None):
    """Atoms may nominate initial columns; the dual scans ALL pool columns.

    A final certificate refers to the complete supplied pool only after the
    pricing residual is small. Failure to terminate is explicitly reported.
    """
    A=array(A,2); n=A.shape[1]
    cols=sorted(set(int(i) for i in initial))
    if not cols or cols[0]<0 or cols[-1]>=n: raise ValueError("bad initial set")
    rounds=n if max_rounds is None else max_rounds
    if rounds<1: raise ValueError("max_rounds must be positive")
    res=None; violation=float('inf')
    for _ in range(rounds):
        res=solve_frontier(A[:,cols],tolerance)
        values=res.witness@A
        j=int(values.argmax()); violation=float(values[j]-res.margin)
        if violation <= tolerance*(1+abs(res.margin)): break
        if j in cols: break
        cols.append(j)
    # Important: re-solve after the final added column before reporting weights.
    res=solve_frontier(A[:,cols],tolerance)
    up=float((res.witness@A).max()); gap=up-res.margin
    q=np.zeros(n);q[cols]=res.weights
    return FrontierResult(q,res.margin,A@q,res.witness,up,gap,
        bool(gap<=tolerance*(1+abs(up))),"all supplied columns, pricing checked"),cols

def descent_lower_bound(target_gradients, direction, step, smoothness,
                        gradient_error=0., direction_error=0.):
    """Deterministic Euclidean smoothness bound for L(theta)-L(theta-eta*d).

    Gradient and direction error bounds must be justified externally. This is
    not an Adam update certificate unless `direction` is the actual update.
    """
    V=array(target_gradients,2);d=array(direction,1)
    if V.shape[1]!=len(d) or step<0: raise ValueError("invalid direction/step")
    L=np.broadcast_to(array(smoothness),(len(V),))
    e=np.broadcast_to(array(gradient_error),(len(V),))
    if np.any(L<0) or np.any(e<0) or direction_error<0: raise ValueError("negative bound")
    norm=np.linalg.norm(d)
    first=V@d-e*norm-(np.linalg.norm(V,axis=1)+e)*direction_error
    return step*first-.5*L*step**2*(norm+direction_error)**2

def stochastic_descent_bound(target_gradients, mean_direction, step,
                             smoothness, trace_covariance, batch_size):
    """IID with-replacement mean update; E||d_hat||²=||d||²+tr(Cov)/B."""
    if batch_size<1 or trace_covariance<0: raise ValueError("invalid sampling")
    base=descent_lower_bound(target_gradients,mean_direction,step,smoothness)
    return base-.5*np.asarray(smoothness)*step**2*trace_covariance/batch_size

def dual_face_gain(existing_responses,new_responses,tolerance=1e-9):
    """Obstruction over the ENTIRE old optimal dual face, not one witness.

    h(B)=min_{lambda in Lambda*} max_{b in B} lambda^T b - t*.
    Exact arithmetic: adding B improves the finite-pool max-min margin iff h>0.
    Also equals the right derivative when assigning epsilon total mass to B.
    Numerical tolerance relaxes the dual-face constraints; returned 'positive'
    requires a safety tolerance. Existing and new responses must share units.
    This is standard LP sensitivity applied to data acquisition, not a claim of
    a novel general optimization theorem.
    """
    A=array(existing_responses,2);B=array(new_responses,2)
    if A.shape[0]!=B.shape[0] or not B.shape[1]:raise ValueError('objective/new-pool mismatch')
    old=solve_frontier(A);m=A.shape[0]
    # variables lambda_m,z; enforce membership in old optimal face.
    ub=np.vstack([np.c_[A.T,np.zeros(A.shape[1])],np.c_[B.T,-np.ones(B.shape[1])]])
    rhs=np.r_[np.full(A.shape[1],old.dual_upper+tolerance),np.zeros(B.shape[1])]
    opt=linprog(np.r_[np.zeros(m),1.],A_ub=ub,b_ub=rhs,
                A_eq=[np.r_[np.ones(m),0.]],b_eq=[1.],
                bounds=[(0,None)]*m+[(None,None)],method='highs')
    if not opt.success:raise RuntimeError(opt.message)
    gap=float(opt.x[-1]-old.margin)
    return {'face_gain':gap,'obstructing_witness':opt.x[:m],
            'positive':gap>max(10*tolerance,1e-7),'old_margin':old.margin,
            'scope':'fixed finite response matrices; uncertain responses require valid lower bounds'}

def solve_smooth_step(target_gradients,update_directions,step,smoothness,
                      gradient_error=0.,base=None,floor=0.,tolerance=1e-7):
    """Maximize the worst externally-justified smoothness lower bound.

    Convex finite-direction problem, solved numerically by SLSQP. This verifies
    feasibility, NOT a dual gap or global numerical optimum certificate.
    If any lower bound is nonpositive, no update is returned (abstention).
    Assumptions such as population-gradient error and valid neighborhood-wide
    smoothness must come from the caller; sample curvature is not sufficient.
    """
    from scipy.optimize import minimize
    V=array(target_gradients,2);G=array(update_directions,2)
    if V.shape[1]!=G.shape[1] or not len(V) or not len(G) or step<=0:raise ValueError('shape/step')
    n=len(G);p=np.full(n,1/n) if base is None else simplex(base)
    if p.shape!=(n,) or not 0<=floor<=1:raise ValueError('base/floor')
    L=np.broadcast_to(array(smoothness),(len(V),));err=np.broadcast_to(array(gradient_error),(len(V),))
    if np.any(L<0) or np.any(err<0):raise ValueError('negative bounds')
    def lower(q):return descent_lower_bound(V,q@G,step,L,err)
    x0=np.r_[p,float(lower(p).min())]
    opt=minimize(lambda x:-x[-1],x0,bounds=[(floor*v,1.) for v in p]+[(None,None)],
                 constraints=[{'type':'eq','fun':lambda x:x[:-1].sum()-1},
                              {'type':'ineq','fun':lambda x:lower(x[:-1])-x[-1]}],
                 method='SLSQP',options={'maxiter':2000,'ftol':1e-11})
    q=opt.x[:-1]
    if not opt.success or abs(q.sum()-1)>tolerance or np.any(q<floor*p-tolerance):
        raise RuntimeError('smooth-step solver failed: '+opt.message)
    q=np.maximum(q,0);q/=q.sum();bounds=lower(q)
    if np.min(bounds-opt.x[-1]) < -tolerance:raise RuntimeError('smooth-step feasibility failed')
    accept=bool(bounds.min()>tolerance)
    return {'weights':q,'lower_bounds':bounds,'direction':q@G if accept else np.zeros(G.shape[1]),
            'accepted':accept,'scope':'conditional on external smoothness/error bounds; numerical feasibility only'}

def solve_capped_frontier(responses, max_weight_multiple=2., floor=0., tolerance=1e-7):
    """Local LP with q_i in [floor/n, multiple/n]; no global-pool certificate."""
    A=array(responses,2)
    m,n=A.shape
    if not m or not n or max_weight_multiple<1 or not 0<=floor<=1:
        raise ValueError("Invalid response matrix or weight cap")
    cap=min(1.,max_weight_multiple/n);lower=floor/n
    c=np.r_[np.zeros(n),-1.]
    Aub=np.c_[-A,np.ones(m)];Aeq=np.array([np.r_[np.ones(n),0.]])
    result=linprog(c,A_ub=Aub,b_ub=np.zeros(m),A_eq=Aeq,b_eq=[1.],
                   bounds=[(lower,cap)]*n+[(None,None)],method="highs")
    if not result.success:raise RuntimeError(result.message)
    q=result.x[:n]
    dual_min=float(result.eqlin.marginals.sum()
        +lower*result.lower.marginals[:n].sum()+cap*result.upper.marginals[:n].sum())
    residual=c-Aub.T@result.ineqlin.marginals-Aeq.T@result.eqlin.marginals-result.lower.marginals-result.upper.marginals
    gains=A@q;value=float(gains.min());upper=-dual_min;gap=upper-value
    valid=(np.max(np.abs(residual))<=tolerance and abs(q.sum()-1)<=tolerance
           and q.min()>=lower-tolerance and q.max()<=cap+tolerance
           and gap>=-tolerance and gap<=tolerance*(1+abs(upper)))
    witness=-result.ineqlin.marginals
    return FrontierResult(q,value,gains,witness,upper,gap,bool(valid),
                          "box-constrained queried columns only")
