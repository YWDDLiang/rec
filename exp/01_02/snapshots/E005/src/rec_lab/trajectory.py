"""IDEA-03. Local order effects and exact finite-trajectory adjoint sensitivity."""
from __future__ import annotations
import numpy as np
from .numerics import array

def quadratic_step(theta,H,b,eta,weight=1.):
    t=array(theta,1);H=array(H,2);b=array(b,1)
    if H.shape!=(len(t),len(t)) or b.shape!=t.shape: raise ValueError("shape mismatch")
    return t-eta*weight*(H@t-b)

def rollout_quadratics(theta, hessians, linear_terms, rates, weights=None):
    H=array(hessians,3);b=array(linear_terms,2);eta=array(rates,1)
    T=len(H);w=np.ones(T) if weights is None else array(weights,1)
    if len(b)!=T or len(eta)!=T or len(w)!=T: raise ValueError("trajectory length mismatch")
    states=[array(theta,1).copy()]
    for k in range(T): states.append(quadratic_step(states[-1],H[k],b[k],eta[k],w[k]))
    return np.array(states)

def adjoint_weight_effects(states,hessians,linear_terms,rates,terminal_gradient,weights=None):
    """d terminal loss / d w_t, including propagation through later updates."""
    states=array(states,2);H=array(hessians,3);b=array(linear_terms,2);eta=array(rates,1)
    T=len(H);w=np.ones(T) if weights is None else array(weights,1)
    if len(states)!=T+1 or len(b)!=T or len(eta)!=T or len(w)!=T: raise ValueError("length mismatch")
    v=array(terminal_gradient,1).copy();effect=np.zeros(T)
    for t in reversed(range(T)):
        g=H[t]@states[t]-b[t]
        effect[t]=-eta[t]*np.dot(v,g)
        v=(np.eye(len(v))-eta[t]*w[t]*H[t]).T@v
    return effect

def order_bracket(theta,H_a,b_a,H_b,b_b):
    """theta_{a then b} - theta_{b then a} = eta² bracket for quadratics."""
    t=array(theta,1);Ha=array(H_a,2);Hb=array(H_b,2)
    ga=Ha@t-array(b_a,1);gb=Hb@t-array(b_b,1)
    return Hb@ga-Ha@gb
