"""Conservative response uncertainty. NOT automatically calibrated on reused val."""
import numpy as np
from .numerics import array
from .frontier import solve_frontier

def robust_frontier(mean_response,radius):
    """Box uncertainty: min_{A in [hatA-U,hatA+U]} min_m (Aq)_m.

    q>=0 makes the componentwise lower matrix exact for this uncertainty set.
    Whether the set covers population responses is a separate statistical claim.
    """
    A=array(mean_response,2);U=array(radius,2)
    if A.shape!=U.shape or np.any(U<0):raise ValueError('invalid uncertainty box')
    return solve_frontier(A-U)

def bounded_response_estimate(per_record_target_gradients,updates,clip_norm,delta=.05):
    """Target gradient batches [objective][n_m, d], independent of fixed pool.

    Hoeffding simultaneous interval, using actual norm-clipped reference grads.
    Certificate concerns the CLIPPED target-gradient population. Reusing the same
    labels across adaptive model states is not justified by this fixed-state bound.
    """
    G=array(updates,2)
    if clip_norm<=0 or not 0<delta<1:raise ValueError('clip/delta')
    M=len(per_record_target_gradients);N=len(G)
    if M<1 or N<1:raise ValueError('empty targets or pool')
    means=[];radii=[]
    for raw in per_record_target_gradients:
        V=array(raw,2)
        if V.shape[1]!=G.shape[1] or len(V)<1:raise ValueError('shape')
        norms=np.linalg.norm(V,axis=1,keepdims=True)
        clipped=V*np.minimum(1.,clip_norm/np.maximum(norms,1e-30))
        means.append(clipped.mean(0)@G.T)
        B=clip_norm*np.linalg.norm(G,axis=1)
        radii.append(B*np.sqrt(2*np.log(2*M*N/delta)/len(V)))
    return np.asarray(means),np.asarray(radii)
