import numpy as np,pytest
from rec_lab.observational import *
from rec_lab.allocation import *

def test_mar_dr_exact_enumeration():
    # Identical x, deterministic y=.8, R frequencies exactly e=.25.
    y=np.array([.8,np.nan,np.nan,np.nan]);r=np.array([1,0,0,0],bool)
    est=missing_at_random_mean(y,r,np.full(4,.25),np.full(4,.1))
    assert est['estimate']==pytest.approx(.8)
    assert est['pseudo_outcomes'][0]>1 # must not feed blindly into BCE

def test_dr_reward_model_correct_even_behavior_misspecified():
    y=np.array([.2,.8]);mu=np.array([[.2,.8],[.2,.8]]);pi=np.array([[.3,.7],[.3,.7]])
    assert dr_policy_value(y,[0,1],[.8,.2],pi,mu)['estimate']==pytest.approx(.62)

def test_dr_behavior_correct_reward_model_wrong():
    # Four equally weighted logged events, mu=0; logging frequencies .75/.25.
    out=dr_policy_value([.2,.2,.2,.8],[0,0,0,1],[.75,.75,.75,.25],np.tile([.3,.7],(4,1)),np.zeros((4,2)))
    assert out['estimate']==pytest.approx(.62)

@pytest.mark.parametrize('p',[0.,-1.,1.1,np.nan])
def test_propensity_invalid(p):
    with pytest.raises(ValueError):missing_at_random_mean([1],[1],[p],[.5])

def test_action_indices_not_rounded():
    with pytest.raises(ValueError):dr_policy_value([1],[.5],[1],[[1]],[[0]])

def test_maturity_and_unknown_feedback():
    assert mature_mask([0,4],5,7).tolist()==[True,False]
    y,m=validate_feedback_labels([[1,np.nan]],[[True,False]])
    assert y[0,1]==0 and not m[0,1]

@pytest.mark.parametrize('seed',range(5))
def test_scalar_allocation_optimal_and_minimax(seed):
    rng=np.random.default_rng(seed);p=rng.uniform(.1,1,6);p/=p.sum();a=rng.uniform(.1,2,6)
    q=scalar_variance_optimal(p,a);value=variance_objectives(p,q,a[:,None])[0]
    assert value<=variance_objectives(p,p,a[:,None])[0]+1e-8
    opt=minimax_allocation(p,a[:,None],floor=.001)['q']
    assert np.allclose(q,opt,atol=2e-5)

def test_vector_minimax_floor():
    p=np.array([.6,.3,.1]);A=np.array([[1,9],[4,1],[2,3.]])
    res=minimax_allocation(p,A,.1);q=res['q']
    assert np.all(q>=.1*p-1e-8) and np.isclose(q.sum(),1)
    assert max(variance_objectives(p,q,A))<=max(variance_objectives(p,p,A))+1e-7

def test_importance_unbiased_exact_enumeration():
    p=np.array([.8,.2]);q=np.array([.25,.75]);X=np.array([[1.,2],[4.,-1]])
    # Four samples exactly realize q.
    got=corrected_estimate(X,[0,1,1,1],p,q)
    assert np.allclose(got,p@X)
    with pytest.raises(ValueError):corrected_estimate(X,[0],p,[1,0])

def test_cost_allocation():
    n=continuous_cost_allocation([.5,.5],[1,4],[1,4],100)
    assert np.isclose(n@np.array([1,4]),100) and np.allclose(n,[20,20])


def test_corrected_estimate_rejects_fractional_indices():
    with pytest.raises(ValueError): corrected_estimate([1.,2.],[0.7],[.5,.5],[.5,.5])
