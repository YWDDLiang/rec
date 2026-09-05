import numpy as np,pytest
from rec_lab.frontier import *

@pytest.mark.parametrize('seed',range(10))
def test_primal_dual(seed):
    A=np.random.default_rng(seed).normal(size=(3,11));r=solve_frontier(A)
    assert r.certified and r.gap<1e-6
    assert r.weights.min()>=0 and np.isclose(r.weights.sum(),1)
    assert np.isclose(r.margin,min(A@r.weights))
    assert np.isclose(r.dual_upper,max(r.witness@A))

def test_pool_gap_not_intrinsic_conflict():
    V=np.eye(2);G=np.array([[1,-2],[-2,1]])
    assert parameter_feasibility(V)['margin']>0
    assert solve_frontier(response_matrix(V,G)).margin<0
    V2=np.array([[1.,0],[-1.,0]])
    assert abs(parameter_feasibility(V2)['margin'])<1e-8

def test_acquisition_witness_necessary_not_sufficient():
    A=np.array([[1.,-1.],[-1.,1.]])
    old=solve_frontier(A);lam=old.witness
    # No violation leaves the old dual witness feasible.
    new=np.array([-2.,-2.]);r=solve_frontier(np.c_[A,new])
    assert lam@new<=old.margin and np.isclose(r.margin,old.margin)
    # Violation of ONE optimum witness is not sufficient: another witness may bind.
    base=np.array([[0.],[0.]]);w=np.array([1.,0.]);new=np.array([1.,0.])
    assert w@new>0 and solve_frontier(np.c_[base,new]).margin==pytest.approx(0)

def test_floor_is_optimized_not_added_afterwards():
    A=np.array([[2.,-1.],[-.5,1.]]);p=np.array([.9,.1]);eps=.2
    r=solve_frontier(response_with_floor(A,p,eps));q=eps*p+(1-eps)*r.weights
    assert np.all(q>=eps*p-1e-10)
    assert np.allclose(A@q,r.target_gains)

@pytest.mark.parametrize('seed',range(7))
def test_priced_atoms_matches_full_frontier(seed):
    A=np.random.default_rng(seed).normal(size=(4,17));r,c=column_generation(A,[0])
    exact=solve_frontier(A)
    assert r.certified and abs(r.margin-exact.margin)<1e-6

def test_quadratic_descent_bound_and_noise_penalty():
    theta=np.array([.4,-.6]);d=np.array([.2,-.1]);eta=.1
    H=np.diag([1.,3.]);V=np.array([H@theta]);L=3.
    actual=.5*theta@H@theta-.5*(theta-eta*d)@H@(theta-eta*d)
    lower=descent_lower_bound(V,d,eta,L)[0]
    assert actual>=lower-1e-10
    stochastic=stochastic_descent_bound(V,d,eta,L,10.,2)
    assert stochastic[0]<lower

def test_zero_direction_and_rejections():
    assert np.all(descent_lower_bound(np.eye(2),[0,0],.1,1)==0)
    for bad in [np.empty((2,0)),np.array([[np.nan]])]:
        with pytest.raises(ValueError):solve_frontier(bad)
    with pytest.raises(ValueError):response_with_floor([[1]], [1],1.1)

def test_scalarization_destroys_objective_information():
    a=np.array([1.,0.]);c=np.array([0.,1.])
    ga=np.array([a+c,a-c]);gb=np.array([a-c,a+c])
    assert np.allclose(ga.mean(0),gb.mean(0))
    assert not np.allclose(np.array([.8,.2])@ga,np.array([.8,.2])@gb)

def test_dual_face_repairs_single_witness_false_positive():
    A=np.zeros((2,1));B=np.array([[1.],[0.]])
    assert np.array([1.,0.])@B[:,0]>0
    face=dual_face_gain(A,B)
    assert not face['positive'] and abs(face['face_gain'])<1e-7

def test_complementary_acquisition_not_submodular():
    from rec_lab.acquisition import best_acquisition_batch
    A=np.zeros((2,1));B=np.array([[2.,-1.],[-1.,2.]])
    # Each singleton has zero gain; together they enable strict common descent.
    assert solve_frontier(np.c_[A,B[:,0]]).margin==pytest.approx(0)
    assert solve_frontier(np.c_[A,B[:,1]]).margin==pytest.approx(0)
    joint=solve_frontier(np.c_[A,B]);face=dual_face_gain(A,B)
    assert joint.margin==pytest.approx(.5) and face['face_gain']==pytest.approx(.5)
    out=best_acquisition_batch(A,B,[1,1],2)
    assert out['selected']==[0,1] and out['gain']==pytest.approx(.5)

@pytest.mark.parametrize('seed',range(5))
def test_face_gain_directional_derivative(seed):
    rng=np.random.default_rng(seed);A=rng.normal(size=(3,5));B=rng.normal(size=(3,3))
    old=solve_frontier(A).margin;face=dual_face_gain(A,B)['face_gain'];eps=1e-5
    # Cartesian columns allow independent old/new mixtures with fixed mass eps.
    mixed=np.stack([(1-eps)*A[:,i]+eps*B[:,j] for i in range(5) for j in range(3)],axis=1)
    fd=(solve_frontier(mixed).margin-old)/eps
    assert abs(fd-face)<2e-3


def test_smooth_step_rejects_overshoot_and_selects_safe_direction():
    from rec_lab.frontier import solve_smooth_step
    # L(theta)=.5(theta-1)^2 at theta=0; raw LP prefers huge gradient.
    V=np.array([[-1.]])
    G=np.array([[-100.],[-1.]])
    raw=solve_frontier(V@G.T)
    assert raw.weights[0]>0.99
    safe=solve_smooth_step(V,G,.1,[1.])
    assert safe['accepted']
    after=-.1*safe['direction'][0]
    improvement=.5-.5*(after-1)**2
    assert improvement>=safe['lower_bounds'][0]-1e-7
    assert improvement>.49


def test_smooth_step_large_uncertainty_abstains():
    from rec_lab.frontier import solve_smooth_step
    r=solve_smooth_step([[-1.]],[[-1.]],.1,[1.],gradient_error=[2.])
    assert not r['accepted']
    assert np.allclose(r['direction'],0)
