import numpy as np,pytest
from rec_lab.atoms import *
from rec_lab.trajectory import *

@pytest.mark.parametrize('seed',range(5))
def test_nuisance_invariance(seed):
    rng=np.random.default_rng(seed);Z=rng.normal(size=(30,3));G=rng.normal(size=(30,9));C=rng.normal(size=(3,9));w=rng.uniform(.3,2,30)
    R,_=nuisance_residual(G,Z,w);R2,_=nuisance_residual(G+Z@C,Z,w)
    assert np.allclose(R,R2,atol=1e-10)
    assert np.allclose(Z.T@(w[:,None]*R),0,atol=1e-9)

def test_crossfit_not_falsely_same_sample_projection():
    rng=np.random.default_rng(1);G=rng.normal(size=(20,3));Z=np.c_[np.ones(20),rng.normal(size=20)]
    R=crossfit_residual(G,Z,np.arange(20)%2)
    assert not np.allclose(Z.T@R,0)
    with pytest.raises(ValueError):crossfit_residual(G,Z,np.zeros(20))

def test_sparse_signed_atoms_and_fit():
    D=np.eye(3);X=np.array([[1.,0,-2],[0,2,0],[1,0,0],[0,0,2]])
    A=omp_encode(X,D,2)
    assert np.allclose(A@D,X) and A.min()<0
    fit=learn_dictionary(X,n_atoms=3,sparsity=2,iterations=5,seed=2)
    assert fit['mse']>=0 and np.isfinite(fit['mse'])
    assert np.all((abs(fit['codes'])>1e-12).sum(1)<=2)

def test_projection_seed_and_invalid_sparsity():
    X=np.eye(3);p,r=random_projection(X,5,3);p2,_=random_projection(X,5,3)
    assert np.allclose(p,p2)
    with pytest.raises(ValueError):omp_encode(X,X,4)

@pytest.mark.parametrize('seed',range(5))
def test_adjoint_matches_finite_difference(seed):
    rng=np.random.default_rng(seed);T=4;D=3;M=rng.normal(size=(T,D,D));H=np.einsum('tji,tjk->tik',M,M)
    b=rng.normal(size=(T,D));eta=np.ones(T)*.03;w=rng.uniform(.5,1.5,T);t=rng.normal(size=D);goal=rng.normal(size=D)
    states=rollout_quadratics(t,H,b,eta,w);v=states[-1]-goal
    effect=adjoint_weight_effects(states,H,b,eta,v,w);num=[]
    def obj(w):
        f=rollout_quadratics(t,H,b,eta,w)[-1]-goal;return .5*f@f
    for j in range(T):
        delta=np.eye(T)[j]*1e-5;num.append((obj(w+delta)-obj(w-delta))/2e-5)
    assert np.allclose(effect,num,atol=1e-7)

def test_order_bracket_and_null():
    t=np.array([.3,.8]);Ha=np.diag([1.,2.]);Hb=np.array([[2.,.5],[.5,1.]])
    a=np.array([.8,0]);b=np.array([-.1,1.]);eta=.2
    ab=quadratic_step(quadratic_step(t,Ha,a,eta),Hb,b,eta)
    ba=quadratic_step(quadratic_step(t,Hb,b,eta),Ha,a,eta)
    assert np.allclose(ab-ba,eta**2*order_bracket(t,Ha,a,Hb,b))
    assert not np.allclose(ab,ba)
    assert np.allclose(order_bracket(t,Ha,a,Ha,a),0)
