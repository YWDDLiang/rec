import itertools,numpy as np,pytest
from rec_lab.science import *
from rec_lab.acquisition import *

@pytest.mark.parametrize('seed',range(5))
def test_coverage_diminishing_returns_and_greedy(seed):
    rng=np.random.default_rng(seed);P=rng.random((6,4));w=np.ones(4)
    A=[0];B=[0,1,2];i=4
    assert coverage_value(P,w,A+[i])-coverage_value(P,w,A)>=coverage_value(P,w,B+[i])-coverage_value(P,w,B)-1e-10
    S,v=greedy_coverage(P,w,3);best=max(coverage_value(P,w,s) for s in itertools.combinations(range(6),3))
    assert v>=(1-1/np.e)*best-1e-10

def test_information_and_posterior():
    X=np.array([[1.,0],[0,1],[1,1]]);C=np.eye(2)
    v=gaussian_information(X,C,1,[0]);assert v==pytest.approx(.5*np.log(2))
    m,c=posterior_update([0,0],C,[1,0],2,1)
    assert np.allclose(m,[1,0]) and np.allclose(c,np.diag([.5,1]))
    assert np.linalg.eigvalsh(c).min()>0
    with pytest.raises(ValueError):posterior_update([0,0],-C,[1,0],2,1)

def test_eligibility_and_time():
    c=ScientificCandidate('a','doi:verified',8,frozenset({'XRD'}),5)
    assert c.eligible(16,{'XRD'},6)
    assert not c.eligible(4,{'XRD'},6) and not c.eligible(16,{'XRD'},4)
    S,_=greedy_coverage(np.eye(3),[1,1,1],2,[True,False,False]);assert S==[0]

def test_noise_correction_exact():
    t=np.array([.4,-1]);y=np.array([1,0]);e=.1
    # Expectation over independent label flip equals clean logistic loss.
    corrected=(1-e)*noise_corrected_logistic_loss(t,y,e)+e*noise_corrected_logistic_loss(t,1-y,e)
    clean=np.logaddexp(0,t)-y*t
    assert np.allclose(corrected,clean)
    with pytest.raises(ValueError):noise_corrected_logistic_loss(t,y,.5)

def test_synthetic_preference_is_not_verified():
    good=PreferencePair('p','a','b','expert',.9,'expert-panel-1','train')
    synthetic=PreferencePair('q','a','b','synthetic',.99,'model','train')
    accepted=accept_pairs([good,synthetic])
    assert len(accepted)==1 and accepted[0]==good

def test_acquisition_uses_margin_before_cost():
    order,score=rank_acquisition([1.],[[3.,2.]],[[0.,0.]],[100.,1.],current_margin=2.)
    assert order[0]==0 and np.allclose(score,[.01,0.])


def test_verified_method_dataset_bundle():
    from rec_lab.science import MethodDatasetBundle
    m=ScientificCandidate("m","paper:m",4.,frozenset(),0.)
    d=ScientificCandidate("d","data:d",2.,frozenset(),0.)
    b=MethodDatasetBundle("b",m,d,"execution:001","execution",1.,True)
    assert b.eligible(8,[],2)
    assert not b.eligible(8,[],0.5)
    fake=MethodDatasetBundle("fake",m,d,"generated","llm_self_judgment",1.,True)
    assert not fake.eligible(8,[],2)
