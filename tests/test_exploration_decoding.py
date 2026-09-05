import numpy as np,pytest
from rec_lab.exploration import *
from rec_lab.decoding import *

def test_pagerank_dangling_and_stationarity():
    W=np.array([[0.,1],[0,0]]);v=np.array([.8,.2]);r=pagerank(W,v)
    P=np.array([[0.,1],v])
    assert np.allclose(r,.85*P.T@r+.15*v,atol=1e-10)
    assert np.isclose(r.sum(),1)

def test_absorbing_value_and_divergence():
    Q=np.array([[.5,0],[.1,.4]]);r=np.ones(2);value=absorbing_value(Q,r)
    assert np.allclose(value,r+Q@value)
    with pytest.raises(ValueError):absorbing_value(np.eye(2),r)

def test_safety_mixture():
    p=np.array([1.,0]);e=np.array([0.,1]);q,epsilon=safe_mixture(p,e,[.9,.3],.6)
    assert epsilon==pytest.approx(.5)
    assert q@np.array([.9,.3])>=.6-1e-12

def test_kl_tilt_not_additive_mixture():
    p=np.array([.8,.2]);r=np.array([.1,.9]);q=kl_tilt(p,np.log(r),1)
    assert np.allclose(q,p*r/(p@r))
    assert not np.allclose(q,.5*p+.5*r)
    assert coverage_new_probability([.2,.3,.5],[True,False,False])==pytest.approx(.8)

def test_trie_prefix_eos_and_collisions():
    trie=CatalogTrie([[1],[1,2],[3,2]])
    q=np.array([.2,.3,.5])
    assert all(abs(trie.item_path_probability(i,q)-q[i])<1e-12 for i in range(3))
    with pytest.raises(ValueError):CatalogTrie([[1],[1]])
    with pytest.raises(ValueError):CatalogTrie([[-1]])

@pytest.mark.parametrize('seed',range(8))
def test_speculative_exact_identity(seed):
    rng=np.random.default_rng(seed);p=rng.dirichlet(np.ones(7));q=rng.dirichlet(np.ones(7))
    out,acceptance=speculative_identity(p,q)
    assert np.allclose(out,p)
    assert acceptance==pytest.approx(1-.5*np.abs(p-q).sum())

def test_speculation_zero_support_and_monte_carlo():
    p=np.array([.1,.7,.2]);q=np.array([1.,0,0]);out,acc=speculative_identity(p,q)
    assert np.allclose(out,p) and acc==pytest.approx(.1)
    rng=np.random.default_rng(9);a=[exact_speculative_step(p,q,rng)[0] for _ in range(4000)]
    assert np.max(np.abs(np.bincount(a,minlength=3)/4000-p))<.035


def test_trie_bad_distribution_is_not_zero_probability():
    trie=CatalogTrie([(1,),(2,)])
    with pytest.raises(ValueError): trie.item_path_probability(0,[-1,2])
    with pytest.raises(ValueError): trie.item_path_probability(-1,[.5,.5])
