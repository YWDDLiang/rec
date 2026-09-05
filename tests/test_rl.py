import numpy as np,torch,pytest,itertools
from rec_lab.rl import *

def test_loo_score_gradient_unbiased_by_enumeration():
    p=np.array([.2,.3,.5]);reward=np.array([1.,.2,-.1]);g=np.zeros(3)
    for pair in itertools.product(range(3),repeat=2):
        pr=p[pair[0]]*p[pair[1]]
        r=torch.tensor(reward[list(pair)])[None,:,None]
        a=group_advantages(r,'loo').numpy()[0,:,0]
        estimate=sum(a[j]*(np.eye(3)[act]-p) for j,act in enumerate(pair))/2
        g+=pr*estimate
    exact=p*(reward-p@reward)
    assert np.allclose(g,exact)

def test_constant_groups_and_detach():
    r=torch.ones(2,4,2,requires_grad=True)
    for mode in ['loo','grpo']:
        a=group_advantages(r,mode);assert not a.requires_grad and torch.all(a==0)

def test_ppo_prompt_weights_not_selfnormalized():
    logp=torch.zeros(2,2,requires_grad=True);old=logp.detach();adv=torch.ones(2,2,1)
    a=policy_surrogate(logp,old,adv,[1.],prompt_ratio=[2.,2.])
    b=policy_surrogate(logp,old,adv,[1.])
    assert torch.isclose(a,2*b)
    a.backward();assert torch.isfinite(logp.grad).all()

def test_invalid_group_and_kl():
    with pytest.raises(ValueError):group_advantages(torch.ones(2,1,2))
    x=torch.log_softmax(torch.tensor([[1.,2,3]]),-1)
    assert torch.allclose(categorical_kl(x,x),torch.tensor([0.]))

def test_grpo_is_not_raw_reward_gradient():
    r=torch.tensor([[[0.],[2.]]]);r2=r*10
    assert torch.allclose(group_advantages(r,'grpo'),group_advantages(r2,'grpo'))
    assert torch.allclose(group_advantages(r2,'loo'),10*group_advantages(r,'loo'))
