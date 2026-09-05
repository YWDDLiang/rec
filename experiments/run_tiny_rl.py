"""Fixed-estimand complete-group RL, exact categorical environment; no real users."""
import argparse,json,itertools
from pathlib import Path
import numpy as np,torch
from rec_lab.rl import group_advantages,policy_surrogate
from rec_lab.allocation import minimax_allocation

def run(output,seeds=(0,1,2),steps=80):
    torch.set_num_threads(1);all_runs=[]
    # Four prompts, three actions, two reward components in a known toy oracle.
    reward=torch.tensor([[[1.,.4],[.2,1.],[0.,0.]],[[.8,.8],[.1,.3],[0.,0.]],
                         [[.1,.9],[1.,.1],[.4,.5]],[[.9,.9],[.2,.2],[.1,.4]]])
    p=np.array([.4,.3,.2,.1]);n=4;G=4;B=12
    for seed in seeds:
        for method in ['base_prompt','variance_prompt']:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);logits=torch.nn.Parameter(torch.zeros(n,3))
            optim=torch.optim.SGD([logits],lr=.2);trace=[]
            for step in range(steps):
                probs=logits.detach().softmax(-1)
                # Analytic SINGLE-rollout score second moment gives conservative
                # allocation proxy; it is NOT the exact group-level covariance.
                moments=np.zeros((n,2))
                for i in range(n):
                    pr=probs[i].numpy();mu=(pr[:,None]*reward[i].numpy()).sum(0)
                    for a in range(3):
                        score=np.eye(3)[a]-pr
                        moments[i]+=pr[a]*np.dot(score,score)*(reward[i,a].numpy()-mu)**2
                q=p.copy() if method=='base_prompt' else minimax_allocation(p,moments+1e-8,floor=.1)["q"]
                ids=rng.choice(n,B,p=q);ids_t=torch.tensor(ids)
                acts=torch.multinomial(probs[ids_t],G,replacement=True)
                rewards=reward[ids_t[:,None],acts]
                old=probs[ids_t].log().gather(1,acts)
                logp=logits.log_softmax(-1)[ids_t].gather(1,acts)
                advantage=group_advantages(rewards,'loo')
                loss=policy_surrogate(logp,old,advantage,[.5,.5],p[ids]/q[ids])
                optim.zero_grad();loss.backward();optim.step()
                trace.append({'surrogate_value':float(loss.detach()),'gradient_norm':float(logits.grad.norm())})
            policy=logits.detach().softmax(-1)
            value=(torch.tensor(p)[:,None,None]*policy[:,:,None]*reward).sum((0,1)).tolist()
            all_runs.append({'seed':seed,'method':method,'final_expected_reward':value,'loss_trace':trace,
                             'rollouts':steps*B*G,'group_size':G,'importance_correction':True,'baseline_distribution':p.tolist(),
                             'allocation_moments':'analytic single-rollout proxy, not exact full-group moments'})
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    data={'evidence':'synthetic complete-feedback categorical RL; no LLM, no real-user causal evidence','runs':all_runs}
    (out/'tiny_rl.json').write_text(json.dumps(data,indent=2));return data
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results');p.add_argument('--steps',type=int,default=80);a=p.parse_args();run(a.output,steps=a.steps)
