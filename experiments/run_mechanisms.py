"""Ten idea-specific analytic/numeric demos plus counterexamples. CPU only."""
from pathlib import Path
import argparse,json,itertools,platform,time
import numpy as np
from rec_lab.frontier import *
from rec_lab.atoms import *
from rec_lab.trajectory import *
from rec_lab.observational import *
from rec_lab.allocation import *
from rec_lab.acquisition import *
from rec_lab.exploration import *
from rec_lab.decoding import *
from rec_lab.science import *
from rec_lab.uncertainty import *

def run(out):
    rng=np.random.default_rng(42);cases={}
    V=np.eye(2);G=np.array([[1,-2],[-2,1.]])
    A=response_matrix(V,G);old=solve_frontier(A);new=solve_frontier(np.c_[A,[1.,1.]])
    cases['01']={'unrestricted_margin':parameter_feasibility(V)['margin'],'existing_pool_margin':old.margin,
                 'with_verified_bridge_margin':new.margin,'primal_dual_gap':new.gap,
                 'interpretation':'Constructed support gap, not an observed recommendation improvement.'}
    Z=np.c_[np.ones(20),rng.normal(size=(20,2))];X=rng.normal(size=(20,7));C=rng.normal(size=(3,7))
    R,_=nuisance_residual(X,Z);R2,_=nuisance_residual(X+Z@C,Z)
    cases['02']={'invariance_max_error':float(abs(R-R2).max()),'weighted_orthogonality_error':float(abs(Z.T@R).max()),
                 'not_proven':'Residuals equal true personalized interests.'}
    theta=np.array([.4,-.3]);Ha=np.diag([1.,2.]);Hb=np.array([[2.,.4],[.4,1.]]);ba=np.array([1.,0.]);bb=np.array([0.,1.]);eta=.1
    ab=quadratic_step(quadratic_step(theta,Ha,ba,eta),Hb,bb,eta);ba_=quadratic_step(quadratic_step(theta,Hb,bb,eta),Ha,ba,eta)
    error=np.linalg.norm(ab-ba_-eta**2*order_bracket(theta,Ha,ba,Hb,bb))
    cases['03']={'order_identity_error':float(error),'order_effect_norm':float(np.linalg.norm(ab-ba_)),
                 'commuting_null':float(np.linalg.norm(order_bracket(theta,Ha,ba,Ha,ba)))}
    dr=missing_at_random_mean([.8,np.nan,np.nan,np.nan],[1,0,0,0],[.25]*4,[.1]*4)
    cases['04']={'true_mean':.8,'complete_case_designed_dr_estimate':dr['estimate'],
                 'unknown_propensity_handling':'raises ValueError, never replaced by a uniform assumption'}
    p=np.array([.7,.2,.1]);mom=np.array([[1.,2.],[9.,1.],[4.,12.]])
    alloc=minimax_allocation(p,mom)
    cases['05']={'base_max_second_moment':float(max(variance_objectives(p,p,mom))),
                 'allocated_max_second_moment':float(max(alloc['objective_second_moments'])),'q':alloc['q'].tolist(),
                 'scope':'Fixed synthetic vector score distribution, not GRPO true-reward guarantee.'}
    order,score=rank_acquisition([.5,.5],[[1.,-.2],[1.,-.1]],[[.1,.1],[.1,.1]],[2.,1.],old.margin)
    pairs=[PreferencePair('a','x','y','expert',.95,'panel-1'),PreferencePair('b','x','z','synthetic',.99,'model-only')]
    cases['06']={'rank':order.tolist(),'scores':score.tolist(),'accepted_verified_pairs':len(accept_pairs(pairs)),
                 'warning':'Witness violation is necessary, not sufficient. No actual expert labeling performed.'}
    r=pagerank([[0,1,0],[0,0,1],[1,0,0]],[.8,.1,.1]);Q=np.array([[.4,.3],[.1,.5]])
    q,eps=safe_mixture([1,0],[0,1],[.9,.3],.6)
    cases['07']={'pagerank_mass':float(r.sum()),'absorbing_value':absorbing_value(Q,[1.,1.]).tolist(),
                 'one_step_epsilon':eps,'one_step_lower_value':float(q@np.array([.9,.3])),
                 'real_retention_validated':False}
    trie=CatalogTrie([[1],[1,2],[3]]);p=np.array([.2,.3,.5]);d=np.array([.6,.3,.1]);output,acc=speculative_identity(p,d)
    cases['08']={'trie_probability_max_error':float(max(abs(trie.item_path_probability(i,p)-p[i]) for i in range(3))),
                 'speculative_identity_max_error':float(abs(output-p).max()),'acceptance':acc,
                 'wallclock_speedup_measured':False}
    P=np.array([[.9,.1,.0],[.8,.1,.0],[.0,.8,.7],[.1,.1,.9]]);S,val=greedy_coverage(P,[1,1,1],2)
    opt=max(coverage_value(P,[1,1,1],s) for s in itertools.combinations(range(4),2))
    cases['09']={'selected':S,'coverage':val,'optimal_coverage_small_enumeration':opt,'ratio':val/opt,
                 'expert_fit_calibrated':False}
    features=np.array([[1.,0.],[.99,.01],[0,1.]])
    S,info=greedy_information(features,np.eye(2),.2,2)
    cases['10']={'selected_experiments':S,'gaussian_information':info,
                 'redundant_pair_information':gaussian_information(features,np.eye(2),.2,[0,1]),
                 'chemistry_or_wet_lab_validated':False}
    # Intentionally include a non-passing confidence gate.
    A,U=bounded_response_estimate([np.ones((2,1))],[[1.]],1.)
    cases['confidence_gate']={'empirical_response':float(A[0,0]),'lower_response':float((A-U)[0,0]),
                              'decision':'INSUFFICIENT_EVIDENCE, not safe improvement'}
    payload={'evidence_level':'CPU numerical mechanism checks, synthetic constructs only','cases':cases}
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'mechanisms.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2));return payload
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results');a=p.parse_args();run(a.output)
