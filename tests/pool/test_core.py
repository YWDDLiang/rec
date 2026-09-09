from dataclasses import replace
import json
import copy
import numpy as np
import pytest
import torch
from rec_pool.contracts import *
from rec_pool.geometry import *
from rec_pool.selection import Config,select,objective,concentration
from rec_pool.refill import refill
from rec_pool.baselines import pdf_rule_reference,nested_root_reservoir
from rec_pool.sft import loss,extract,validate_cache,compatible,manifest,trainable_hash,schedule,train_plan
from rec_pool.data import build_task_rows,normalize_amazon23
from conftest import row,TinyCausal


def test_record_roundtrip(tmp_path):
    r=[row(0),row(1,cost=3)];write_records(r,tmp_path/'rows.jsonl')
    assert read_records(tmp_path/'rows.jsonl')==r
    assert r[1].cost==3

@pytest.mark.parametrize('change',[
 {'record_id':''},{'labels':(-100,-100,-100)},{'input_ids':(1,True,2)},
 {'labels':(-100,-100,3)},{'split':'fake'},{'recipe_id':''},{'labels':(1,-100,2)}])
def test_invalid_records(change):
    with pytest.raises(ValueError):replace(row(),**change)


def test_lineage_leakage():
    with pytest.raises(ValueError):validate_partitions([row(0)],[row(1,split='select',root='root-0')])
    with pytest.raises(ValueError):validate_partitions([row(0)],[row(0,split='select',root='other')])
    validate_partitions([row(0,user='u')],[row(1,split='select',user='u')])


def test_variant_task_not_false_duplicate():
    a=row(0,root='r');b=row(1,root='r',task='description')
    assert a.provenance_group!=b.provenance_group
    assert replace(a,record_id='x').provenance_group==a.provenance_group


def test_nested_roots():
    roots=['a','a','b','c','d','e'];small=nested_root_reservoir(roots,2);large=nested_root_reservoir(roots,4)
    assert set(small)<=set(large)
    assert (0 in small)==(1 in small)
    order=np.array([5,1,3,0,2,4]);rev=nested_root_reservoir(np.array(roots)[order],2)
    assert set(np.array(roots)[small])==set(np.array(roots)[order][rev])


def test_signed_omp():
    d=np.eye(4);x=np.array([[2.,-3,0,0],[0,0,-1,4]])
    c=omp(x,d,2);np.testing.assert_allclose(c@d,x)
    assert (c<0).any() and ((c!=0).sum(1)==2).all()


def test_dictionary_gram_counterexample():
    d=unit([[1,0],[1,1]]);c=np.eye(2);z=coordinates(c,d)
    assert (c@c.T)[0,1]==0
    assert (z@z.T)[0,1]>.7
    np.testing.assert_allclose(z@z.T,(c@d)@(c@d).T,atol=1e-12)


def test_dictionary_sign_permutation_invariant():
    rng=np.random.default_rng(21);d=unit(rng.normal(size=(4,7)));c=rng.normal(size=(20,4));p=np.array([2,0,3,1]);s=np.array([-1,1,-1,1])
    z=coordinates(c,d);other=coordinates(c[:,p]*s,d[p]*s[:,None])
    np.testing.assert_allclose(z@z.T,other@other.T,atol=1e-10)
    a,_=select(z,np.ones(20,dtype=int),['a']*20,list(map(str,range(20))),Config(7,shortlist=0))
    b,_=select(other,np.ones(20,dtype=int),['a']*20,list(map(str,range(20))),Config(7,shortlist=0))
    assert set(a)==set(b)


def test_signed_alignment_not_abs():
    d=np.eye(2);c=np.array([[1,0],[-1,0]]);r=np.array([[1,0]])
    np.testing.assert_allclose(alignment(c,d,r).ravel(),[1,-1])
    assert objective(c,[0])==objective(c,[1]) # information != favorable direction


def test_logdet_diminishing_returns():
    x=np.array([[1.,0],[1.,0],[0,1]])
    first=objective(x,[0])-objective(x,[])
    second=objective(x,[0,1])-objective(x,[0])
    new=objective(x,[0,2])-objective(x,[0])
    assert second<first and new==pytest.approx(first)


def test_scarce_preserve_all():
    z=np.eye(4);ids,report=select(z,np.ones(4,dtype=int),['a']*4,list('abcd'),Config(10))
    assert ids.tolist()==list(range(4)) and report['status']=='keep_all'


def test_quota_protection_and_budget():
    z=np.vstack([np.tile([1.,0],(20,1)),np.tile([0,1.],(3,1))]);cost=np.ones(23,dtype=int)
    ids,r=select(z,cost,['common']*20+['rare']*3,list(map(str,range(23))),Config(8,protected=('rare',),shortlist=0))
    assert set([20,21,22])<=set(ids);assert r['tokens']==8


def test_exact_cell_counts():
    z=np.eye(8);c=['a']*4+['b']*4
    ids,r=select(z,np.ones(8,dtype=int),c,list('abcdefgh'),Config(8,min_rows={'a':2,'b':3},max_rows={'a':2,'b':3}))
    assert r['cell_counts']=={'a':2,'b':3} and len(ids)==5

@pytest.mark.parametrize('cfg',[Config(2,protected=('bad',)),Config(2,min_rows={'a':3}),Config(1,min_rows={'a':1,'b':1})])
def test_infeasible_fail_closed(cfg):
    with pytest.raises(ValueError):select(np.eye(2),np.array([2,2]),['a','b'],['x','y'],cfg)


def test_provenance_cap():
    z=np.eye(4);ids,r=select(z,np.ones(4,dtype=int),['a']*4,['r','r','s','s'],Config(4,provenance_cap=1))
    assert len(ids)==2


def test_no_zero_lower_anchors():
    ids,r=select(np.eye(5),np.ones(5,dtype=int),['a']*5,list('abcde'),Config(1,shortlist=0))
    assert len(ids)==1

@pytest.mark.parametrize('cost',[np.array([0,1]),np.array([1.,2.]),np.array([-1,2])])
def test_invalid_costs(cost):
    with pytest.raises(ValueError):select(np.eye(2),cost,['a']*2,['a','b'],Config(1))


def test_refill_invariants():
    score=np.array([-2.,-1,2,1]);cost=np.array([2,3,2,3]);cells=['a']*4
    ids,r=refill(np.array([0,1]),score,cost,cells,max_swaps=2)
    assert ids.tolist()==[2,3] and r['before_tokens']==r['after_tokens']==5
    assert sum(x['score_gain'] for x in r['swaps'])==6


def test_refill_protects_cell_and_no_cross_cost():
    ids,r=refill(np.array([0,1]),np.array([-2,-1,2,1]),np.array([2,3,3,4]),['a','b','a','b'],max_swaps=3,protected_cells=['b'])
    assert ids.tolist()==[0,1] and r['actual_swaps']==0


def test_random_refill_equal_token_count():
    ids,r=refill(np.arange(5),np.arange(10),np.ones(10,dtype=int),['a']*10,max_swaps=3,random_control=True)
    assert len(ids)==5 and r['actual_swaps']==3 and r['before_tokens']==r['after_tokens']


def test_pdf_small_atoms_not_upsampled():
    codes=np.eye(6);ids=pdf_rule_reference(codes,np.ones(6),['x']*6,cap=3,min_after_truncation=2)
    assert ids.tolist()==list(range(6))


def test_pdf_protected_can_exceed_cap():
    codes=np.tile([1.,0],(20,1));ids=pdf_rule_reference(codes,np.arange(20),['x']*20,user_mask=np.ones(20,bool),cap=4,min_after_truncation=2)
    assert len(ids)==20


def test_pdf_quality_filter_explicit():
    ids=pdf_rule_reference(np.eye(3),np.ones(3),['x']*3,quality_mask=[True,False,True],cap=3,min_after_truncation=1)
    assert ids.tolist()==[0,2]


def test_concentration():
    assert concentration([2,2])['ess']==2
    assert concentration([4,0])['ess']==1


def test_causal_shift_and_response_mask():
    torch.manual_seed(4);m=TinyCausal();r=row(0,cost=2)
    value=loss(m,[r]);pred=m(torch.tensor([r.input_ids])).logits[0,1:3]
    target=torch.tensor(r.labels[2:]);expected=torch.nn.functional.cross_entropy(pred,target)
    torch.testing.assert_close(value,expected)


def test_probe_roundtrip_and_nonmutation(tmp_path):
    torch.manual_seed(4);m=TinyCausal();m.train();before=trainable_hash(m);r=[row(i) for i in range(8)]
    for p in m.parameters():p.grad=torch.ones_like(p)
    oldgrads=[p.grad.clone() for p in m.parameters()]
    meta=extract(m,r,tmp_path/'cache',dimension=8,base_revision='tiny-fixture')
    assert before==trainable_hash(m) and m.training
    for p,g in zip(m.parameters(),oldgrads):torch.testing.assert_close(p.grad,g)
    _,raw,un,norms,l=validate_cache(r,tmp_path/'cache')
    np.testing.assert_allclose(np.linalg.norm(un,axis=1),1,atol=1e-6)
    assert np.all(norms>0) and np.all(l>0)
    assert meta['manifest']==manifest(m)
    with pytest.raises(ValueError):validate_cache(r[::-1],tmp_path/'cache')


def test_test_gradient_rejected(tmp_path):
    with pytest.raises(ValueError):extract(TinyCausal(),[row(0,split='test')],tmp_path/'cache',base_revision='fixture')


def test_cache_coordinate_check():
    a={'recipe_id':'a','base_revision':'b','trainable_state_hash':'c','manifest':[],'projection_hash':'d'}
    compatible(a,a)
    with pytest.raises(ValueError):compatible(a,dict(a,projection_hash='e'))


def test_schedule_tracks_repeats_and_tokens():
    rows=[row(0,cost=2),row(1,cost=3)];plan,r=schedule(rows,np.array([0,1]),13)
    assert r['tokens']<=13 and r['unused_tokens']<2
    assert len({p['occurrence_id'] for p in plan})==len(plan)
    assert r['duplicate_occurrence_fraction']>0


def test_grad_accumulation_same_update():
    torch.manual_seed(10);m=TinyCausal();m2=copy.deepcopy(m);r=[row(0,cost=1),row(1,cost=3),row(2,cost=2)]
    plan,_=schedule(r,np.arange(3),6)
    a=torch.optim.SGD(m.parameters(),lr=.1);b=torch.optim.SGD(m2.parameters(),lr=.1)
    train_plan(m,r,plan,a,batch_size=3,microbatch_size=1,clip_norm=100)
    train_plan(m2,r,plan,b,batch_size=3,microbatch_size=3,clip_norm=100)
    for p,q in zip(m.parameters(),m2.parameters()):torch.testing.assert_close(p,q,atol=1e-7,rtol=1e-5)


def test_selected_examples_actually_change_training():
    torch.manual_seed(11);a=TinyCausal();b=copy.deepcopy(a);rows=[row(0,answer=2),row(1,answer=9)]
    pa,_=schedule(rows,np.array([0]),12);pb,_=schedule(rows,np.array([1]),12)
    ra=train_plan(a,rows,pa,torch.optim.AdamW(a.parameters(),lr=.02),batch_size=3)
    rb=train_plan(b,rows,pb,torch.optim.AdamW(b.parameters(),lr=.02),batch_size=3)
    assert set(ra['consumed_ids'])=={'0'} and set(rb['consumed_ids'])=={'1'}
    assert trainable_hash(a)!=trainable_hash(b) and ra['tokens']==rb['tokens']==12


def test_bad_plan_detected():
    m=TinyCausal();r=[row(0)];p,_=schedule(r,np.array([0]),1);p[0]['content_id']='wrong'
    with pytest.raises(ValueError):train_plan(m,r,p,torch.optim.SGD(m.parameters(),lr=.1))


def test_forged_plan_budget_rejected():
    m=TinyCausal();r=[row(0)];p,_=schedule(r,np.array([0]),1);p[0]['response_tokens']=1000
    with pytest.raises(ValueError):train_plan(m,r,p,torch.optim.SGD(m.parameters(),lr=.1))


def test_temporal_builder_never_uses_equal_future_time():
    events=[dict(event_id=str(i),user_id='u',item_id=str(i),domain='a',timestamp=t) for i,t in enumerate([1,2,2,4,8])]
    catalog={('a',str(i)):{'title':f'item {i}','category':'c'} for i in range(5)}
    rows=build_task_rows(events,catalog,train_end=3,select_end=6)
    assert rows
    for r in rows:
        if 'history_timestamps' in r:assert max(r['history_timestamps'])<r['target_timestamp']
        if r['task']=='id_to_title':assert r['split']=='train'
    assert not any(r['task']=='id_to_title' and r['output_text']=='item 4' for r in rows)


def test_amazon_requires_actual_timestamp(tmp_path):
    f=tmp_path/'a.jsonl';f.write_text(json.dumps({'user_id':'u','parent_asin':'i'})+'\n')
    with pytest.raises(ValueError):list(normalize_amazon23(f,'a'))


def test_cache_tampering_rejected(tmp_path):
    m=TinyCausal();rows=[row(0)];p=tmp_path/'c';extract(m,rows,p,dimension=4,base_revision='fixture')
    x=np.load(p/'raw.npy');x[0,0]+=1;np.save(p/'raw.npy',x)
    with pytest.raises(ValueError):validate_cache(rows,p)


def test_invalid_prediction_not_removed():
    from rec_pool.metrics import evaluate_rankings
    out=evaluate_rankings([{'query_id':'q','domain':'d','targets':['b'],'predictions':['illegal','b']}],{'d':['a','b']},2)
    assert out['micro_ndcg']==pytest.approx(1/np.log2(3))
    assert out['micro_recall']==1


def test_duplicate_predictions_not_double_counted():
    from rec_pool.metrics import evaluate_rankings
    out=evaluate_rankings([{'query_id':'q','domain':'d','targets':['a','b'],'predictions':['a','a']}],{'d':['a','b']},2)
    assert out['micro_recall']==.5 and out['per_query'][0]['invalid_or_duplicate_slots']==1


def test_empty_prediction_scores_zero():
    from rec_pool.metrics import evaluate_rankings
    out=evaluate_rankings([{'query_id':'q','domain':'d','targets':['a'],'predictions':[]}],{'d':['a']},2)
    assert out['micro_ndcg']==0


def test_refill_cap_valid_after_swap():
    ids,rep=refill(np.array([0,1]),np.array([0,0,2,3]),np.ones(4,dtype=int),['a']*4,
                  groups=['x','y','z','z'],provenance_cap=1,max_swaps=2)
    assert not ({2,3}<=set(ids))


def test_dictionary_fit_real_code():
    rng=np.random.default_rng(32);x=rng.normal(size=(24,8));d,c,_=fit_dictionary(x,4,2,24,32,10)
    assert d.shape==(4,8) and c.shape==(24,4) and ((np.abs(c)>1e-12).sum(1)<=2).all()


def test_exact_quota_does_not_preselect_cheapest_rows():
    # With exactly 1 row from each cell, the ranking must still affect choices.
    z=np.eye(4);costs=np.ones(4,dtype=int);cells=['a','a','b','b']
    cfg=Config(2,min_rows={'a':1,'b':1},max_rows={'a':1,'b':1},shortlist=0)
    ids,r=select(z,costs,cells,list('abcd'),cfg,utility=np.array([0,5,0,6]),mode='utility')
    assert set(ids)=={1,3}


def test_residual_budget_quota_remains_feasible():
    # The expensive highest-score row would make the remaining cell impossible.
    cfg=Config(4,min_rows={'a':1,'b':1},max_rows={'a':1,'b':1},shortlist=0)
    ids,r=select(np.eye(3),np.array([4,2,2]),['a','a','b'],list('abc'),cfg,utility=[100,3,3],mode='utility')
    assert set(ids)=={1,2}


def test_optional_negative_marginal_not_forced():
    cfg=Config(2,keep_all_when_fits=False,coverage_weight=0,utility_weight=1,shortlist=0)
    ids,r=select(np.eye(3),np.ones(3,dtype=int),['a']*3,list('abc'),cfg,[-1,-2,1])
    assert ids.tolist()==[2] and r['unused_tokens']==1


@pytest.mark.parametrize('command, aliases', [
    ('train', ('--batch', '--batch-size')),
    ('align', ('--ref-cache', '--reference-cache')),
])
def test_documented_cli_aliases(command, aliases, capsys):
    from rec_pool.__main__ import main
    with pytest.raises(SystemExit) as caught:
        main([command, '--help'])
    assert caught.value.code == 0
    output = capsys.readouterr().out
    assert all(alias in output for alias in aliases)
