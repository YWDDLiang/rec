import dataclasses,json,numpy as np,torch,pytest
from rec_lab.data import *
from rec_lab.synthetic import synthetic_feedback
from rec_lab.training import *
from rec_lab.language import *

def test_record_roundtrip_and_splits(tmp_path):
    rows=synthetic_feedback(1,12,8,8);path=tmp_path/'d.jsonl';write_records(path,rows)
    got=read_records(path);assert len(got)==len(rows)
    assert got[0].history==rows[0].history
    with pytest.raises(ValueError):validate_splits(rows+[rows[0]])

def test_future_history_and_split_overlap():
    r=synthetic_feedback(0,3,2,2)[0]
    with pytest.raises(ValueError):dataclasses.replace(r,history_times=[r.timestamp]*4).validate()
    rows=synthetic_feedback(0,3,2,2);rows[-1].timestamp=0
    with pytest.raises(ValueError):validate_splits(rows)

def test_prompt_never_contains_labels():
    r=synthetic_feedback(0,3,2,2)[0];r.prompt='ANSWER LABELS: 1,0'
    assert 'ANSWER LABELS' not in feedback_prompt(r)
    p=feedback_prompt(r);r.labels=[1-v for v in r.labels]
    assert p==feedback_prompt(r)

def test_masked_sft_shift():
    b=supervised_tokens([[1,2],[1]],[[3],[4,5]],0)
    assert b['labels'].tolist()==[[-100,-100,3],[-100,4,5]]
    logits=torch.zeros(2,3,6);l=shifted_sequence_nll(logits,b['labels'])
    assert torch.allclose(l,torch.full((2,),np.log(6)))
    # First response is predicted at LAST prompt position, not itself.
    logits[0,1,3]=12
    assert shifted_sequence_nll(logits,b['labels'])[0]<.001

@pytest.mark.parametrize('method',['uniform','scalar','frontier','atoms_frontier'])
def test_end_to_end_supervised(method):
    torch.set_num_threads(1);torch.manual_seed(0);data=synthetic_feedback(0,20,12,12)
    model=TinyFeedbackModel([str(i) for i in range(16)],['0','1'])
    result=train_feedback(model,data,method,steps=2,pool_size=6,reference_size=12,seed=0)
    assert len(result['trace'])==2 and len(result['test']['bce'])==2
    assert np.isfinite(result['test']['bce']).all()

def test_gradient_parameter_guard():
    model=TinyFeedbackModel(['0','1'],['0'])
    with pytest.raises(ValueError):select_parameters(model,max_elements=2)
    with pytest.raises(ValueError):select_parameters(model,name_filter='not_present')

def test_local_kuairand_adapter(tmp_path):
    p=tmp_path/'input.csv'
    p.write_text('user_id,video_id,time_ms,tab,is_click,is_like\nu,i1,1,0,1,0\nu,i2,3,0,0,0\nu,i3,6,1,1,1\n')
    result=prepare_kuairand([p],tmp_path/'out.jsonl',2,5)
    data=read_records(tmp_path/'out.jsonl')
    assert len(data)==3 and data[-1].history==['i1','i2'] and data[-1].propensity is None
    with pytest.raises(ValueError):prepare_kuairand([p],tmp_path/'x',2,5,max_rows=2)


def test_direct_split_validator_checks_history_invariant():
    from rec_lab.synthetic import synthetic_feedback
    from rec_lab.data import validate_splits
    records=synthetic_feedback(0)
    records[0].history=["future"]
    records[0].history_times=[records[0].timestamp+1]
    with pytest.raises(ValueError):validate_splits(records)


def test_large_timestamp_tie_keeps_strictly_earlier_history(tmp_path):
    import csv
    from rec_lab.data import prepare_kuairand,read_records
    source=tmp_path/'events.csv'
    rows=[['u','old',1,'0',1,0]]+[['u',f'tied{i}',2,'0',1,0] for i in range(12)]+[['u','sel',3,'0',1,0],['u','test',4,'0',1,0]]
    with source.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['user_id','video_id','time_ms','tab','is_click','is_like']);w.writerows(rows)
    out=tmp_path/'out.jsonl'
    prepare_kuairand([source],out,2.5,3.5,max_history=2)
    records=read_records(out)
    tied=[r for r in records if r.timestamp==2]
    assert len(tied)==12
    assert all(r.history==['old'] for r in tied)
