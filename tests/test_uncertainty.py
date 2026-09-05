import numpy as np,pytest
from rec_lab.uncertainty import *
from rec_lab.frontier import solve_frontier

def test_robust_box_exact_and_conservative():
    A=np.array([[1.,.2],[.1,1.]]);U=np.full_like(A,.3)
    r=robust_frontier(A,U);plain=solve_frontier(A)
    assert r.margin<=plain.margin
    assert np.allclose(r.target_gains,(A-U)@r.weights)

def test_intervals_contract_with_sample_count_and_preserve_zero_direction():
    raw=np.tile([1.,0.],(10,1));G=np.array([[1.,0.],[0,0]])
    A,U=bounded_response_estimate([raw],G,1)
    A2,U2=bounded_response_estimate([np.tile(raw,(4,1))],G,1)
    assert np.allclose(A,A2) and np.allclose(U/2,U2)
    assert U[0,1]==0
    with pytest.raises(ValueError):robust_frontier(A,-U)

def test_tiny_reference_cannot_fake_confidence():
    A,U=bounded_response_estimate([np.ones((2,1))],[[1.]],1.)
    assert A[0,0]>0 and robust_frontier(A,U).margin<0
