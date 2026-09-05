"""Constructive quadratic intervention; no real recommendation claim."""
import json
from pathlib import Path
import numpy as np
from rec_lab.frontier import solve_frontier,dual_face_gain

def main():
    A=np.zeros((2,1));B=np.array([[2.,-1.],[-1.,2.]])
    theta=np.zeros(2);eta=.1
    cases={"existing":A,"plus_left":np.c_[A,B[:,0]],"plus_right":np.c_[A,B[:,1]],"plus_pair":np.c_[A,B]}
    out={"scope":"constructed exact quadratic target losses, not a benchmark","eta":eta,"cases":{}}
    for name,M in cases.items():
        result=solve_frontier(M)
        # Target gradients are -I at zero; G=-M.T realized by quadratic train losses.
        direction=-M@result.weights
        after=theta-eta*direction
        before_loss=.5*(theta-1)**2;after_loss=.5*(after-1)**2
        out["cases"][name]={"margin":result.margin,"weights":result.weights.tolist(),
                           "target_loss_improvements":(before_loss-after_loss).tolist()}
    out['batch_face_gain']=dual_face_gain(A,B)['face_gain']
    Path('results').mkdir(exist_ok=True)
    Path('results/complementarity.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
