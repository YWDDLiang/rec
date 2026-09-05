"""Explicit simulators for mechanism tests, never a claim about actual users."""
import numpy as np
from .data import Record

def synthetic_feedback(seed=0,n_train=192,n_selection=96,n_test=96,items=16):
    rng=np.random.default_rng(seed);records=[]
    for split,n,offset in [('train',n_train,0),('selection',n_selection,10000),('test',n_test,20000)]:
        for j in range(n):
            scene=int(rng.choice(2,p=[.8,.2] if split=='train' else [.5,.5]));pref=int(rng.integers(0,4))
            history=[int((pref+4*rng.integers(0,items//4))%items) for _ in range(4)]
            item=int(rng.integers(items));match=float(item%4==pref)
            # Distinct, partially conflicting synthetic click and appreciation.
            x1=-.6+1.5*match+.7*(item<items//2)+.3*scene
            x2=-.7+1.8*match-.6*(item<items//2)+.4*(scene==1)
            labels=[float(rng.random()<1/(1+np.exp(-v))) for v in (x1,x2)]
            t=float(offset+j+100)
            records.append(Record(f'{split}-{j}',f'{split}-user-{j}',[str(k) for k in history],
                [t-k for k in (5,4,3,2)],t,str(item),str(scene),labels,split,source='synthetic_mechanism_only'))
    return records
