"""Strict rank-list cross-check. Official benchmark evaluators remain authoritative.

Invalid/duplicate recommendations occupy their ORIGINAL positions; they are not
removed before scoring. Candidate-sampled and full-catalog results must be tagged.
"""
from collections import defaultdict
import math
import numpy as np


def evaluate_rankings(queries,catalogs,k=10,scope='declared-candidate-set'):
    if type(k) is not int or k<1:raise ValueError('invalid k')
    if scope not in {'full-catalog','declared-candidate-set'}:raise ValueError('declare ranking universe')
    seen=set();per=[];by=defaultdict(list);total_invalid=0;total_slots=0
    for q in queries:
        qid=q['query_id'];domain=q['domain'];task=q.get('task','recommendation')
        if not isinstance(qid,str) or not qid or qid in seen:raise ValueError('invalid/duplicate query_id')
        seen.add(qid)
        if domain not in catalogs:raise ValueError('unknown catalog domain')
        catalog=set(catalogs[domain]);truth=set(q['targets']);pred=q['predictions']
        if not truth or not truth<=catalog:raise ValueError('target absent from declared catalog')
        if not isinstance(pred,list):raise ValueError('prediction must be a list')
        emitted=set();hit=0;dcg=0.;invalid=0
        for position,item in enumerate(pred[:k],1):
            if item not in catalog or item in emitted:invalid+=1;continue
            emitted.add(item)
            if item in truth:hit+=1;dcg+=1/math.log2(position+1)
        ideal=sum(1/math.log2(i+1) for i in range(1,min(k,len(truth))+1))
        r={'query_id':qid,'domain':domain,'task':task,'recall':hit/len(truth),
           'ndcg':dcg/ideal,'invalid_or_duplicate_slots':invalid,'returned_slots':min(len(pred),k)}
        per.append(r);by[(task,domain)].append(r);total_invalid+=invalid;total_slots+=min(len(pred),k)
    if not per:raise ValueError('no queries')
    groups=[{'task':t,'domain':d,'queries':len(rs),'recall':float(np.mean([r['recall'] for r in rs])),
             'ndcg':float(np.mean([r['ndcg'] for r in rs]))} for (t,d),rs in sorted(by.items())]
    return {'k':k,'scope':scope,'queries':len(per),'micro_recall':float(np.mean([r['recall'] for r in per])),
            'micro_ndcg':float(np.mean([r['ndcg'] for r in per])),
            'macro_recall':float(np.mean([g['recall'] for g in groups])),
            'macro_ndcg':float(np.mean([g['ndcg'] for g in groups])),
            'invalid_or_duplicate_rate_among_returned_slots':total_invalid/max(total_slots,1),
            'groups':groups,'per_query':per,
            'boundary':'No online satisfaction, CTR, or propensity-corrected causal claim. Empty lists score zero.'}
