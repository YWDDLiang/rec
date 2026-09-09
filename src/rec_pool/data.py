"""Streaming Amazon23 normalization and explicit event-time SFT task construction.

The builder uses only past events. It does NOT reproduce LC-Rec's full generated
auxiliary-task data, RQ-VAE or any benchmark's original split. Native protocols
must be exported with root IDs and kept separate from this diagnostic builder.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import gzip
import json
from .contracts import digest


def normalize_amazon23(path,domain,timestamp_field='timestamp'):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8') as f:
        for lineno,line in enumerate(f,1):
            if not line.strip():continue
            d=json.loads(line)
            for name in ['user_id','parent_asin',timestamp_field]:
                if name not in d:raise ValueError(f'{path}:{lineno}: missing {name}; no guessed timestamp')
            ts=d[timestamp_field]
            if type(ts) is not int or ts<0:raise ValueError('invalid timestamp')
            yield {'event_id':digest([domain,d['user_id'],d['parent_asin'],ts]),'user_id':str(d['user_id']),
                   'item_id':str(d['parent_asin']),'timestamp':ts,'domain':domain,'rating':d.get('rating')}


def build_task_rows(events,catalog,*,train_end,select_end,max_history=20,metadata_snapshot='static-transductive'):
    """Three evidence-grounded tasks: next item, history-category choice, ID->title.

    Item metadata is admitted only for items in training histories/targets. Repeated
    ID->title tasks share a stable item root and are emitted once, in train only.
    For native SID tasks, caller must supply a pinned official item 'answer'.
    Derived history category is an observable summary, NOT true/latent preference.
    """
    if not train_end<select_end or max_history<1:raise ValueError('invalid temporal boundaries')
    grouped=defaultdict(list);seen=set()
    for e in events:
        for key in ['event_id','user_id','item_id','timestamp','domain']:
            if key not in e:raise ValueError(f'missing {key}')
        if type(e['timestamp']) is not int:raise ValueError('invalid timestamp')
        if e['event_id'] in seen:raise ValueError('duplicate event identity')
        seen.add(e['event_id']);grouped[(e['domain'],e['user_id'])].append(e)
    out=[];training_items=set()
    for (domain,user),evs in sorted(grouped.items()):
        evs.sort(key=lambda e:(e['timestamp'],e['event_id']))
        for i,e in enumerate(evs):
            history=[h for h in evs[:i] if h['timestamp']<e['timestamp']][-max_history:]
            if not history:continue
            split='train' if e['timestamp']<train_end else ('select' if e['timestamp']<select_end else 'test')
            item=catalog.get((domain,e['item_id']),{})
            answer=item.get('answer',e['item_id'])
            past=[catalog.get((domain,h['item_id']),{}).get('answer',h['item_id']) for h in history]
            base={'root_id':e['event_id'],'domain':domain,'source':'temporal_events','split':split,
                  'user_id':user,'target_timestamp':e['timestamp'],'history_timestamps':[h['timestamp'] for h in history],
                  'metadata_scope':metadata_snapshot}
            out.append(dict(base,record_id=e['event_id']+':next',task='next_item',
                            input_text='History: '+json.dumps(past)+'\nRecommend the next item.',output_text=answer,label_origin='observed'))
            if split=='train':
                training_items.add((domain,e['item_id']))
                training_items|={(domain,h['item_id']) for h in history}
                cats=[catalog.get((domain,h['item_id']),{}).get('category') for h in history]
                cats=[c for c in cats if isinstance(c,str) and c]
                if cats:
                    from collections import Counter
                    category=sorted(Counter(cats),key=lambda c:(-cats.count(c),c))[0]
                    out.append(dict(base,record_id=e['event_id']+':history_category',task='history_category_summary',
                        input_text='History: '+json.dumps(past)+'\nReturn its most frequent recorded category.',
                        output_text=category,label_origin='history-derived-not-user-intent'))
    for domain,item_id in sorted(training_items):
        item=catalog.get((domain,item_id),{})
        if not item.get('title'):continue
        root=digest(['catalog',domain,item_id])
        out.append({'record_id':root+':id_title','root_id':root,'domain':domain,'source':'item_metadata',
                    'split':'train','user_id':None,'task':'id_to_title','input_text':'Describe item '+item.get('answer',item_id),
                    'output_text':item['title'],'label_origin':'observed-metadata','metadata_scope':metadata_snapshot})
    return out
