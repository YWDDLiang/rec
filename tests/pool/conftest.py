from types import SimpleNamespace
import torch
from rec_pool.contracts import Record


def row(i=0, *, split='train', task='next', domain='a', root=None, answer=2, cost=1, user=None):
    ids=(1,3+(i%7))+tuple([answer]*cost)
    return Record(str(i),root or f'root-{i}',task,domain,'fixture',split,ids,(-100,-100)+tuple([answer]*cost),'recipe-fixture',user)


class TinyCausal(torch.nn.Module):
    def __init__(self, vocab=16, hidden=8):
        super().__init__();self.emb=torch.nn.Embedding(vocab,hidden);self.proj=torch.nn.Linear(hidden,vocab)
    def forward(self,input_ids,attention_mask=None):
        # Strictly causal positionwise cumulative context, no look-ahead.
        h=self.emb(input_ids)
        if attention_mask is not None:h=h*attention_mask.unsqueeze(-1)
        h=h.cumsum(1)/torch.arange(1,h.shape[1]+1,device=h.device)[None,:,None]
        return SimpleNamespace(logits=self.proj(h))
