"""Optional local Hugging Face / PEFT feedback ranker; integration NOT GPU-tested.

No remote executable model code. Language backbone is a carrier, not the novelty.
Only canonical labels supervise feedback_head. Raw Record.prompt is ignored.
"""
from __future__ import annotations
import torch
from torch import nn
from .data import feedback_prompt

class HFFeedbackRanker(nn.Module):
    def __init__(self,model_path,objectives=2,max_length=512,lora_rank=8,device='cpu',local_files_only=True):
        super().__init__()
        try:
            from transformers import AutoTokenizer,AutoModelForCausalLM
        except ImportError as e: raise ImportError('Install optional dependencies: pip install -e ".[hf]"') from e
        self.run_config={"base_model_path":str(model_path),"objectives":objectives,
                         "max_length":max_length,"lora_rank":lora_rank,
                         "integration_status":"not executed with pretrained weights in delivered validation"}
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=local_files_only,trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None: raise ValueError('tokenizer needs pad or eos token')
            self.tokenizer.pad_token=self.tokenizer.eos_token
        self.tokenizer.padding_side='right';self.max_length=max_length
        model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=local_files_only,trust_remote_code=False,
                                                  torch_dtype=torch.float32 if device=='cpu' else torch.bfloat16)
        if lora_rank>0:
            from peft import get_peft_model,LoraConfig
            # Caller must use a standard architecture supporting all-linear LoRA.
            model=get_peft_model(model,LoraConfig(r=lora_rank,lora_alpha=2*lora_rank,target_modules='all-linear',
                                                 lora_dropout=0.0,bias='none',task_type='CAUSAL_LM'))
        else:
            for p in model.parameters(): p.requires_grad_(False)
        self.backbone=model
        hidden=getattr(model.config,'hidden_size',None)
        if hidden is None: raise ValueError('model config has no hidden_size')
        self.feedback_head=nn.Linear(hidden,objectives).to(dtype=next(model.parameters()).dtype)
        self.to(device)
    def forward(self,records):
        text=[feedback_prompt(r) for r in records]
        batch=self.tokenizer(text,return_tensors='pt',padding=True,truncation=True,max_length=self.max_length)
        device=next(self.parameters()).device;batch={k:v.to(device) for k,v in batch.items()}
        # Explicit final non-padding index works for both left/right padding.
        mask=batch['attention_mask'];positions=torch.arange(mask.shape[1],device=device)[None,:].expand_as(mask)
        idx=positions.masked_fill(~mask.bool(),-1).max(1).values
        if (idx<0).any(): raise ValueError('empty tokenized prompt')
        out=self.backbone(**batch,output_hidden_states=True,use_cache=False)
        hidden=out.hidden_states[-1][torch.arange(len(records),device=device),idx]
        return self.feedback_head(hidden).float()
    def save(self,path):
        from pathlib import Path
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        self.backbone.save_pretrained(path/'backbone');self.tokenizer.save_pretrained(path/'backbone')
        torch.save(self.feedback_head.state_dict(),path/'feedback_head.pt')
        import json
        (path/'ranker_config.json').write_text(json.dumps(self.run_config,indent=2))

    @classmethod
    def load_saved(cls,path,device='cpu'):
        """Reconstruct a saved adapter against the recorded LOCAL base path.

        This entrypoint is supplied but has not been integration-tested with
        transformers/PEFT or real model weights in the CPU-only delivery.
        """
        from pathlib import Path
        import json
        path=Path(path);cfg=json.loads((path/'ranker_config.json').read_text())
        obj=cls(cfg['base_model_path'],cfg['objectives'],cfg['max_length'],
                lora_rank=0,device=device,local_files_only=True)
        if cfg['lora_rank']>0:
            from peft import PeftModel
            obj.backbone=PeftModel.from_pretrained(obj.backbone,path/'backbone',is_trainable=True)
        obj.feedback_head.load_state_dict(torch.load(path/'feedback_head.pt',map_location=device,weights_only=True))
        obj.run_config=cfg
        return obj.to(device)
