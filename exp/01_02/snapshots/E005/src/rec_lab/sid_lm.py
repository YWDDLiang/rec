"""Small pretrained LLM with frozen SID catalog and explicit LoRA gradients."""
from pathlib import Path
import json,re,time
import numpy as np
import torch
from torch.nn import functional as F

def readout_sketch(hidden,logits,targets,buckets,signs,projection):
    """Rank-one CountSketch of the partial output-weight gradient, hidden fixed."""
    probabilities=logits.float().softmax(-1)
    delta=torch.zeros(len(targets),projection.shape[1],device=hidden.device,dtype=torch.float32)
    delta.scatter_add_(1,buckets.unsqueeze(0).expand(len(targets),-1),probabilities*signs)
    rows=torch.arange(len(targets),device=hidden.device)
    delta[rows,buckets[targets]]-=signs[targets]
    return delta*(hidden.float()@projection)

class SIDLanguageModel:
    def __init__(self,model_path,data_dir,device="cuda",rank=8,max_length=384,
                 seed=2026,adapter=None,gradient_checkpointing=True,proxy_dim=64):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        from peft import LoraConfig,get_peft_model,PeftModel
        model_path=Path(model_path).resolve()
        if not model_path.is_relative_to(Path("/zhdd/home/ywliang/models").resolve()):
            raise ValueError("Base models must live in the requested model directory")
        self.run_config={"model_path":str(model_path),"data_dir":str(data_dir),"rank":rank,
                         "max_length":max_length,"seed":seed,"proxy_dim":proxy_dim}
        self.device=torch.device(device);self.max_length=max_length;self.cache={}
        torch.manual_seed(seed)
        manifest=json.loads((Path(data_dir)/"manifest.json").read_text())
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        self.tokenizer.add_special_tokens({"additional_special_tokens":manifest["new_tokens"]})
        self.tokenizer.pad_token=self.tokenizer.eos_token
        self.tokenizer.padding_side="left"
        self.code_ids=set(self.tokenizer.convert_tokens_to_ids(t) for t in manifest["new_tokens"]
                          if re.fullmatch(r"<s_[a-d]_\d+>",t))
        new_ids=self.tokenizer.convert_tokens_to_ids(manifest["new_tokens"])
        dtype=torch.bfloat16 if self.device.type=="cuda" else torch.float32
        base=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,
            trust_remote_code=False,torch_dtype=dtype,attn_implementation="sdpa")
        base.resize_token_embeddings(max(len(self.tokenizer),base.get_input_embeddings().num_embeddings),
                                     mean_resizing=False)
        if adapter:
            model=PeftModel.from_pretrained(base,adapter,is_trainable=True)
        else:
            config=LoraConfig(r=rank,lora_alpha=2*rank,lora_dropout=0.0,
                 target_modules=["q_proj","v_proj"],bias="none",task_type="CAUSAL_LM",
                 trainable_token_indices=new_ids)
            model=get_peft_model(base,config)
        self.model=model.to(self.device)
        self.model.config.use_cache=False
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            self.model.enable_input_require_grads()
        self.parameters=[p for _,p in self.model.named_parameters() if p.requires_grad]
        self.parameter_names=[n for n,p in self.model.named_parameters() if p.requires_grad]
        self.gradient_elements=sum(p.numel() for p in self.parameters)
        if self.gradient_elements>10_000_000:
            raise ValueError("Pilot exact-gradient subspace exceeds 10M parameters")
        generator=torch.Generator(device="cpu").manual_seed(seed+901)
        vocab=self.model.get_output_embeddings().weight.shape[0]
        hidden=self.model.config.hidden_size
        self.buckets=torch.randint(proxy_dim,(vocab,),generator=generator).to(self.device)
        self.signs=(torch.randint(2,(vocab,),generator=generator)*2-1).float().to(self.device)
        self.projection=(torch.randint(2,(hidden,proxy_dim),generator=generator)*2-1).float().to(self.device)

    def encode(self,record):
        key=(record.record_id,record.prompt,record.answer)
        if key in self.cache:return self.cache[key]
        messages=[{"role":"user","content":record.prompt}]
        text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,
                                                enable_thinking=False)
        prompt=self.tokenizer.encode(text,add_special_tokens=False)
        answer=self.tokenizer.encode(record.answer,add_special_tokens=False)
        code_positions=[i for i,t in enumerate(answer) if t in self.code_ids]
        if len(code_positions)!=4:
            raise ValueError("Every answer must have exactly four SID code tokens")
        allowance=self.max_length-len(answer)
        if allowance<16:raise ValueError("max_length too small")
        prompt=prompt[-allowance:]
        ids=prompt+answer
        positions=[len(prompt)+i-1 for i in code_positions]
        labels=[answer[i] for i in code_positions]
        value=(ids,positions,labels)
        self.cache[key]=value
        return value

    def batch(self,records):
        encoded=[self.encode(r) for r in records]
        length=max(len(x[0]) for x in encoded)
        ids=torch.full((len(records),length),self.tokenizer.pad_token_id,
                        dtype=torch.long,device=self.device)
        attention=torch.zeros_like(ids);positions=[];targets=[]
        for row,(seq,pos,label) in enumerate(encoded):
            pad=length-len(seq)
            ids[row,pad:]=torch.tensor(seq,device=self.device);attention[row,pad:]=1
            positions.extend((row,pad+p) for p in pos);targets.extend(label)
        return ids,attention,positions,torch.tensor(targets,device=self.device)

    def losses(self,records,with_proxy=False):
        ids,attention,positions,targets=self.batch(records)
        core=self.model.get_base_model()
        position_ids=(attention.cumsum(-1)-1).clamp_min(0)
        hidden=core.model(input_ids=ids,attention_mask=attention,position_ids=position_ids,
                          use_cache=False,return_dict=True).last_hidden_state
        row=torch.tensor([x[0] for x in positions],device=self.device)
        col=torch.tensor([x[1] for x in positions],device=self.device)
        state=hidden[row,col]
        logits=core.get_output_embeddings()(state)
        per=F.cross_entropy(logits.float(),targets,reduction="none").reshape(len(records),4).mean(1)
        if not torch.isfinite(per).all():raise FloatingPointError("Non-finite SID NLL")
        if with_proxy:
            proxy=readout_sketch(state,logits,targets,self.buckets,self.signs,self.projection)
            return per,proxy.reshape(len(records),4,-1).mean(1)
        return per

    @torch.no_grad()
    def evaluate(self,records,batch_size=8):
        self.model.eval()
        losses=[]
        for start in range(0,len(records),batch_size):
            losses.extend(self.losses(records[start:start+batch_size]).float().cpu().tolist())
        return {"sid_code_nll":float(np.mean(losses)),"n":len(losses)}

    @torch.no_grad()
    def proxies(self,records,batch_size=8):
        self.model.eval();blocks=[]
        for start in range(0,len(records),batch_size):
            _,proxy=self.losses(records[start:start+batch_size],with_proxy=True)
            blocks.append(proxy.float().cpu().numpy())
        return np.concatenate(blocks)

    def gradient(self,records,batch_size=8,return_loss=False):
        self.model.eval()
        total=np.zeros(self.gradient_elements,dtype=np.float32)
        mean_loss=0.
        for start in range(0,len(records),batch_size):
            chunk=records[start:start+batch_size]
            loss=self.losses(chunk).mean()
            mean_loss+=float(loss.detach())*len(chunk)/len(records)
            gradient=torch.autograd.grad(loss,self.parameters,allow_unused=True)
            flat=torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().reshape(-1)
                             for p,g in zip(self.parameters,gradient)])
            total+=flat.cpu().numpy()*(len(chunk)/len(records))
        return (total,mean_loss) if return_loss else total

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        self.model.save_pretrained(path,save_embedding_layers=False)
        self.tokenizer.save_pretrained(path)
        (path/"rec_atom_config.json").write_text(json.dumps(self.run_config,indent=2))

    @torch.no_grad()
    def ranking(self,records,catalog,k=20,batch_size=4,return_records=False,return_candidates=False):
        self.model.eval()
        trie={};sid_lookup={}
        for item in catalog.values():
            node=trie
            sequence=self.tokenizer.encode(item["sid"],add_special_tokens=False)
            if return_candidates:sid_lookup[tuple(sequence)]=item["sid"]
            for token in sequence:
                node=node.setdefault(token,{})
            node[self.tokenizer.eos_token_id]={}
        recalls={10:[],20:[]};ndcgs={10:[],20:[]};details=[]
        eos=self.tokenizer.eos_token_id
        for start in range(0,len(records),batch_size):
            chunk=records[start:start+batch_size];sequences=[]
            for record in chunk:
                text=self.tokenizer.apply_chat_template([{"role":"user","content":record.prompt}],
                        tokenize=False,add_generation_prompt=True,enable_thinking=False)
                sequences.append(self.tokenizer.encode(text,add_special_tokens=False)[-(self.max_length-6):])
            inputs=self.tokenizer.pad({"input_ids":sequences},padding=True,return_tensors="pt").to(self.device)
            width=inputs["input_ids"].shape[1]
            def allowed(batch_id,ids):
                node=trie
                for token in ids[width:].tolist():
                    if token==eos:return [eos]
                    if token not in node:return [eos]
                    node=node[token]
                return list(node) or [eos]
            output=self.model.generate(**inputs,num_beams=k,num_return_sequences=k,do_sample=False,
                max_new_tokens=7,prefix_allowed_tokens_fn=allowed,renormalize_logits=True,
                use_cache=True,pad_token_id=eos,eos_token_id=eos)
            output=output[:,width:].reshape(len(chunk),k,-1)
            for row,record in enumerate(chunk):
                truth=tuple(self.tokenizer.encode(record.answer,add_special_tokens=False))
                candidates=[tuple(ids[:len(truth)].tolist()) for ids in output[row]]
                rank=next((j+1 for j,x in enumerate(candidates) if x==truth),None)
                details.append({"record_id":record.record_id,"user_id":record.user_id,"item_id":record.item_id,"rank":rank})
                if return_candidates:details[-1]["candidate_sids"]=[sid_lookup[x] for x in candidates if x in sid_lookup]
                for cutoff in [10,20]:
                    hit=rank is not None and rank<=min(cutoff,k)
                    recalls[cutoff].append(float(hit))
                    ndcgs[cutoff].append(1/np.log2(rank+1) if hit else 0.)
        return {"n":len(records),**{f"recall@{x}":float(np.mean(recalls[x])) for x in recalls},
                **{f"ndcg@{x}":float(np.mean(ndcgs[x])) for x in ndcgs},
                "protocol":"full frozen catalog, constrained normalized beam search, no history masking",
                **({"per_record":details} if return_records else {})}

    def tail_layout(self,last_layers=2,dim=128):
        threshold=self.model.config.num_hidden_layers-last_layers
        indices=[];parameters=[];offset=0
        for name,parameter in zip(self.parameter_names,self.parameters):
            match=re.search(r"layers\.(\d+)\.",name)
            if match and int(match.group(1))>=threshold and "lora_" in name:
                indices.extend(range(offset,offset+parameter.numel()))
                parameters.append(parameter)
            offset+=parameter.numel()
        if not parameters:raise ValueError("No tail LoRA parameters")
        rng=np.random.default_rng(self.run_config["seed"]+1403)
        buckets=rng.integers(dim,size=len(indices))
        signs=rng.choice(np.array([-1.,1.],dtype=np.float32),size=len(indices))
        return np.asarray(indices),parameters,buckets,signs

    def tail_proxies(self,records,last_layers=2,dim=128):
        self.model.eval()
        _,parameters,buckets,signs=self.tail_layout(last_layers,dim)
        output=[]
        with torch.enable_grad():
            for record in records:
                loss=self.losses([record]).mean()
                grads=torch.autograd.grad(loss,parameters,allow_unused=True)
                flat=torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().reshape(-1)
                                for p,g in zip(parameters,grads)]).cpu().numpy()
                output.append(np.bincount(buckets,weights=flat*signs,minlength=dim))
        return np.asarray(output,dtype=np.float32)
