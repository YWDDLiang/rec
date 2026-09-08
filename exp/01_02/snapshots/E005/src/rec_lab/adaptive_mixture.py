"""Persistent executable behavior mixtures scored through current AdamW state."""
from dataclasses import replace
import time
import numpy as np
import torch
from torch.nn import functional as F

from .adam_mixture import bounded_proposal, predict_adamw_update
from .atoms import nuisance_residual, learn_dictionary, omp_encode
from .sid_policy import SIDPolicy


class WindowSelector:
    def __init__(self, lm, optimizer, references, catalog, config, method):
        self.lm, self.optimizer, self.references = lm, optimizer, references
        self.catalog, self.config, self.method = catalog, config, method
        self.policy = SIDPolicy(lm, catalog)
        self.features, self.nuisance = [], []
        rng = np.random.default_rng(config["seed"]+1501)
        self.buckets = rng.integers(128, size=lm.gradient_elements)
        self.signs = rng.choice(np.array([-1.,1.],dtype=np.float32), size=lm.gradient_elements)

    def rank_gradient(self, positive, negative):
        lm = self.lm
        lm.model.eval()
        result = torch.zeros(lm.gradient_elements, device=lm.device)
        mean_loss = 0.
        for start in range(0, len(positive), 8):
            pos, neg = positive[start:start+8], negative[start:start+8]
            losses = F.softplus(self.policy.log_probabilities(neg)-self.policy.log_probabilities(pos))
            loss = losses.sum()/len(positive)
            mean_loss += float(loss.detach())
            gradients = torch.autograd.grad(loss, lm.parameters, allow_unused=True)
            result += torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().reshape(-1)
                                 for p,g in zip(lm.parameters,gradients)])
        return result, mean_loss

    def choose(self, bank, base, window):
        config, lm = self.config, self.lm
        started = time.perf_counter()
        rng = np.random.default_rng(config["seed"]+4001*window)
        gradients, features, nuisance = [], [], []
        for unit in bank:
            subset = [unit[int(i)] for i in rng.choice(len(unit), config["query_records_per_unit"], replace=False)]
            gradient = lm.gradient(subset)
            gradients.append(gradient)
            norm = max(float(np.linalg.norm(gradient)),1e-12)
            features.append(np.bincount(self.buckets, weights=gradient*self.signs, minlength=128)/norm)
            nuisance.append(np.mean([r.nuisance for r in subset],axis=0))
        gradient_bank = torch.from_numpy(np.stack(gradients)).to(lm.device)
        nref = config["reference_rec_per_fold"]*2
        ref = [self.references["rec"][int(i)] for i in rng.choice(len(self.references["rec"]), nref, replace=False)]
        candidates = lm.ranking(ref, self.catalog, k=config["negative_beams"], batch_size=4,
                                return_records=True, return_candidates=True)["per_record"]
        negatives = []
        for record, output in zip(ref,candidates):
            sid = next((sid for sid in output["candidate_sids"] if sid != record.answer), None)
            if sid is None:
                sid = next(item["sid"] for item in self.catalog.values() if item["sid"] != record.answer)
            negatives.append(replace(record, record_id=record.record_id+":ranking_negative", answer=sid))
        half = config["reference_rec_per_fold"]
        first, first_loss = self.rank_gradient(ref[:half], negatives[:half])
        second, second_loss = self.rank_gradient(ref[half:], negatives[half:])
        ground_ref = [self.references["ground"][int(i)] for i in rng.choice(len(self.references["ground"]),config["reference_ground"],replace=False)]
        ground = torch.from_numpy(lm.gradient(ground_ref)).to(lm.device)
        references = torch.stack([first,second,ground])
        proposals, labels = [], []
        if self.method == "raw_window":
            proposals = list(np.eye(len(bank)))
            labels = [f"raw_unit_{i}" for i in range(len(bank))]
        else:
            self.features.extend(features)
            self.nuisance.extend(nuisance)
            self.features = self.features[-config["atom_history_units"]:]
            self.nuisance = self.nuisance[-config["atom_history_units"]:]
            train = np.asarray(self.features)
            shared = np.asarray(self.nuisance)
            residual, coefficient = nuisance_residual(train,shared)
            scale = max(float(np.sqrt(np.mean(residual**2))),1e-12)
            learned = learn_dictionary(residual/scale,config["atoms"],config["sparsity"],config["dictionary_iterations"],config["seed"])
            current = (np.asarray(features)-np.asarray(nuisance)@coefficient)/scale
            codes = omp_encode(current, learned["dictionary"], config["sparsity"])
            for atom in range(codes.shape[1]):
                for sign in [1,-1]:
                    mass = np.maximum(sign*codes[:,atom],0)
                    if np.count_nonzero(mass>1e-12) >= 3 and mass.sum()>1e-12:
                        proposals.append(mass/mass.sum())
                        labels.append(f"atom_{atom}_{sign}")
        for task_index, task in enumerate(["rec","ground"]):
            distribution = np.zeros(len(bank))
            distribution[task_index*config["units_per_task"]:(task_index+1)*config["units_per_task"]] = 1/config["units_per_task"]
            proposals.append(distribution)
            labels.append(task+"_mean")
        evaluated, fingerprints = [], set()

        def evaluate(weights, name):
            signature = np.round(weights,10).tobytes()
            if signature in fingerprints:
                return
            fingerprints.add(signature)
            gradient = torch.as_tensor(weights,device=lm.device,dtype=torch.float32)@gradient_bank
            update = predict_adamw_update(lm.parameters,self.optimizer,gradient,config["clip_norm"])
            response = (references@update).cpu().numpy().astype(float)
            evaluated.append({"weights":weights,"name":name,"response":response,
                              "score":float(min(response[:2])),"update_norm":float(update.norm())})

        evaluate(base,"baseline")
        baseline_response = evaluated[0]["response"]
        for proposal,name in zip(proposals,labels):
            for strength in config["proposal_strengths"]:
                weights = bounded_proposal(base,proposal,strength,config["maximum_weight_ratio"],config["minimum_base_mass"])
                evaluate(weights,f"{name}_mix{strength}")
        candidates = sorted(range(len(evaluated)),key=lambda i:evaluated[i]["score"],reverse=True)[:3]
        candidates += sorted(range(len(evaluated)),key=lambda i:evaluated[i]["response"][2],reverse=True)[:2]
        candidates = sorted(set(candidates))
        for left_index,left in enumerate(candidates):
            for right in candidates[left_index+1:]:
                for fraction in config["pair_strengths"]:
                    weights=(1-fraction)*evaluated[left]["weights"]+fraction*evaluated[right]["weights"]
                    evaluate(weights,f"pair_{left}_{right}_{fraction}")
        floor = min(0.,float(baseline_response[2]))
        allowed = [item for item in evaluated if item["response"][2]>=floor-1e-10]
        selected = max(allowed,key=lambda item:item["score"])
        info = {"selected":selected["name"],"predicted_response":selected["response"].tolist(),
                "baseline_predicted_response":baseline_response.tolist(),"predicted_update_norm":selected["update_norm"],
                "proposal_evaluations":len(evaluated),"learned_proposal_count":len(proposals)-2,
                "exact_query_records":len(bank)*config["query_records_per_unit"],
                "reference_records":len(ref)+len(ground_ref),"ranking_reference_loss":[first_loss,second_loss],
                "selection_seconds":time.perf_counter()-started,
                "scope":"Exact AdamW simulation for estimated current gradients; not a certificate for future sampled window updates"}
        return selected["weights"],info
