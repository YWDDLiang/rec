"""Full-training-pool behavior mixtures with bounded importance-sampled responses."""
from dataclasses import replace
import time

import numpy as np
import torch
from torch.nn import functional as F

from .adam_mixture import predict_adamw_update


def cap_distribution(candidate, base, domain_ids, preserve_domain_mass=True, lower=.25, upper=4.):
    """Rescale and clip a positive proposal, respecting probability and domain mass."""
    candidate, base = np.asarray(candidate, float), np.asarray(base, float)
    domain_ids = np.asarray(domain_ids)
    if candidate.shape != base.shape or np.any(base <= 0) or np.any(candidate < 0):
        raise ValueError("Invalid distribution")
    if not np.isclose(base.sum(), 1) or not 0 <= lower < 1 < upper:
        raise ValueError("Invalid probability bounds")
    candidate = np.maximum(candidate, 1e-30)
    result = np.zeros_like(base)
    masks = [domain_ids == domain for domain in np.unique(domain_ids)] if preserve_domain_mass else [np.ones(len(base), bool)]
    for mask in masks:
        values, lower_bound, upper_bound = candidate[mask], lower*base[mask], upper*base[mask]
        target = float(base[mask].sum())
        left, right = 0., 1.
        while np.clip(right*values, lower_bound, upper_bound).sum() < target:
            right *= 2
        for _ in range(45):
            middle = (left+right)/2
            if np.clip(middle*values, lower_bound, upper_bound).sum() < target:
                left = middle
            else:
                right = middle
        result[mask] = np.clip((left+right)/2*values, lower_bound, upper_bound)
    if abs(result.sum()-1) > 1e-8:
        raise ArithmeticError("Capped probabilities do not sum to one")
    return result


def kmeans_groups(features, fit_indices, count, seed):
    rng = np.random.default_rng(seed)
    fit = np.asarray(features[fit_indices], dtype=np.float64)
    centers = fit[rng.choice(len(fit), count, replace=False)].copy()
    for _ in range(20):
        distance = (fit**2).sum(1)[:,None] + (centers**2).sum(1)[None,:] - 2*fit@centers.T
        labels = distance.argmin(1)
        for index in range(count):
            if np.any(labels == index):
                centers[index] = fit[labels == index].mean(0)
    labels = []
    for start in range(0, len(features), 4096):
        values = np.asarray(features[start:start+4096], dtype=np.float64)
        distance = (values**2).sum(1)[:,None] + (centers**2).sum(1)[None,:] - 2*values@centers.T
        labels.extend(distance.argmin(1))
    labels = np.asarray(labels)
    groups = []
    for index in range(count):
        mass = (labels == index).astype(float)
        if mass.sum() >= 256:
            groups.append(mass/mass.sum())
    return np.asarray(groups, dtype=np.float32), centers


class DomainMixtureSelector:
    def __init__(self, lm, optimizer, train, domain_ids, references, config, method, groups=None):
        self.lm, self.optimizer, self.train = lm, optimizer, train
        self.domain_ids, self.references, self.config, self.method = np.asarray(domain_ids), references, config, method
        self.domain_indices = [np.flatnonzero(self.domain_ids == index) for index in range(len(lm.domains))]
        self.base = np.zeros(len(train))
        for indices in self.domain_indices:
            self.base[indices] = 1/len(lm.domains)/len(indices)
        self.groups = groups

    def reference_gradient(self, positive, negative):
        lm = self.lm
        total = torch.zeros(lm.gradient_elements, device=lm.device)
        for start in range(0, len(positive), 8):
            pos, neg = positive[start:start+8], negative[start:start+8]
            loss = F.softplus(lm.log_probabilities(neg)-lm.log_probabilities(pos)).sum()/len(positive)
            grads = torch.autograd.grad(loss, lm.parameters, allow_unused=True)
            total += torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().reshape(-1)
                                for p,g in zip(lm.parameters, grads)])
        return total

    def choose(self, window):
        config, lm = self.config, self.lm
        started = time.perf_counter()
        rng = np.random.default_rng(config["seed"] + 4001*window)
        query_ids = np.concatenate([rng.choice(indices, config["query_per_domain"], replace=False) for indices in self.domain_indices])
        gradients = [lm.gradient([self.train[int(index)]]) for index in query_ids]
        gradient_bank = torch.from_numpy(np.stack(gradients)).to(lm.device)
        del gradients
        reference_gradients, reference_ids = [], []
        for domain in lm.domains:
            rows = self.references[domain]
            chosen = rng.choice(len(rows), 2*config["reference_per_fold"], replace=False)
            positive = [rows[int(index)] for index in chosen]
            output = lm.rank_domain(positive, domain, beams=config["negative_beams"], batch_size=8, candidates=True)["per_record"]
            negative = []
            for record, result in zip(positive, output):
                answer = next((sid for sid in result["candidate_sids"] if sid != record.answer), None)
                if answer is None:
                    answer = next(item["sid"] for item in lm.catalogs[domain].values() if item["sid"] != record.answer)
                negative.append(replace(record, record_id=record.record_id+":competitor", answer=answer))
            half = config["reference_per_fold"]
            for lo, hi in [(0, half), (half, 2*half)]:
                reference_gradients.append(self.reference_gradient(positive[lo:hi], negative[lo:hi]))
            reference_ids += [record.record_id for record in positive]
        references = torch.stack(reference_gradients)
        scales = references.norm(dim=1).clamp_min(1e-8)
        proposals = [(self.base, "baseline")]
        if self.method == "domain_window":
            for mass in [.25, .375, .625, .75]:
                distribution = self.base.copy()
                distribution[self.domain_ids == 0] *= 2*mass
                distribution[self.domain_ids == 1] *= 2*(1-mass)
                proposals.append((distribution, f"domain_mass_{mass}"))
        else:
            for index, group in enumerate(self.groups):
                for strength in config["proposal_strengths"]:
                    distribution = cap_distribution((1-strength)*self.base + strength*group, self.base, self.domain_ids,
                        True, config["minimum_base_mass"], config["maximum_weight_ratio"])
                    proposals.append((distribution, f"group_{index}_mix{strength}"))
        evaluated = []

        def evaluate(distribution, label):
            # Stratified Horvitz-Thompson estimate. q/p0 <= 4 limits importance weights.
            coefficients = distribution[query_ids]/self.base[query_ids]/len(query_ids)
            gradient = torch.as_tensor(coefficients, dtype=torch.float32, device=lm.device)@gradient_bank
            update = predict_adamw_update(lm.parameters, self.optimizer, gradient, config["clip_norm"])
            response = ((references@update)/scales).detach().cpu().numpy()
            evaluated.append({"weights": distribution, "name": label, "response": response,
                "score": float(response.min()), "estimated_importance_mass": float(coefficients.sum()),
                "predicted_update_norm": float(update.norm())})

        for distribution, label in proposals:
            evaluate(distribution, label)
        # Small candidate menu; actual AdamW is reevaluated for each combined distribution.
        best = sorted(range(len(evaluated)), key=lambda index: evaluated[index]["score"], reverse=True)[:4]
        if self.method != "domain_window":
            best += [max(range(len(evaluated)), key=lambda i: min(evaluated[i]["response"][2*d:2*d+2])) for d in [0,1]]
            best = sorted(set(best))
            for index, left in enumerate(best):
                for right in best[index+1:]:
                    for fraction in [.25, .5, .75]:
                        evaluate((1-fraction)*evaluated[left]["weights"] + fraction*evaluated[right]["weights"],
                                 f"pair_{left}_{right}_{fraction}")
        selected = max(evaluated, key=lambda value: value["score"])
        info = {key: value for key, value in selected.items() if key not in ["weights", "response"]}
        info.update(predicted_response=selected["response"].tolist(), baseline_predicted_response=evaluated[0]["response"].tolist(),
            query_record_ids=[self.train[int(index)].record_id for index in query_ids], reference_record_ids=reference_ids,
            query_records=len(query_ids), reference_records=len(reference_ids), proposal_evaluations=len(evaluated),
            selection_seconds=time.perf_counter()-started, domain_mass=[float(selected["weights"][self.domain_ids == d].sum()) for d in [0,1]],
            scope="Unbiased stratified gradient estimate before nonlinear AdamW/clipping; finite proposal menu, not a population or future-window certificate")
        return selected["weights"], info
