"""Domain-conditioned legal-SID SFT, with shared backbone and disjoint codebooks."""
from collections import defaultdict
import json
from pathlib import Path
import re

import torch
from torch.nn import functional as F

from .sid_data import SID_PATTERN
from .sid_lm import SIDLanguageModel


UPSTREAM_TOKEN = re.compile(r"<([a-d])_(\d+)>")


def namespace_catalog(index, metadata, domain, domain_index):
    """Preserve released semantic prefixes; append a deterministic collision code."""
    prefixes = defaultdict(list)
    for item_id, tokens in index.items():
        codes = [UPSTREAM_TOKEN.fullmatch(token) for token in tokens]
        if len(codes) != 3 or any(code is None for code in codes):
            raise ValueError("Expected the pinned release's three-level codes")
        if [code.group(1) for code in codes] != list("abc"):
            raise ValueError("Unexpected code-level order")
        values = tuple(int(code.group(2)) for code in codes)
        if any(value < 0 or value >= 256 for value in values):
            raise ValueError("Unexpected codebook size")
        prefixes[values].append(str(item_id))
    result, collisions = {}, []
    offset = 256 * domain_index
    for prefix, ids in sorted(prefixes.items()):
        if len(ids) > 256:
            raise ValueError("Collision suffix exhausted")
        for suffix, item_id in enumerate(sorted(ids)):
            codes = tuple(prefix) + (suffix,)
            sid = "<|sid_begin|>" + "".join(f"<s_{level}_{value+offset}>" for level, value in zip("abcd", codes)) + "<|sid_end|>"
            result[f"{domain}:{item_id}"] = {**metadata.get(item_id, {}), "sid": sid,
                "domain": domain, "original_item_id": item_id}
            if suffix:
                collisions.append({"item_id": item_id, "semantic_prefix": list(prefix), "suffix": suffix})
    if len({item["sid"] for item in result.values()}) != len(result):
        raise AssertionError("SID collision survived namespace construction")
    return result, collisions


def legal_code_layout(tokenizer, catalogs):
    """Return each target's allowed tokens at every legal-prefix decision."""
    layouts = {}
    for domain, catalog in catalogs.items():
        trie = {}
        for item in catalog.values():
            node = trie
            for code in tokenizer.convert_tokens_to_ids(SID_PATTERN.findall(item["sid"])):
                node = node.setdefault(code, {})
        for item in catalog.values():
            sid = item["sid"]
            codes = tokenizer.convert_tokens_to_ids(SID_PATTERN.findall(sid))
            node, allowed, labels = trie, [], []
            for code in codes:
                choices = tuple(sorted(node))
                allowed.append(choices)
                labels.append(choices.index(code))
                node = node[code]
            key = (domain, sid)
            layouts[key] = (allowed, labels)
    return layouts


def legal_decisions(logits, allowed, labels):
    width = max(len(row) for row in allowed)
    indices = torch.zeros((len(allowed), width), dtype=torch.long, device=logits.device)
    mask = torch.zeros_like(indices, dtype=torch.bool)
    for index, choices in enumerate(allowed):
        indices[index, :len(choices)] = torch.as_tensor(choices, device=logits.device)
        mask[index, :len(choices)] = True
    restricted = logits.float().gather(1, indices).masked_fill(~mask, -torch.inf)
    targets = torch.as_tensor(labels, dtype=torch.long, device=logits.device)
    return restricted, targets, indices


def legal_cross_entropy(logits, allowed, labels):
    """Vectorized restricted CE; nonexistent branches never enter normalization."""
    restricted, targets, _ = legal_decisions(logits, allowed, labels)
    return F.cross_entropy(restricted, targets, reduction="none")


class MultiDomainSIDLanguageModel(SIDLanguageModel):
    def __init__(self, model_path, data_dir, **kwargs):
        super().__init__(model_path, data_dir, **kwargs)
        manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
        self.domains = list(manifest["domains"])
        self.catalogs = {domain: json.loads((Path(data_dir) / f"catalog_{domain}.json").read_text())
                         for domain in self.domains}
        self.legal_layout = legal_code_layout(self.tokenizer, self.catalogs)

    def losses(self, records, with_proxy=False):
        ids, attention, positions, _ = self.batch(records)
        core = self.model.get_base_model()
        hidden = core.model(input_ids=ids, attention_mask=attention,
            position_ids=(attention.cumsum(-1)-1).clamp_min(0), use_cache=False,
            return_dict=True).last_hidden_state
        row = torch.as_tensor([x[0] for x in positions], device=self.device)
        col = torch.as_tensor([x[1] for x in positions], device=self.device)
        states = hidden[row, col]
        logits = core.get_output_embeddings()(states)
        allowed, labels = [], []
        for record in records:
            choices, targets = self.legal_layout[(record.task, record.answer)]
            allowed.extend(choices)
            labels.extend(targets)
        restricted, targets, indices = legal_decisions(logits, allowed, labels)
        losses = F.cross_entropy(restricted, targets, reduction="none").reshape(len(records), 4).mean(1)
        if not torch.isfinite(losses).all():
            raise FloatingPointError("Nonfinite domain-conditioned SID loss")
        if with_proxy:
            probabilities = restricted.softmax(-1)
            delta = torch.zeros((len(indices), self.projection.shape[1]), device=self.device)
            delta.scatter_add_(1, self.buckets[indices], probabilities*self.signs[indices])
            rows = torch.arange(len(indices), device=self.device)
            target_ids = indices[rows, targets]
            delta[rows, self.buckets[target_ids]] -= self.signs[target_ids]
            proxy = delta*(states.float()@self.projection)
            return losses, proxy.reshape(len(records), 4, -1).mean(1)
        return losses

    def log_probabilities(self, records):
        return -4 * self.losses(records)

    def rank_domain(self, records, domain, beams=10, batch_size=8, candidates=False):
        if any(record.task != domain for record in records):
            raise ValueError("Mixed domains must not share a decoding catalog")
        result = self.ranking(records, self.catalogs[domain], k=beams, batch_size=batch_size,
                              return_records=True, return_candidates=candidates)
        return {key: value for key, value in result.items() if "@20" not in key} | {
            "domain": domain, "beam_size": beams, "catalog_items": len(self.catalogs[domain])}
