"""Our constrained four-code SID policy for matched SFT/RLOO diagnostics."""
from dataclasses import replace

import numpy as np
import torch

from .rl import group_advantages
from .sid_data import SID_PATTERN


class SIDPolicy:
    def __init__(self, language_model, catalog):
        self.lm = language_model
        self.catalog = catalog
        self.code_trie = {}
        self.sequence_trie = {}
        self.sid_by_sequence = {}
        eos = self.lm.tokenizer.eos_token_id
        for item in catalog.values():
            sequence = tuple(self.lm.tokenizer.encode(item["sid"], add_special_tokens=False))
            if sequence in self.sid_by_sequence:
                raise ValueError("Catalog SID collision")
            self.sid_by_sequence[sequence] = item["sid"]
            node = self.sequence_trie
            for token in sequence + (eos,):
                node = node.setdefault(token, {})
            codes = self.lm.tokenizer.convert_tokens_to_ids(SID_PATTERN.findall(item["sid"]))
            if len(codes) != 4:
                raise ValueError("Policy requires four variable code tokens")
            node = self.code_trie
            for token in codes:
                node = node.setdefault(token, {})

    def log_probabilities(self, records):
        """Log probability under legal-prefix normalization; wrappers forced."""
        lm = self.lm
        ids, attention, positions, targets = lm.batch(records)
        core = lm.model.get_base_model()
        hidden = core.model(
            input_ids=ids, attention_mask=attention,
            position_ids=(attention.cumsum(-1) - 1).clamp_min(0),
            use_cache=False, return_dict=True,
        ).last_hidden_state
        row = torch.tensor([entry[0] for entry in positions], device=lm.device)
        column = torch.tensor([entry[1] for entry in positions], device=lm.device)
        logits = core.get_output_embeddings()(hidden[row, column]).float()
        codes = targets.reshape(len(records), 4).detach().cpu().tolist()
        totals = []
        for index, item_codes in enumerate(codes):
            node = self.code_trie
            terms = []
            for depth, code in enumerate(item_codes):
                if code not in node:
                    raise ValueError("Target path missing from legal catalog")
                values = logits[index * 4 + depth]
                terms.append(values[code] - torch.logsumexp(values[list(node)], dim=0))
                node = node[code]
            totals.append(torch.stack(terms).sum())
        return torch.stack(totals)

    @torch.no_grad()
    def evaluate(self, records, microbatch=8):
        self.lm.model.eval()
        values = []
        for start in range(0, len(records), microbatch):
            values.extend((-self.log_probabilities(records[start:start + microbatch]) / 4).cpu().tolist())
        return np.asarray(values, dtype=np.float64)

    def gradient(self, records, coefficients=None, microbatch=8):
        """Gradient of -sum(coefficients * log pi)/4, in lm.parameters order."""
        self.lm.model.eval()
        if coefficients is None:
            coefficients = np.full(len(records), 1 / len(records))
        coefficients = np.asarray(coefficients, dtype=np.float32)
        if coefficients.shape != (len(records),) or not np.isfinite(coefficients).all():
            raise ValueError("Invalid gradient coefficients")
        total = np.zeros(self.lm.gradient_elements, dtype=np.float32)
        if not coefficients.any():
            return total
        for start in range(0, len(records), microbatch):
            weights = torch.as_tensor(coefficients[start:start + microbatch], device=self.lm.device)
            if not weights.any():
                continue
            value = -(self.log_probabilities(records[start:start + microbatch]) * weights).sum() / 4
            gradients = torch.autograd.grad(value, self.lm.parameters, allow_unused=True)
            offset = 0
            for parameter, gradient in zip(self.lm.parameters, gradients):
                count = parameter.numel()
                if gradient is not None:
                    total[offset:offset + count] += gradient.detach().float().reshape(-1).cpu().numpy()
                offset += count
        return total

    def mean_gradient(self, records, microbatch=8):
        return self.gradient(records, microbatch=microbatch)

    @torch.no_grad()
    def sample_groups(self, records, group_size, replicates=1, temperature=1.0):
        if group_size < 2 or replicates < 1 or temperature != 1.0:
            raise ValueError("This initial matched-policy experiment uses G>=2, temperature 1")
        lm = self.lm
        lm.model.eval()
        sequences = []
        for record in records:
            ids, _, _ = lm.encode(record)
            answer_length = len(lm.tokenizer.encode(record.answer, add_special_tokens=False))
            sequences.append(ids[:-answer_length])
        inputs = lm.tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt").to(lm.device)
        width = inputs["input_ids"].shape[1]
        eos = lm.tokenizer.eos_token_id

        def allowed(_batch_id, ids):
            node = self.sequence_trie
            for token in ids[width:].tolist():
                if token == eos:
                    return [eos]
                if token not in node:
                    raise ValueError("Generated prefix outside legal trie")
                node = node[token]
            return list(node) or [eos]

        outputs = lm.model.generate(
            **inputs, do_sample=True, num_beams=1,
            num_return_sequences=group_size * replicates,
            temperature=1.0, top_p=1.0, top_k=0,
            repetition_penalty=1.0, max_new_tokens=7,
            prefix_allowed_tokens_fn=allowed, renormalize_logits=True,
            use_cache=True, pad_token_id=eos, eos_token_id=eos,
            return_dict_in_generate=True, output_scores=True,
        )
        generated = outputs.sequences[:, width:]
        transition = lm.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True,
        ).sum(1).float().cpu().numpy()
        per_prompt = group_size * replicates
        results = []
        for index, record in enumerate(records):
            answers = []
            rewards = []
            for local_index in range(per_prompt):
                tokens = generated[index * per_prompt + local_index].tolist()
                if eos in tokens:
                    tokens = tokens[:tokens.index(eos)]
                sid = self.sid_by_sequence.get(tuple(tokens))
                if sid is None:
                    raise ValueError("Sampled output does not identify one catalog item")
                answers.append(sid)
                rewards.append(float(sid == record.answer))
            results.append({
                "record_id": record.record_id,
                "answers": answers,
                "rewards": rewards,
                "old_log_probabilities": transition[index * per_prompt:(index + 1) * per_prompt].tolist(),
                "group_size": group_size,
                "replicates": replicates,
                "generated_tokens": int(generated.shape[1] * per_prompt),
            })
        return results

    def rollout_gradient(self, record, rollout, microbatch=8):
        size = rollout["group_size"]
        replicates = rollout["replicates"]
        rewards = torch.tensor(rollout["rewards"], dtype=torch.float32).reshape(replicates, size, 1)
        advantage = group_advantages(rewards, "loo").reshape(-1).numpy()
        candidates = [replace(record, record_id=f"{record.record_id}:sample:{index}", answer=answer)
                      for index, answer in enumerate(rollout["answers"])]
        return self.gradient(candidates, advantage / (size * replicates), microbatch)
