"""Pure shadow AdamW updates and bounded distributions for window selection."""
import copy
import numpy as np
import torch


def bounded_proposal(base, proposal, strength, maximum_ratio=4., minimum_base_mass=.25):
    base, proposal = np.asarray(base, dtype=float), np.asarray(proposal, dtype=float)
    if base.shape != proposal.shape or base.ndim != 1 or np.any(base <= 0) or np.any(proposal < 0):
        raise ValueError("Invalid proposal")
    if not np.isclose(base.sum(), 1) or not np.isclose(proposal.sum(), 1) or not 0 <= strength <= 1 or maximum_ratio < 1:
        raise ValueError("Invalid distribution constraints")
    ratio = float(np.max(proposal / base))
    allowed = (maximum_ratio - 1) / (ratio - 1) if ratio > 1 + 1e-12 else 1.
    actual = min(strength, 1-minimum_base_mass, allowed)
    result = (1-actual)*base + actual*proposal
    if np.any(result > maximum_ratio*base + 1e-10) or np.any(result < minimum_base_mass*base - 1e-10):
        raise RuntimeError("Distribution constraint violation")
    return result


@torch.no_grad()
def predict_adamw_update(parameters, optimizer, flat_gradient, clip_norm=1.):
    """Return theta-theta_next using a private copy of actual AdamW state.

    Includes parameter-group learning rates, dtype rounding, moments, bias
    correction, weight decay and clipping. Does not mutate model or optimizer.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("AdamW required")
    if flat_gradient.numel() != sum(p.numel() for p in parameters):
        raise ValueError("Gradient size differs from parameters")
    clones = [torch.nn.Parameter(p.detach().clone(), requires_grad=False) for p in parameters]
    indices = {id(parameter): i for i, parameter in enumerate(parameters)}
    groups = []
    for group in optimizer.param_groups:
        replacement = {key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
        replacement["params"] = [clones[indices[id(p)]] for p in group["params"]]
        groups.append(replacement)
    shadow = torch.optim.AdamW(groups)
    shadow.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    offset = 0
    for clone in clones:
        size = clone.numel()
        clone.grad = flat_gradient[offset:offset+size].reshape_as(clone).to(clone.dtype).clone()
        offset += size
    torch.nn.utils.clip_grad_norm_(clones, clip_norm)
    shadow.step()
    return torch.cat([(original.detach().float()-updated.detach().float()).reshape(-1)
                      for original, updated in zip(parameters, clones)])
