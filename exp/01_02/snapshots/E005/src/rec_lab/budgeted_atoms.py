"""Budgeted IDEA-01+02 selection; full-gradient access is explicit and counted.

The atoms are a proposal representation. Actual updates and responses are never
residualized. No guarantee for unqueried response columns is claimed.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np
from .atoms import nuisance_residual, omp_encode
from .frontier import solve_frontier, response_with_floor, solve_capped_frontier
from .numerics import array

def farthest_first(features, count, seed=0):
    X = array(features, 2)
    n = len(X)
    if not 1 <= count <= n:
        raise ValueError("count must be in [1,n]")
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(n))]
    distance = ((X - X[chosen[0]]) ** 2).sum(1)
    for _ in range(1, count):
        distance[chosen] = -np.inf
        index = int(np.argmax(distance))
        chosen.append(index)
        distance = np.minimum(distance, ((X - X[index]) ** 2).sum(1))
    return np.asarray(chosen, dtype=int)

def fit_atoms(proxies, nuisance, n_atoms=16, sparsity=2, iterations=4, seed=0):
    X = array(proxies, 2)
    Z = array(nuisance, 2)
    if len(X) != len(Z) or not len(X):
        raise ValueError("proxy/nuisance rows must match")
    if not 1 <= n_atoms <= len(X) or not 1 <= sparsity <= n_atoms or iterations < 1:
        raise ValueError("invalid dictionary configuration")
    residual, coefficient = nuisance_residual(X, Z)
    scale = max(float(np.sqrt(np.mean(residual ** 2))), 1e-12)
    R = residual / scale
    seeds = farthest_first(R, n_atoms, seed)
    D = R[seeds].copy()
    rng = np.random.default_rng(seed)
    for j in range(n_atoms):
        if np.linalg.norm(D[j]) < 1e-12:
            D[j] = rng.normal(size=X.shape[1])
    D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    best = None
    for _ in range(iterations):
        C = omp_encode(R, D, sparsity)
        error = float(np.mean((R - C @ D) ** 2))
        if best is None or error < best[0]:
            best = (error, C.copy(), D.copy())
        D = np.linalg.lstsq(C, R, rcond=1e-8)[0]
        norms = np.linalg.norm(D, axis=1)
        for j in range(n_atoms):
            if norms[j] < 1e-10:
                D[j] = R[seeds[j]] + 1e-5 * rng.normal(size=X.shape[1])
        D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    return {"codes": best[1], "dictionary": best[2], "mse": best[0],
            "residual": residual, "nuisance_coefficient": coefficient,
            "scale": scale, "scope": "training-fold proxy decomposition only"}

class QueryLedger:
    def __init__(self, oracle: Callable, pool_size: int, budget: int):
        if not 1 <= budget <= pool_size:
            raise ValueError("invalid exact-gradient query budget")
        self.oracle = oracle
        self.pool_size = pool_size
        self.budget = budget
        self.queried = []
        self.responses = {}

    def query(self, indices):
        raw = np.asarray(indices)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("indices must be a one-dimensional integer array")
        ids = list(map(int, raw))
        if len(set(ids)) != len(ids) or any(i < 0 or i >= self.pool_size for i in ids):
            raise ValueError("duplicate or out-of-range query")
        new = [i for i in ids if i not in self.responses]
        if len(self.queried) + len(new) > self.budget:
            raise ValueError("exact-gradient query budget exceeded")
        if new:
            A = array(self.oracle(new), 2)
            if A.shape[1] != len(new) or not A.shape[0]:
                raise ValueError("oracle must return objectives x queried columns")
            if self.responses and len(next(iter(self.responses.values()))) != len(A):
                raise ValueError("oracle objective count changed")
            for j, i in enumerate(new):
                self.responses[i] = A[:, j].copy()
            self.queried.extend(new)
        if not ids:
            raise ValueError("empty query")
        return np.stack([self.responses[i] for i in ids], axis=1)

def select_budgeted(proxies, nuisance, oracle, budget, method="atoms_frontier",
                    n_atoms=16, sparsity=2, floor=0.0, seed=0, max_weight_multiple=None, preserve_shared=False):
    X = array(proxies, 2)
    Z = array(nuisance, 2)
    n = len(X)
    if len(Z) != n or not 1 <= budget <= n or not 0 <= floor <= 1:
        raise ValueError("invalid pool, budget or floor")
    rng = np.random.default_rng(seed)
    fit = None
    if method in {"random_frontier", "uniform"}:
        ids = rng.choice(n, budget, replace=False)
    elif method == "raw_frontier":
        ids = farthest_first(X, budget, seed)
    elif method == "residual_frontier":
        residual, _ = nuisance_residual(X, Z)
        ids = farthest_first(residual, budget, seed)
    elif method in {"atoms_frontier", "atoms_uniform", "random_dictionary_frontier"}:
        k = min(n_atoms, n)
        if method == "random_dictionary_frontier":
            residual, coefficient = nuisance_residual(X, Z)
            scale = max(float(np.sqrt(np.mean(residual ** 2))), 1e-12)
            D = rng.normal(size=(k,X.shape[1]))
            D /= np.linalg.norm(D, axis=1, keepdims=True)
            codes = omp_encode(residual / scale, D, min(sparsity, k))
            fit = {"mse":float(np.mean((residual/scale-codes@D)**2))}
        else:
            fit = fit_atoms(X, Z, k, min(sparsity, k), seed=seed)
            codes = fit["codes"]
        if preserve_shared:
            def unit_block(block):
                centered=block-block.mean(0,keepdims=True)
                rms=max(float(np.sqrt(np.mean(np.sum(centered**2,axis=1)))),1e-12)
                return centered/rms
            nomination_features=np.c_[unit_block(codes),unit_block(Z)]
        else:
            nomination_features=codes
        ids = farthest_first(nomination_features, budget, seed)
    else:
        raise ValueError("unknown selection method")
    info = {"indices": ids.tolist(), "method": method,
            "exact_query_budget": budget, "exact_query_count": 0,
            "global_certificate": False, "scope": "queried candidate pool only", "preserve_shared": bool(preserve_shared)}
    if fit is not None:
        info["atom_reconstruction_mse"] = fit["mse"]
    if method in {"uniform", "atoms_uniform"}:
        info["weights"] = (np.ones(len(ids)) / len(ids)).tolist()
        return info
    if oracle is None:
        raise ValueError("frontier selection needs an explicit response oracle")
    ledger = QueryLedger(oracle, n, budget)
    A = ledger.query(np.asarray(ids, dtype=int))
    p = np.ones(len(ids)) / len(ids)
    if max_weight_multiple is None:
        result = solve_frontier(response_with_floor(A, p, floor))
        weights = floor * p + (1 - floor) * result.weights
    else:
        result = solve_capped_frontier(A,max_weight_multiple,floor)
        weights = result.weights
        info["max_weight_multiple"] = max_weight_multiple
    info.update(weights=weights.tolist(), exact_query_count=len(ledger.queried),
                response_margin=float((A @ weights).min()),
                local_matrix_gap=float(result.gap),
                local_matrix_solved=bool(result.certified))
    return info

def response_cover_radius(full_responses, queried_indices):
    """Diagnostic needing the full matrix; never called by select_budgeted."""
    A = array(full_responses, 2)
    S = np.asarray(queried_indices)
    if S.ndim != 1 or not np.issubdtype(S.dtype, np.integer) or not len(S):
        raise ValueError("nonempty integer subset required")
    if np.any(S < 0) or np.any(S >= A.shape[1]):
        raise ValueError("subset out of range")
    distance = np.max(np.abs(A[:, :, None] - A[:, S][:, None, :]), axis=0)
    return float(np.min(distance, axis=1).max())
