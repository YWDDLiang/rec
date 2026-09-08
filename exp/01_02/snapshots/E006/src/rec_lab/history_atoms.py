"""History-sensitive persistent atoms and guarded response selection.

This module contains the v2 IDEA-01+02 primitives.  A persistent atom bank is
fit on an explicit training window and only transformed on later candidate
pools.  Exact response access remains explicit and budgeted.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import linprog

from .atoms import omp_encode
from .budgeted_atoms import QueryLedger, farthest_first, fit_atoms
from .numerics import array


def history_sensitive_proxy(full_history_proxy, control_history_proxies):
    """Subtract matched short/permuted-history controls from a full-history proxy.

    Controls may have shape ``[records, dimensions]`` or
    ``[records, controls, dimensions]``.  The operation has an exact algebraic
    cancellation property for components shared by full and control views; it
    is not, by itself, a causal effect estimator.
    """
    full = array(full_history_proxy, 2)
    controls = np.asarray(control_history_proxies, dtype=float)
    if controls.ndim == 2:
        controls = controls[:, None, :]
    if controls.ndim != 3 or controls.shape[0] != len(full) or controls.shape[2] != full.shape[1]:
        raise ValueError("controls must match full-history records and dimensions")
    if controls.shape[1] < 1 or not np.isfinite(controls).all():
        raise ValueError("at least one finite control per record is required")
    return full - controls.mean(axis=1)


@dataclass(frozen=True)
class PersistentAtomBank:
    nuisance_coefficient: np.ndarray
    scale: float
    dictionary: np.ndarray
    sparsity: int
    training_reconstruction_mse: float

    def transform(self, proxies, nuisance):
        """Encode a later pool without fitting nuisance coefficients or atoms."""
        values = array(proxies, 2)
        shared = array(nuisance, 2)
        if len(values) != len(shared) or shared.shape[1] != self.nuisance_coefficient.shape[0]:
            raise ValueError("proxy/nuisance shape differs from the fitted bank")
        if values.shape[1] != self.dictionary.shape[1]:
            raise ValueError("proxy dimension differs from the fitted bank")
        residual = (values - shared @ self.nuisance_coefficient) / self.scale
        return omp_encode(residual, self.dictionary, self.sparsity)


def fit_persistent_atom_bank(proxies, nuisance, n_atoms=16, sparsity=2,
                             iterations=8, seed=0):
    """Fit once on a declared replay/training window, then freeze for selection."""
    fit = fit_atoms(proxies, nuisance, n_atoms=n_atoms, sparsity=sparsity,
                    iterations=iterations, seed=seed)
    return PersistentAtomBank(
        nuisance_coefficient=fit["nuisance_coefficient"],
        scale=float(fit["scale"]),
        dictionary=fit["dictionary"],
        sparsity=sparsity,
        training_reconstruction_mse=float(fit["mse"]),
    )


@dataclass(frozen=True)
class GuardedBatchResult:
    weights: np.ndarray
    lower_response: np.ndarray
    primary_value: float
    guardrails: dict
    max_weight: float
    scope: str = "provided conservative response matrix"


def solve_guarded_batch(lower_responses, primary_weights, guardrails,
                        max_weight_multiple=2.0, floor=0.0):
    """Maximize a primary lower bound while enforcing response guardrails.

    ``lower_responses[m, i]`` must be an exact response or a justified lower
    bound.  A feasible solution therefore satisfies the same guardrails for the
    unknown true responses.  This implication is conditional on validity of the
    supplied lower bounds.
    """
    lower = array(lower_responses, 2)
    primary = array(primary_weights, 1)
    objectives, records = lower.shape
    if primary.shape != (objectives,) or np.any(primary < 0) or primary.sum() <= 0:
        raise ValueError("primary weights must be nonnegative and nonzero")
    if not isinstance(guardrails, dict):
        raise ValueError("guardrails must map objective rows to minimum responses")
    if not 1 <= max_weight_multiple <= records or not 0 <= floor <= 1:
        raise ValueError("invalid cap or floor")
    rows = []
    thresholds = []
    for raw_row, raw_threshold in guardrails.items():
        row = int(raw_row)
        threshold = float(raw_threshold)
        if row != raw_row or not 0 <= row < objectives or not np.isfinite(threshold):
            raise ValueError("invalid guardrail")
        rows.append(row)
        thresholds.append(threshold)
    primary = primary / primary.sum()
    lower_bound = floor / records
    upper_bound = max_weight_multiple / records
    result = linprog(
        -(primary @ lower),
        A_ub=-lower[rows] if rows else None,
        b_ub=-np.asarray(thresholds) if rows else None,
        A_eq=np.ones((1, records)),
        b_eq=[1.0],
        bounds=[(lower_bound, upper_bound)] * records,
        method="highs",
    )
    if not result.success:
        raise RuntimeError("guarded batch is infeasible: " + result.message)
    weights = np.maximum(result.x, 0)
    weights /= weights.sum()
    response = lower @ weights
    tolerance = 1e-8 * (1 + np.max(np.abs(response)))
    if any(response[row] < threshold - tolerance for row, threshold in zip(rows, thresholds)):
        raise RuntimeError("guarded batch failed numerical feasibility")
    return GuardedBatchResult(
        weights=weights,
        lower_response=response,
        primary_value=float(primary @ response),
        guardrails={row: threshold for row, threshold in zip(rows, thresholds)},
        max_weight=float(weights.max()),
    )


def _ridge_response_fit(features, responses, ridge):
    design = np.c_[np.ones(len(features)), features]
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    gram = design.T @ design + penalty
    inverse = np.linalg.pinv(gram, rcond=1e-10)
    coefficient = inverse @ design.T @ responses.T
    prediction = design @ coefficient
    leverage = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", design, inverse, design), 0.0))
    return prediction.T, leverage


def active_response_queries(features, oracle: Callable, budget, initial_queries=2,
                            ridge=1e-3, exploration=1.0, seed=0):
    """Adapt exact queries using optimistic multi-objective response prediction.

    The leverage bonus is an exploration heuristic, not a calibrated confidence
    interval.  Only queried exact responses are returned for downstream updates.
    """
    values = array(features, 2)
    records = len(values)
    if not 1 <= initial_queries <= budget <= records:
        raise ValueError("invalid initial or total query budget")
    if ridge <= 0 or exploration < 0:
        raise ValueError("ridge must be positive and exploration nonnegative")
    centered = values - values.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(centered ** 2, axis=0))
    normalized = centered / np.where(scale > 1e-12, scale, 1.0)
    initial = farthest_first(normalized, initial_queries, seed)
    ledger = QueryLedger(oracle, records, budget)
    ledger.query(initial)
    trace = [{"queried": int(index), "stage": "initial"} for index in initial]
    while len(ledger.queried) < budget:
        queried = np.asarray(ledger.queried, dtype=int)
        response = ledger.query(queried)
        prediction, leverage = _ridge_response_fit(normalized[queried], response, ridge)
        all_design = np.c_[np.ones(records), normalized]
        query_design = np.c_[np.ones(len(queried)), normalized[queried]]
        penalty = np.eye(all_design.shape[1]) * ridge
        penalty[0, 0] = 0.0
        inverse = np.linalg.pinv(query_design.T @ query_design + penalty, rcond=1e-10)
        coefficient = inverse @ query_design.T @ response.T
        all_prediction = (all_design @ coefficient).T
        all_leverage = np.sqrt(np.maximum(
            np.einsum("ij,jk,ik->i", all_design, inverse, all_design), 0.0))
        optimistic = np.min(all_prediction, axis=0) + exploration * all_leverage
        optimistic[queried] = -np.inf
        chosen = int(np.argmax(optimistic))
        ledger.query(np.asarray([chosen]))
        trace.append({"queried": chosen, "stage": "adaptive",
                      "optimistic_score": float(optimistic[chosen])})
    queried = np.asarray(ledger.queried, dtype=int)
    exact = ledger.query(queried)
    fitted, in_sample_leverage = _ridge_response_fit(normalized[queried], exact, ridge)
    return {
        "indices": queried,
        "responses": exact,
        "query_count": len(queried),
        "trace": trace,
        "queried_fit": fitted,
        "queried_leverage": in_sample_leverage,
        "scope": "exact responses for queried records only; leverage bonus is uncalibrated",
    }
