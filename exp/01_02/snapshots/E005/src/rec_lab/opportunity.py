"""Executable normalized batch directions and frozen-choice opportunity audits."""
import numpy as np

from .frontier import solve_frontier


def normalized_units(gradients, floor=1e-12):
    values = np.asarray(gradients)
    if values.ndim != 2 or not np.isfinite(values).all() or floor <= 0:
        raise ValueError("Finite unit gradients required")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    result = np.zeros_like(values, dtype=np.float32)
    active = norms > floor
    result[active] = values[active] / norms[active, None]
    return result, norms


def optimal_opportunity(responses):
    """Optimize the entire allowed pool, never prefer an inferior 'strict' pair."""
    response = np.asarray(responses, dtype=float)
    if response.ndim != 2 or not response.shape[1] or not np.isfinite(response).all():
        raise ValueError("Finite objectives x units required")
    augmented = np.column_stack([response, np.zeros(response.shape[0])])
    solved = solve_frontier(augmented)
    index = int(np.argmax(augmented.min(0)))
    weights = solved.weights[:-1]
    return {"weights": weights, "no_update_mass": float(solved.weights[-1]),
            "best_single": index if index < response.shape[1] else None,
            "single_margin": float(augmented[:, index].min()),
            "mixed_margin": solved.margin, "gap": float(solved.margin - augmented[:, index].min()),
            "support": np.flatnonzero(weights > 1e-8).tolist(), "solver_gap": solved.gap}


def executable_distribution(membership, unit_norms, weights):
    """Return q and scale such that G.T@q*scale equals normalized-unit mixture.

    Columns of membership must be distributions. Zero/no-update mass is permitted.
    A zero-norm unit is executable as a no-op and contributes no record mass.
    """
    p = np.asarray(membership, dtype=float)
    norms, w = np.asarray(unit_norms, dtype=float), np.asarray(weights, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(norms) or w.shape != norms.shape:
        raise ValueError("Mismatched dimensions")
    if not np.isfinite(p).all() or not np.isfinite(norms).all() or not np.isfinite(w).all():
        raise ValueError("Non-finite mixture")
    if np.any(p < 0) or np.any(w < -1e-10) or np.any(norms < 0) or w.sum() > 1 + 1e-7 or not np.allclose(p.sum(0), 1):
        raise ValueError("Invalid probabilities or norms")
    coefficient = np.divide(np.maximum(w, 0), norms, out=np.zeros_like(norms), where=norms > 1e-12)
    raw = p @ coefficient
    scale = float(raw.sum())
    return (raw / scale if scale > 0 else np.zeros(p.shape[0])), scale


def bootstrap_gap(rec, ground, weights, seed=10043, repeats=2000):
    """Descriptive paired bootstrap against best of ALL fixed single units.

    rec/ground hold per-record scaled directional gains; choices are fixed elsewhere.
    Task populations are independently resampled, pairing is retained across units.
    This is not an exact finite-sample confidence guarantee for nonsmooth maxima.
    """
    r, g, w = np.asarray(rec, dtype=float), np.asarray(ground, dtype=float), np.asarray(weights, dtype=float)
    if r.ndim != 2 or g.ndim != 2 or r.shape[1] != g.shape[1] or w.shape != (r.shape[1],):
        raise ValueError("Audit shape mismatch")
    if not len(r) or not len(g) or repeats < 1 or not all(np.isfinite(x).all() for x in [r, g, w]):
        raise ValueError("Invalid audit")
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, repeats, 100):
        count = min(100, repeats - start)
        rm = r[rng.integers(len(r), size=(count, len(r)))].mean(1)
        gm = g[rng.integers(len(g), size=(count, len(g)))].mean(1)
        best = np.maximum(0, np.minimum(rm, gm).max(1))
        samples.extend((np.minimum(rm @ w, gm @ w) - best).tolist())
    rm, gm = r.mean(0), g.mean(0)
    return {"gap": float(min(rm @ w, gm @ w) - max(0, np.minimum(rm, gm).max())),
            "interval_95": np.quantile(samples, [.025, .975]).tolist(),
            "mixed_scaled_gains": [float(rm @ w), float(gm @ w)],
            "best_single_index_on_audit": int(np.argmax(np.minimum(rm, gm))),
            "scope": "Descriptive development bootstrap with frozen choice; audit best is diagnostic, not reselection"}
