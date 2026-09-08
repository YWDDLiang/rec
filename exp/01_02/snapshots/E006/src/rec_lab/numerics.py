"""Strict numerical contracts shared by all research directions."""
from __future__ import annotations
import numpy as np

def array(x, ndim=None, name="array"):
    a = np.asarray(x, dtype=np.float64)
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"{name}: expected {ndim} dimensions, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: non-finite values")
    return a

def probability(x, name="probabilities"):
    a = array(x, 1, name)
    if a.size == 0 or np.any(a < 0) or a.sum() <= 0:
        raise ValueError(f"{name}: must have nonnegative, nonzero mass")
    return a / a.sum()

def simplex(x, name="simplex"):
    a = array(x, 1, name)
    if a.size == 0 or np.any(a < 0) or not np.isclose(a.sum(), 1, atol=1e-8):
        raise ValueError(f"{name}: expected normalized probability vector")
    return a

def softmax(x):
    z = array(x)
    z = z-z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e/e.sum(axis=-1, keepdims=True)
