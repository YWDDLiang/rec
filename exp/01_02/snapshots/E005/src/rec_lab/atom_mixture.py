"""Signed behavior atoms as actual data-mixture units, with pair diagnostics."""
from dataclasses import dataclass
import numpy as np

from .atoms import learn_dictionary, nuisance_residual, omp_encode


@dataclass
class MixtureUnits:
    membership: np.ndarray  # records x units; every column is a distribution
    names: list
    metadata: list

    def mix(self, weights):
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (self.membership.shape[1],) or np.any(weights < 0):
            raise ValueError("Invalid mixture weights")
        if not np.isclose(weights.sum(), 1):
            raise ValueError("Mixture weights must sum to one")
        return self.membership @ weights


def signed_units(codes, minimum_records=8, minimum_ess=6):
    codes = np.asarray(codes, dtype=float)
    if codes.ndim != 2 or not len(codes) or not np.isfinite(codes).all():
        raise ValueError("Finite record x atom codes required")
    columns, names, metadata = [], [], []
    for atom in range(codes.shape[1]):
        for sign in [1, -1]:
            mass = np.maximum(sign * codes[:, atom], 0)
            support = int(np.count_nonzero(mass > 1e-12))
            if support < minimum_records or mass.sum() <= 1e-12:
                continue
            probability = mass / mass.sum()
            ess = float(1 / (probability @ probability))
            if ess < minimum_ess:
                continue
            columns.append(probability)
            names.append(f"atom_{atom}_{'positive' if sign > 0 else 'negative'}")
            metadata.append({"atom": atom, "sign": sign, "support": support, "ess": ess})
    if not columns:
        raise ValueError("No atom unit meets the predeclared support and ESS floors")
    return MixtureUnits(np.stack(columns, axis=1), names, metadata)


def build_units(fit_features, candidate_features, fit_nuisance, candidate_nuisance,
                kind="atoms", n_atoms=12, sparsity=3, iterations=8, seed=8041,
                minimum_records=8, minimum_ess=6):
    fit_features = np.asarray(fit_features, dtype=float)
    candidate_features = np.asarray(candidate_features, dtype=float)
    residual, coefficient = nuisance_residual(fit_features, fit_nuisance)
    scale = max(float(np.sqrt(np.mean(residual ** 2))), 1e-12)
    train = residual / scale
    candidate = (candidate_features - np.asarray(candidate_nuisance) @ coefficient) / scale
    rng = np.random.default_rng(seed)
    if kind in {"atoms", "random_atoms"}:
        if kind == "atoms":
            learned = learn_dictionary(train, n_atoms, sparsity, iterations, seed)
            dictionary = learned["dictionary"]
        else:
            dictionary = rng.normal(size=(n_atoms, train.shape[1]))
            dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
        codes = omp_encode(candidate, dictionary, sparsity)
        units = signed_units(codes, minimum_records, minimum_ess)
        details = {"kind": kind, "fit_mse": float(np.mean(
            (train - omp_encode(train, dictionary, sparsity) @ dictionary) ** 2)),
            "candidate_mse": float(np.mean((candidate - codes @ dictionary) ** 2))}
        return units, details
    if kind == "raw_clusters":
        # Fit ordinary gradient-space clusters without nuisance projection.
        mean = fit_features.mean(0)
        rms = max(float(np.sqrt(np.mean((fit_features - mean) ** 2))), 1e-12)
        train = (fit_features - mean) / rms
        candidate = (candidate_features - mean) / rms
        centers = train[rng.choice(len(train), n_atoms, replace=False)].copy()
        for _ in range(iterations):
            assignment = ((train[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
            for index in range(n_atoms):
                if np.any(assignment == index):
                    centers[index] = train[assignment == index].mean(0)
        assignment = ((candidate[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
        columns, names, metadata = [], [], []
        for index in range(n_atoms):
            mask = assignment == index
            support = int(mask.sum())
            if support < max(minimum_records, minimum_ess):
                continue
            columns.append(mask.astype(float) / support)
            names.append(f"raw_cluster_{index}")
            metadata.append({"support": support, "ess": float(support)})
        if not columns:
            raise ValueError("No supported gradient cluster")
        return MixtureUnits(np.stack(columns, 1), names, metadata), {"kind": kind}
    raise ValueError("Unknown unit construction")


def categorical_units(categories):
    values = np.asarray(categories)
    columns, names, metadata = [], [], []
    for category in np.unique(values):
        mask = values == category
        columns.append(mask.astype(float) / mask.sum())
        names.append(str(category))
        metadata.append({"support": int(mask.sum()), "ess": float(mask.sum())})
    return MixtureUnits(np.stack(columns, 1), names, metadata)


def best_pair(responses, tolerance=1e-9):
    """Exact two-unit max-min search; prefer genuine singleton-failing pairs."""
    matrix = np.asarray(responses, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Finite objectives x units required")
    best = None
    for left in range(matrix.shape[1]):
        for right in range(left + 1, matrix.shape[1]):
            a, b = matrix[:, left], matrix[:, right]
            fractions = [0.0, 1.0]
            slope = b - a
            for first in range(len(a)):
                for second in range(first + 1, len(a)):
                    denominator = slope[first] - slope[second]
                    if abs(denominator) > 1e-14:
                        fraction = (a[second] - a[first]) / denominator
                        if 0 <= fraction <= 1:
                            fractions.append(float(fraction))
            fraction = max(fractions, key=lambda f: float(np.min(a + f * slope)))
            combined = a + fraction * slope
            margin = float(combined.min())
            singleton = float(max(0.0, a.min(), b.min()))
            gain = margin - singleton
            strict = bool(a.min() <= tolerance and b.min() <= tolerance and margin > tolerance)
            key = (strict, gain, margin)
            if best is None or key > best[0]:
                best = (key, {"left": left, "right": right, "right_fraction": fraction,
                              "left_response": a.tolist(), "right_response": b.tolist(),
                              "combined_response": combined.tolist(), "margin": margin,
                              "gain_over_singletons_and_no_update": gain,
                              "strict_predicted_complementarity": strict})
    return None if best is None else best[1]


def matched_random_distribution(distribution, strata):
    """Preserve each declared stratum's total mass while removing atom choice."""
    probability = np.asarray(distribution, dtype=float)
    strata = np.asarray(strata)
    if probability.shape != strata.shape or np.any(probability < 0) or not np.isclose(probability.sum(), 1):
        raise ValueError("Invalid source distribution or strata")
    matched = np.zeros_like(probability)
    for value in np.unique(strata):
        mask = strata == value
        matched[mask] = probability[mask].sum() / mask.sum()
    return matched
