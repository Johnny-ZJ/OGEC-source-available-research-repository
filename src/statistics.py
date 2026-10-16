from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def stratified_bootstrap_accuracy_ci(
    y_true,
    y_pred,
    n_boot: int = 5000,
    confidence: float = 0.95,
    random_state: int = 2026,
):
    """Class-stratified bootstrap CI for pooled accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(random_state)

    classes = np.unique(y_true)
    class_indices = {c: np.flatnonzero(y_true == c) for c in classes}
    samples = []

    for _ in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(class_indices[c], size=len(class_indices[c]), replace=True)
                for c in classes
            ]
        )
        samples.append(np.mean(y_true[idx] == y_pred[idx]))

    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def paired_wilcoxon(x, y):
    """Two-sided paired Wilcoxon signed-rank test."""
    result = wilcoxon(x, y, alternative="two-sided")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def holm_adjust(p_values):
    """Holm step-down multiplicity correction."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)

    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(running, 1.0)

    adjusted = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted
