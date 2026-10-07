"""Uncertainty and significance (manuscript section 4.11).

Confidence intervals come from resampling the test set 2000 times at the group
level (patient, case or duplicate cluster), stratified by band, so that
correlated images from one patient move together. Every metric is computed for
each seed and averaged over seeds, so the interval reflects both test-set and
seed variability. Differences between configurations use a paired bootstrap on
the same resamples, with Holm adjustment across comparisons.
"""


from __future__ import annotations

import numpy as np

__all__ = ["group_bootstrap_indices", "bootstrap_metric", "paired_bootstrap_test",
           "holm_adjust"]


def group_bootstrap_indices(groups, bands, n_boot: int = 2000, seed: int = 0):
    """Yield resampled row indices, drawing whole groups with replacement.

    Groups are resampled within each band so the band composition of the test set
    is preserved across resamples.
    """
    groups = np.asarray(groups)
    bands = np.asarray(bands)
    rng = np.random.default_rng(seed)

    by_band = {}
    for band in np.unique(bands):
        mask = bands == band
        uniq = np.unique(groups[mask])
        rows = {g: np.flatnonzero(mask & (groups == g)) for g in uniq}
        by_band[band] = (uniq, rows)

    for _ in range(n_boot):
        picked = []
        for uniq, rows in by_band.values():
            if uniq.size == 0:
                continue
            for g in rng.choice(uniq, size=uniq.size, replace=True):
                picked.append(rows[g])
        yield np.concatenate(picked) if picked else np.array([], dtype=int)


def bootstrap_metric(metric_fn, seed_outputs, groups, bands,
                     n_boot: int = 2000, seed: int = 0, level: float = 0.95):
    """Bootstrap a metric that is averaged over training seeds.

    Parameters
    ----------
    metric_fn : callable(idx, seed_output) -> float
    seed_outputs : one entry per training seed (e.g. saved predictions)
    """
    draws = []
    for idx in group_bootstrap_indices(groups, bands, n_boot=n_boot, seed=seed):
        if idx.size == 0:
            continue
        vals = [metric_fn(idx, out) for out in seed_outputs]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            draws.append(float(np.mean(vals)))
    draws = np.asarray(draws, dtype=float)
    if draws.size == 0:
        return {"point": float("nan"), "ci": (float("nan"), float("nan")), "draws": draws}
    lo = (1.0 - level) / 2.0 * 100.0
    point = [metric_fn(np.arange(len(groups)), out) for out in seed_outputs]
    return {
        "point": float(np.mean([p for p in point if np.isfinite(p)])),
        "ci": (float(np.percentile(draws, lo)), float(np.percentile(draws, 100 - lo))),
        "boot_mean": float(draws.mean()),
        "draws": draws,
    }


def paired_bootstrap_test(metric_fn, outputs_a, outputs_b, groups, bands,
                          n_boot: int = 2000, seed: int = 0, level: float = 0.95):
    """Paired bootstrap for the difference between two configurations.

    Both configurations are evaluated on the same resample, so the difference is
    paired. The two-sided p-value is twice the smaller tail proportion
    (manuscript section 4.11).
    """
    diffs = []
    for idx in group_bootstrap_indices(groups, bands, n_boot=n_boot, seed=seed):
        if idx.size == 0:
            continue
        a = np.mean([metric_fn(idx, o) for o in outputs_a])
        b = np.mean([metric_fn(idx, o) for o in outputs_b])
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(float(b - a))
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size == 0:
        return {"diff": float("nan"), "ci": (float("nan"), float("nan")), "p": float("nan")}

    full = np.arange(len(groups))
    point = (np.mean([metric_fn(full, o) for o in outputs_b])
             - np.mean([metric_fn(full, o) for o in outputs_a]))
    lo = (1.0 - level) / 2.0 * 100.0
    tail = min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "diff": float(point),
        "ci": (float(np.percentile(diffs, lo)), float(np.percentile(diffs, 100 - lo))),
        "p": float(min(1.0, 2.0 * tail)),
        "n_boot": int(diffs.size),
    }


def holm_adjust(pvalues) -> list:
    """Holm step-down adjustment, preserving input order."""
    p = np.asarray(list(pvalues), dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()
