"""Fairness and performance metrics (manuscript section 4.11).

Stratum AUROC is computed over the common label space shared by both strata, so
the two bands are always scored on the same classes. Alongside the signed gap the
module reports the absolute gap and worst-group performance, so a model that
becomes unfair in the opposite direction is not rewarded.
"""


from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score)

__all__ = ["common_label_space", "binary_auroc", "macro_auroc_ovr",
           "stratum_metrics", "fairness_metrics", "evaluate", "delta_only"]


def binary_auroc(positive, scores) -> float:
    """AUROC from the rank statistic, with ties handled by average ranks.

    Equivalent to sklearn's roc_auc_score but an order of magnitude cheaper,
    which matters because the bootstrap evaluates it 2000 times per metric.
    """
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(scores, dtype=float))
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def common_label_space(y_true, bands, strata=("lighter", "darker")) -> list:
    """Classes present in every stratum (manuscript section 4.11)."""
    y_true = np.asarray(y_true)
    bands = np.asarray(bands)
    sets = [set(np.unique(y_true[bands == s]).tolist()) for s in strata]
    if not sets:
        return []
    common = set.intersection(*sets) if all(sets) else set()
    return sorted(common)


def macro_auroc_ovr(y_true, probs, classes) -> float:
    """Macro one-vs-rest AUROC restricted to `classes`.

    Classes that are absent or constant in `y_true` are skipped rather than
    silently scored, so the result is always an average over evaluable classes.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)
    scores = []
    for c in classes:
        value = binary_auroc(y_true == c, probs[:, c])
        if np.isfinite(value):
            scores.append(value)
    return float(np.mean(scores)) if scores else float("nan")


def stratum_metrics(y_true, probs, classes) -> dict:
    """AUROC and F1 for one stratum on the common label space."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)
    y_pred = probs.argmax(axis=1)
    keep = np.isin(y_true, classes)
    return {
        "auroc": macro_auroc_ovr(y_true, probs, classes),
        "f1": float(f1_score(y_true[keep], y_pred[keep], labels=list(classes),
                             average="macro", zero_division=0)) if keep.any() else float("nan"),
        "n": int(len(y_true)),
    }


def fairness_metrics(per_stratum: dict) -> dict:
    """Signed gap, absolute gap and worst-group performance."""
    lighter = per_stratum.get("lighter", {})
    darker = per_stratum.get("darker", {})
    a_l, a_d = lighter.get("auroc", np.nan), darker.get("auroc", np.nan)
    delta = float(a_l - a_d)
    aurocs = {k: v["auroc"] for k, v in per_stratum.items() if np.isfinite(v.get("auroc", np.nan))}
    f1s = {k: v["f1"] for k, v in per_stratum.items() if np.isfinite(v.get("f1", np.nan))}
    worst = min(aurocs, key=aurocs.get) if aurocs else None
    return {
        "auroc_lighter": float(a_l),
        "auroc_darker": float(a_d),
        "delta": delta,
        "abs_delta": float(abs(delta)),
        "worst_group": worst,
        "worst_group_auroc": float(aurocs[worst]) if worst else float("nan"),
        "worst_group_f1": float(min(f1s.values())) if f1s else float("nan"),
    }


def evaluate(y_true, probs, bands, classes=None) -> dict:
    """Full metric set for one model on one test split (manuscript table 15)."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)
    bands = np.asarray(bands)
    if classes is None:
        classes = common_label_space(y_true, bands)
    y_pred = probs.argmax(axis=1)

    overall = {
        "macro_auroc": macro_auroc_ovr(y_true, probs, sorted(np.unique(y_true).tolist())),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n_common_classes": len(classes),
    }
    per_stratum = {
        s: stratum_metrics(y_true[bands == s], probs[bands == s], classes)
        for s in ("lighter", "darker") if (bands == s).any()
    }
    return {**overall, **fairness_metrics(per_stratum), "per_stratum": per_stratum}


def delta_only(y_true, probs, bands, classes) -> float:
    """Just the signed fairness gap, for use inside the bootstrap.

    `evaluate` also computes F1, accuracy and balanced accuracy, which the
    bootstrap does not need; skipping them makes 2000 resamples affordable.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)
    bands = np.asarray(bands)
    light = macro_auroc_ovr(y_true[bands == "lighter"], probs[bands == "lighter"], classes)
    dark = macro_auroc_ovr(y_true[bands == "darker"], probs[bands == "darker"], classes)
    return float(light - dark)
