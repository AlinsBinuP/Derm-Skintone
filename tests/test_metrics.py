"""Metrics must use a common label space and never reward reversed unfairness."""
import numpy as np
import pytest

from derm_audit.evaluation.bootstrap import holm_adjust, paired_bootstrap_test
from derm_audit.evaluation.metrics import common_label_space, evaluate


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(1)
    n, c = 1200, 25
    y = rng.integers(0, c, n)
    bands = np.where(rng.random(n) < 0.62, "lighter", "darker")
    for missing in (22, 23, 24):                 # absent from the darker stratum
        y[(bands == "darker") & (y == missing)] = 0
    probs = rng.random((n, c))
    probs[np.arange(n), y] += np.where(bands == "lighter", 0.55, 0.35)
    probs /= probs.sum(axis=1, keepdims=True)
    groups = rng.integers(0, 300, n)
    return y, probs, bands, groups


def test_common_label_space_excludes_classes_missing_from_a_stratum(synthetic):
    y, _, bands, _ = synthetic
    classes = common_label_space(y, bands)
    assert len(classes) == 22
    assert 22 not in classes and 23 not in classes


def test_detects_darker_skin_disadvantage(synthetic):
    y, probs, bands, _ = synthetic
    m = evaluate(y, probs, bands)
    assert m["delta"] > 0
    assert m["worst_group"] == "darker"
    assert m["worst_group_auroc"] == pytest.approx(m["auroc_darker"])


def test_absolute_gap_does_not_reward_reversed_unfairness(synthetic):
    y, probs, bands, _ = synthetic
    flipped = probs.copy()
    flipped[bands == "lighter"] = np.roll(flipped[bands == "lighter"], 1, axis=1)
    m = evaluate(y, flipped, bands)
    assert m["delta"] < 0                   # darker now ahead
    assert m["abs_delta"] > 0               # but the absolute gap still penalises it


def test_all_core_metrics_are_reported(synthetic):
    y, probs, bands, _ = synthetic
    m = evaluate(y, probs, bands)
    for key in ("macro_auroc", "macro_f1", "accuracy", "balanced_accuracy",
                "auroc_lighter", "auroc_darker", "delta", "abs_delta",
                "worst_group_auroc", "worst_group_f1"):
        assert np.isfinite(m[key]), key


def test_paired_bootstrap_detects_a_real_improvement(synthetic):
    y, probs, bands, groups = synthetic
    classes = common_label_space(y, bands)
    better = probs.copy()
    better[bands == "darker", :] = 0.5 * better[bands == "darker", :]
    better[np.arange(len(y)), y] += np.where(bands == "darker", 0.40, 0.0)
    better /= better.sum(axis=1, keepdims=True)
    def fn(idx, p):
        return evaluate(y[idx], p[idx], bands[idx], classes=classes)["delta"]

    res = paired_bootstrap_test(fn, [probs], [better], groups, bands, n_boot=150)
    assert res["diff"] < 0                  # the gap shrank
    assert 0.0 <= res["p"] <= 1.0


def test_holm_is_monotone_and_never_below_raw():
    raw = [0.006, 0.271, 0.284, 0.003, 0.028]
    adj = holm_adjust(raw)
    assert all(a >= r for a, r in zip(adj, raw))
    assert all(0.0 <= a <= 1.0 for a in adj)
    assert adj[3] == pytest.approx(0.015, abs=1e-3)   # smallest p, times 5
