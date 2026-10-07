"""The ablation ladder must refuse a sample size the pool cannot supply."""
import pandas as pd
import pytest

from derm_audit.training.ablations import (COMPARISONS, LADDER, build_ablation,
                                           max_feasible_n, sample_training_set)


@pytest.fixture
def pool():
    """A pool shaped like manuscript table 9's training split."""
    rows = []
    for cls in range(25):
        rows += [{"disease_label": cls, "band": "lighter"}] * (13442 // 25)
        rows += [{"disease_label": cls, "band": "darker"}] * (8398 // 25)
    return pd.DataFrame(rows)


def test_feasible_n_is_bounded_by_the_scarcest_class(pool):
    feas = max_feasible_n(pool)
    assert feas["max_n_per_class"] == 670
    assert feas["binding_class"] is not None


def test_balanced_sampling_refuses_an_unsatisfiable_n(pool):
    with pytest.raises(ValueError, match="not satisfiable"):
        sample_training_set(pool, 1000, "balanced")


def test_balanced_sampling_reaches_fifty_percent(pool):
    built = build_ablation("A3", pool, n_per_class=600)
    assert built["darker_pct_of_resolved"] == pytest.approx(50.0, abs=0.5)
    assert built["n_images"] == 600 * 25


def test_natural_sampling_preserves_the_observed_mix(pool):
    built = build_ablation("A1", pool, n_per_class=600)
    assert built["darker_pct_of_resolved"] == pytest.approx(38.4, abs=1.5)


def test_matched_settings_have_identical_size(pool):
    sizes = {n: build_ablation(n, pool, n_per_class=600)["n_images"]
             for n in ("A1", "A2", "A3", "A4", "A6")}
    assert len(set(sizes.values())) == 1


def test_a3_and_a1_differ_only_in_skin_tone_composition(pool):
    a1, a3 = LADDER["A1"], LADDER["A3"]
    assert a1.colour_constancy == a3.colour_constancy
    assert a1.harmonised == a3.harmonised
    assert a1.tone_weighted_loss == a3.tone_weighted_loss
    assert a1.sampling != a3.sampling


def test_size_matched_setting_requires_n(pool):
    with pytest.raises(ValueError, match="n_per_class is required"):
        build_ablation("A1", pool)


def test_primary_comparison_is_a3_against_a1():
    assert ("A3", "A1", "skin-tone balancing alone") in COMPARISONS
