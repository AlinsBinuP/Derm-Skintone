"""Splitting must be leak-free and preserve the skin-tone composition."""
import numpy as np
import pandas as pd
import pytest

from derm_audit.pipeline.dedup import (assign_group_keys, build_clusters,
                                       connected_components, hamming)
from derm_audit.pipeline.splits import (split_summary, stratified_group_split,
                                        verify_no_leakage)


@pytest.fixture
def pool():
    rng = np.random.default_rng(0)
    n = 4000
    return pd.DataFrame({
        "disease_label": rng.integers(0, 25, n),
        "band": np.where(rng.random(n) < 0.615, "lighter", "darker"),
        "group_key": [f"g{g}" for g in rng.integers(0, 1400, n)],
    })


def test_no_group_spans_two_splits(pool):
    out = stratified_group_split(pool)
    assert verify_no_leakage(out)["ok"]


def test_split_fractions_are_approximately_70_10_20(pool):
    out = stratified_group_split(pool)
    frac = out["split"].value_counts(normalize=True)
    assert frac["train"] == pytest.approx(0.70, abs=0.05)
    assert frac["validation"] == pytest.approx(0.10, abs=0.04)
    assert frac["test"] == pytest.approx(0.20, abs=0.05)


def test_band_composition_is_held_across_splits(pool):
    out = stratified_group_split(pool)
    summary = split_summary(out)
    shares = summary[summary["split"] != "total"]["darker_pct_of_resolved"]
    assert shares.max() - shares.min() < 3.0


def test_split_is_deterministic_under_a_fixed_seed(pool):
    a = stratified_group_split(pool, seed=2026)["split"].tolist()
    b = stratified_group_split(pool, seed=2026)["split"].tolist()
    assert a == b


def test_group_key_prefers_patient_id_over_cluster():
    df = pd.DataFrame({
        "source_dataset": ["pad_ufes_20", "fitzpatrick17k"],
        "patient_id": ["PAT_1", None],
        "dup_cluster_loose": [7, 9],
    })
    out = assign_group_keys(df)
    assert out.loc[0, "group_key_provenance"] == "patient_or_case_id"
    assert out.loc[1, "group_key_provenance"] == "duplicate_cluster"


def test_near_duplicates_cluster_together():
    labels = build_clusters([0b0000, 0b0001, 0b1111111111111111])
    assert labels[0] == labels[1]          # one bit apart
    assert labels[2] != labels[0]


def test_connected_components_is_transitive():
    labels = connected_components(4, [(0, 1), (1, 2)])
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]


def test_hamming_counts_differing_bits():
    assert hamming(0b1010, 0b1001) == 2
