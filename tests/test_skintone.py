"""ITA binning must match manuscript table 7, and precedence must prefer source labels."""
import numpy as np
import pytest

from derm_audit.pipeline.skintone import (BAND_BOUNDARY_DEG, classify_ita, compute_ita,
                                          estimate_skin_tone, fitzpatrick_to_band,
                                          resolve_band)


@pytest.mark.parametrize("ita,chardon,band", [
    (70, "very light", "lighter"),
    (55, "light", "lighter"),
    (45, "light", "lighter"),
    (41, "intermediate", "lighter"),
    (30, "intermediate", "lighter"),
    (28, "tan", "darker"),
    (15, "tan", "darker"),
    (10, "brown", "darker"),
    (-20, "brown", "darker"),
    (-30, "dark", "darker"),
    (-50, "dark", "darker"),
])
def test_table7_bins(ita, chardon, band):
    out = classify_ita(ita)
    assert out["chardon"] == chardon
    assert out["band"] == band


def test_band_boundary_is_28_degrees():
    assert BAND_BOUNDARY_DEG == 28.0
    assert classify_ita(28.001)["band"] == "lighter"
    assert classify_ita(28.0)["band"] == "darker"


def test_ita_equation():
    # arctan((65 - 50) / 12) in degrees
    assert compute_ita(65, 12) == pytest.approx(51.34, abs=0.01)
    assert np.isnan(compute_ita(65, 0))


def test_nan_ita_is_uncertain_not_guessed():
    assert classify_ita(float("nan"))["band"] == "uncertain"
    assert classify_ita(None)["band"] == "uncertain"


def test_empty_pixels_give_uncertain():
    est = estimate_skin_tone(np.empty((0, 3)), 100, 20, 5)
    assert est.band == "uncertain"
    assert est.confidence_level == "low"


def test_source_label_takes_precedence_and_flags_disagreement():
    est = estimate_skin_tone(np.tile([70.0, 5.0, 15.0], (500, 1)), 100, 20, 5)
    assert est.band == "lighter"
    out = resolve_band("IV", est)
    assert out["band"] == "darker"            # source wins
    assert out["provenance"] == "source"
    assert out["disagrees"] is True


def test_low_confidence_without_source_is_uncertain():
    est = estimate_skin_tone(np.tile([70.0, 5.0, 15.0], (5, 1)), 1000, 20, 5)
    assert est.confidence_level == "low"
    assert resolve_band(None, est)["band"] == "uncertain"


@pytest.mark.parametrize("value,expected", [
    (1, "lighter"), (3, "lighter"), (4, "darker"), (6, "darker"),
    ("III", "lighter"), ("V", "darker"), (None, "uncertain"), ("", "uncertain"),
])
def test_fitzpatrick_to_band(value, expected):
    assert fitzpatrick_to_band(value) == expected
