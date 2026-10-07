"""The projection must reproduce the values reported in the paper."""
import numpy as np
import pytest

from derm_audit.part1_audit.learning_curve import (aic, fit_curve, project_gap,
                                                   saturating_exponential)

ANCHORS_N = [20, 50, 100, 200, 500, 1000]
ANCHORS_F = [0.37, 0.60, 0.70, 0.72, 0.72, 0.72]
MEDIAN_LIGHTER, MEDIAN_DARKER = 70, 47


@pytest.fixture(scope="module")
def fitted():
    return fit_curve(ANCHORS_N, ANCHORS_F)


def test_fitted_parameters_match_paper(fitted):
    assert fitted.params[0] == pytest.approx(0.720, abs=5e-4)   # asymptotic macro-F1
    assert fitted.params[1] == pytest.approx(27.8, abs=0.1)     # saturation scale


def test_projection_matches_paper(fitted):
    res = project_gap(fitted, MEDIAN_LIGHTER, MEDIAN_DARKER, n_boot=200)
    assert res["lighter"] == pytest.approx(0.662, abs=2e-3)
    assert res["darker"] == pytest.approx(0.587, abs=2e-3)
    assert res["gap"] == pytest.approx(0.075, abs=2e-3)


@pytest.mark.parametrize("floor,n_lo,n_hi,expected", [
    (10, 68, 44, 0.086),
    (20, 70, 47, 0.075),
    (30, 77, 54, 0.058),
    (50, 91, 69, 0.033),
])
def test_sensitivity_to_adequacy_floor(fitted, floor, n_lo, n_hi, expected):
    assert float(fitted(n_lo) - fitted(n_hi)) == pytest.approx(expected, abs=2e-3)


def test_median_over_all_conditions_is_the_stress_test(fitted):
    assert float(fitted(69) - fitted(24)) == pytest.approx(0.244, abs=3e-3)


def test_exponential_preferred_over_power_law_by_aic():
    exp = fit_curve(ANCHORS_N, ANCHORS_F, model="exponential")
    pwr = fit_curve(ANCHORS_N, ANCHORS_F, model="power")
    assert aic(exp.residuals, 2) < aic(pwr.residuals, 3)
    assert float(pwr(70) - pwr(47)) == pytest.approx(0.049, abs=3e-3)


def test_curve_is_monotone_and_saturating(fitted):
    n = np.arange(1, 2000)
    y = fitted(n)
    assert np.all(np.diff(y) >= -1e-12)
    assert y[-1] == pytest.approx(fitted.params[0], abs=1e-6)


def test_rejects_too_few_anchor_points():
    with pytest.raises(ValueError):
        fit_curve([10, 20], [0.1, 0.2])


def test_equation_one_at_zero_is_zero():
    assert saturating_exponential(0, 0.72, 27.8) == pytest.approx(0.0)
