"""Shades-of-Gray correction must neutralise a colour cast."""
import numpy as np
import pytest

from derm_audit.pipeline.colour import (MINKOWSKI_P, estimate_illuminant,
                                        shades_of_gray)


def test_minkowski_norm_is_six():
    assert MINKOWSKI_P == 6


def test_illuminant_is_unit_norm():
    rng = np.random.default_rng(0)
    e = estimate_illuminant(rng.random((32, 32, 3)))
    assert np.linalg.norm(e) == pytest.approx(1.0)


def test_correction_neutralises_a_cast():
    rng = np.random.default_rng(1)
    img = rng.random((64, 64, 3)) * 0.6
    img[..., 0] *= 1.5            # strong red cast
    corrected = shades_of_gray(img)
    e = estimate_illuminant(corrected)
    assert e == pytest.approx(np.full(3, 1 / np.sqrt(3)), abs=5e-3)


def test_neutral_image_is_left_alone():
    img = np.full((16, 16, 3), 0.5)
    assert shades_of_gray(img) == pytest.approx(img, abs=1e-6)


def test_output_stays_in_range():
    rng = np.random.default_rng(2)
    out = shades_of_gray(rng.random((16, 16, 3)))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_rejects_non_rgb():
    with pytest.raises(ValueError):
        estimate_illuminant(np.zeros((8, 8)))
