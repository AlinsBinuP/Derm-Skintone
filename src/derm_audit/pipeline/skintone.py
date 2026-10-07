"""Skin-tone estimation from healthy skin (manuscript section 4.5).

ITA is treated as an imperfect proxy, not a ground-truth label: every estimate is
stored with the method, the raw angle and a per-image confidence, and a fixed
precedence rule prefers a source-provided Fitzpatrick label where one exists.
"""


from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

BAND_BOUNDARY_DEG = 28.0  # lighter/darker boundary, manuscript table 7

# Chardon categories and their mapping to the two analysis bands (table 7).
CHARDON_BINS = [
    (55.0, np.inf, "very light", "I-II", "lighter"),
    (41.0, 55.0, "light", "II-III", "lighter"),
    (28.0, 41.0, "intermediate", "III", "lighter"),
    (10.0, 28.0, "tan", "IV", "darker"),
    (-30.0, 10.0, "brown", "V", "darker"),
    (-np.inf, -30.0, "dark", "VI", "darker"),
]

__all__ = ["compute_ita", "classify_ita", "ita_confidence", "ITAEstimate",
           "estimate_skin_tone", "resolve_band", "fitzpatrick_to_band"]


def compute_ita(L: float, b: float) -> float:
    """ITA = arctan((L* - 50) / b*) * 180 / pi — manuscript equation (2)."""
    if abs(b) < 1e-8:
        return float("nan")
    return float(np.degrees(np.arctan((float(L) - 50.0) / float(b))))


def classify_ita(ita: float) -> dict:
    """Map an ITA angle to its Chardon category, Fitzpatrick group and band."""
    if ita is None or not np.isfinite(ita):
        return {"chardon": None, "fitzpatrick_group": None, "band": "uncertain"}
    for lo, hi, name, group, band in CHARDON_BINS:
        if lo < ita <= hi or (lo == -np.inf and ita <= hi):
            return {"chardon": name, "fitzpatrick_group": group, "band": band}
    return {"chardon": None, "fitzpatrick_group": None, "band": "uncertain"}


def ita_confidence(n_pixels: int, ita_iqr: float, ita: float,
                   min_pixels: int, max_iqr: float, min_margin: float) -> dict:
    """Per-image confidence from three components (manuscript section 4.5).

    The components are the number of healthy-skin pixels, the dispersion of
    per-pixel ITA within the region, and the distance of the image-level ITA from
    the nearest band boundary. An estimate is high confidence only when all three
    pass their threshold.

    The three thresholds are a configuration choice and must be reported with the
    results; they are not hard-coded here.
    """
    margin = abs(float(ita) - BAND_BOUNDARY_DEG) if np.isfinite(ita) else 0.0
    comp = {
        "pixels": float(np.clip(n_pixels / max(min_pixels, 1), 0.0, 1.0)),
        "dispersion": float(np.clip(1.0 - (ita_iqr / max_iqr), 0.0, 1.0)) if max_iqr > 0 else 0.0,
        "margin": float(np.clip(margin / max(min_margin, 1e-8), 0.0, 1.0)),
    }
    passes = (n_pixels >= min_pixels and ita_iqr <= max_iqr and margin >= min_margin)
    return {"score": float(np.mean(list(comp.values()))),
            "components": comp,
            "level": "high" if passes else "low"}


@dataclass
class ITAEstimate:
    ita: float
    chardon: str | None
    fitzpatrick_group: str | None
    band: str
    confidence: float
    confidence_level: str
    n_skin_pixels: int
    method: str = "ita_cielab_meanLb"

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_skin_tone(lab_pixels, min_pixels: int, max_iqr: float,
                       min_margin: float) -> ITAEstimate:
    """Estimate ITA from segmented healthy-skin pixels in CIE-Lab.

    Parameters
    ----------
    lab_pixels : (N, 3) array of CIE-Lab values for healthy skin only
    min_pixels, max_iqr, min_margin : the three confidence thresholds
    """
    lab = np.asarray(lab_pixels, dtype=float)
    if lab.ndim != 2 or lab.shape[1] != 3:
        raise ValueError("expected an (N, 3) array of CIE-Lab pixels")
    n = int(lab.shape[0])
    if n == 0:
        return ITAEstimate(float("nan"), None, None, "uncertain", 0.0, "low", 0)

    ita = compute_ita(float(lab[:, 0].mean()), float(lab[:, 2].mean()))
    per_pixel = np.degrees(np.arctan(
        (lab[:, 0] - 50.0) / np.where(np.abs(lab[:, 2]) < 1e-8, np.nan, lab[:, 2])
    ))
    per_pixel = per_pixel[np.isfinite(per_pixel)]
    iqr = float(np.subtract(*np.percentile(per_pixel, [75, 25]))) if per_pixel.size else float("inf")

    conf = ita_confidence(n, iqr, ita, min_pixels, max_iqr, min_margin)
    cls = classify_ita(ita)
    return ITAEstimate(ita=ita, chardon=cls["chardon"],
                       fitzpatrick_group=cls["fitzpatrick_group"], band=cls["band"],
                       confidence=conf["score"], confidence_level=conf["level"],
                       n_skin_pixels=n)


def fitzpatrick_to_band(fitzpatrick) -> str:
    """Collapse a Fitzpatrick type (1-6 or I-VI) to an analysis band."""
    if fitzpatrick is None:
        return "uncertain"
    roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}
    try:
        value = int(fitzpatrick)
    except (TypeError, ValueError):
        value = roman.get(str(fitzpatrick).strip().lower(), 0)
    if value in (1, 2, 3):
        return "lighter"
    if value in (4, 5, 6):
        return "darker"
    return "uncertain"


def resolve_band(source_fitzpatrick, ita_estimate: ITAEstimate | None) -> dict:
    """Fixed precedence rule for the final analysis band (manuscript section 4.5).

    A source-provided Fitzpatrick label takes precedence; the ITA band is kept
    alongside it and any disagreement is flagged. Without a source label a
    high-confidence ITA band is used. Images with neither are labelled
    "uncertain": they stay in the corpus but are excluded from the stratified
    fairness metrics and reported separately.
    """
    source_band = fitzpatrick_to_band(source_fitzpatrick)
    ita_band = ita_estimate.band if ita_estimate else "uncertain"
    ita_level = ita_estimate.confidence_level if ita_estimate else "low"

    if source_band != "uncertain":
        return {"band": source_band, "provenance": "source",
                "ita_band": ita_band,
                "disagrees": bool(ita_band != "uncertain" and ita_band != source_band)}
    if ita_level == "high" and ita_band != "uncertain":
        return {"band": ita_band, "provenance": "ita_high_confidence",
                "ita_band": ita_band, "disagrees": False}
    return {"band": "uncertain", "provenance": "none",
            "ita_band": ita_band, "disagrees": False}
