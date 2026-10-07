"""Controlled ablation ladder A0-A6 (manuscript section 4.10, table 11).

Settings A1-A4 draw from the same training pool, use the same 25-class label
space, contain the same number of images per class and are evaluated on the same
test split. They differ from their neighbour in exactly one factor, so the
contribution of each factor is identified.

Skin-tone balancing samples, within each class, as close to a 50% darker-band
share as the class allows. Natural sampling draws the same per-class totals at
random, preserving the pool's observed mix. Both use identical per-class totals,
so the only difference is skin-tone composition.
"""


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["AblationSetting", "LADDER", "max_feasible_n", "sample_training_set",
           "build_ablation", "COMPARISONS"]


@dataclass(frozen=True)
class AblationSetting:
    name: str
    label: str
    harmonised: bool
    colour_constancy: bool
    sampling: str           # "natural" | "balanced" | "as_released"
    tone_weighted_loss: bool
    matched_size: bool
    question: str


LADDER = {
    "A0": AblationSetting("A0", "Original baseline", False, False, "natural", False, False,
                          "Reference from the first submission"),
    "A1": AblationSetting("A1", "Controlled baseline", True, False, "natural", False, True,
                          "Effect of harmonisation and clean splits"),
    "A2": AblationSetting("A2", "+ Colour constancy", True, True, "natural", False, True,
                          "Effect of colour constancy alone"),
    "A3": AblationSetting("A3", "+ Balancing", True, False, "balanced", False, True,
                          "Effect of skin-tone balancing alone"),
    "A4": AblationSetting("A4", "Full pipeline, matched size", True, True, "balanced", False, True,
                          "Combined effect at fixed size"),
    "A5": AblationSetting("A5", "Full corpus", True, True, "as_released", False, False,
                          "Effect of additional data"),
    "A6": AblationSetting("A6", "Loss re-weighting", True, True, "natural", True, True,
                          "Data-level against loss-level correction"),
}

# (treatment, comparator, what the comparison identifies)
COMPARISONS = [
    ("A1", "A0", "harmonisation and clean splitting"),
    ("A2", "A1", "colour constancy alone"),
    ("A3", "A1", "skin-tone balancing alone"),
    ("A4", "A2", "balancing with colour constancy enabled"),
    ("A5", "A4", "additional training data"),
    ("A6", "A2", "loss-level against data-level correction"),
]


def max_feasible_n(pool: pd.DataFrame, class_col="disease_label", band_col="band",
                   darker_share: float = 0.5) -> dict:
    """Largest per-class count N that a balanced design can actually draw.

    A balanced setting needs `darker_share * N` darker-band images in **every**
    class, so N is capped by the scarcest class. Reporting this cap prevents
    specifying an ablation that the pool cannot supply.
    """
    counts = (pool.groupby([class_col, band_col]).size()
                  .unstack(band_col, fill_value=0)
                  .reindex(columns=["lighter", "darker"], fill_value=0))
    per_class_cap = np.minimum(counts["darker"] / darker_share,
                               counts["lighter"] / (1.0 - darker_share))
    binding = per_class_cap.idxmin() if len(per_class_cap) else None
    return {
        "max_n_per_class": int(np.floor(per_class_cap.min())) if len(per_class_cap) else 0,
        "binding_class": binding,
        "per_class_cap": per_class_cap.astype(int).to_dict(),
        "total_darker_available": int(counts["darker"].sum()),
        "total_lighter_available": int(counts["lighter"].sum()),
        "n_classes": int(len(counts)),
        "note": ("A balanced setting needs darker_share * N darker images in every "
                 "class; the scarcest class sets the cap."),
    }


def sample_training_set(pool: pd.DataFrame, n_per_class: int, sampling: str,
                        class_col="disease_label", band_col="band",
                        darker_share: float = 0.5, seed: int = 2026) -> pd.DataFrame:
    """Draw a training set under one sampling rule.

    Raises if a balanced draw is not satisfiable, rather than silently returning
    an unbalanced set that would invalidate the A3-against-A1 comparison.
    """
    rng = np.random.default_rng(seed)
    if sampling == "as_released":
        return pool.copy()
    if sampling not in ("natural", "balanced"):
        raise ValueError(f"unknown sampling rule {sampling!r}")

    chosen = []
    shortfalls = []
    for cls, grp in pool.groupby(class_col):
        if sampling == "natural":
            take = min(n_per_class, len(grp))
            if take < n_per_class:
                shortfalls.append((cls, len(grp)))
            chosen.append(grp.sample(n=take, random_state=int(rng.integers(1 << 31))))
        else:
            want_dark = int(round(n_per_class * darker_share))
            want_light = n_per_class - want_dark
            dark = grp[grp[band_col] == "darker"]
            light = grp[grp[band_col] == "lighter"]
            if len(dark) < want_dark or len(light) < want_light:
                shortfalls.append((cls, {"darker": len(dark), "lighter": len(light),
                                         "needed": (want_dark, want_light)}))
                continue
            chosen.append(pd.concat([
                dark.sample(n=want_dark, random_state=int(rng.integers(1 << 31))),
                light.sample(n=want_light, random_state=int(rng.integers(1 << 31))),
            ]))

    if shortfalls:
        raise ValueError(
            f"{sampling} sampling at N={n_per_class} is not satisfiable for "
            f"{len(shortfalls)} class(es): {shortfalls[:5]}. "
            f"Lower N (see max_feasible_n) or state the shortfall in the paper."
        )
    return pd.concat(chosen).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_ablation(name: str, pool: pd.DataFrame, n_per_class: int | None = None,
                   seed: int = 2026, **kwargs) -> dict:
    """Materialise one rung of the ladder."""
    if name not in LADDER:
        raise KeyError(f"unknown setting {name!r}; expected one of {sorted(LADDER)}")
    setting = LADDER[name]
    if setting.matched_size and n_per_class is None:
        raise ValueError(f"{name} is a size-matched setting: n_per_class is required")

    data = sample_training_set(pool, n_per_class or 0, setting.sampling, seed=seed, **kwargs)
    resolved = data[data["band"].isin(["lighter", "darker"])] if "band" in data else data
    darker = float((resolved["band"] == "darker").mean() * 100) if len(resolved) else float("nan")
    return {
        "setting": setting,
        "data": data,
        "n_images": len(data),
        "n_per_class": n_per_class,
        "darker_pct_of_resolved": round(darker, 1),
        "colour_constancy": setting.colour_constancy,
        "tone_weighted_loss": setting.tone_weighted_loss,
    }
