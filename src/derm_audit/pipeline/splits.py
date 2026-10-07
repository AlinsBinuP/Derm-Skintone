"""Grouped, stratified splitting and leakage checks (manuscript section 4.8)."""


from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

__all__ = ["stratified_group_split", "verify_no_leakage", "split_summary"]


def stratified_group_split(df, class_col="disease_label", band_col="band",
                           group_col="group_key", fractions=(0.70, 0.10, 0.20),
                           seed: int = 2026):
    """Assign groups to train/validation/test, stratified jointly by class and band.

    Stratification uses the combined class-and-band key so that both the class
    distribution and the lighter-darker composition are preserved in every split.
    No group appears in more than one split.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    train_f, val_f, test_f = fractions

    strata = (df[class_col].astype(str) + "||" + df[band_col].astype(str)).to_numpy()
    groups = df[group_col].to_numpy()

    n_splits = max(2, int(round(1.0 / test_f)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(sgkf.split(df, strata, groups))

    rest = df.iloc[rest_idx]
    inner_frac = val_f / (train_f + val_f)
    n_inner = max(2, int(round(1.0 / inner_frac)))
    sgkf2 = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed + 1)
    tr_rel, va_rel = next(sgkf2.split(rest, strata[rest_idx], groups[rest_idx]))

    assignment = pd.Series("train", index=df.index, dtype=object)
    assignment.iloc[rest_idx[va_rel]] = "validation"
    assignment.iloc[test_idx] = "test"
    out = df.copy()
    out["split"] = assignment.to_numpy()
    return out


def verify_no_leakage(df, group_col="group_key", split_col="split") -> dict:
    """Confirm every group falls in exactly one split."""
    per_group = df.groupby(group_col)[split_col].nunique()
    offenders = per_group[per_group > 1]
    return {"ok": bool(offenders.empty),
            "n_groups": int(per_group.size),
            "leaking_groups": offenders.index.tolist()[:50],
            "n_leaking": int(offenders.size)}


def split_summary(df, class_col="disease_label", band_col="band",
                  group_col="group_key", split_col="split") -> pd.DataFrame:
    """Manuscript table 9."""
    rows = []
    for split in ("train", "validation", "test"):
        sub = df[df[split_col] == split]
        if sub.empty:
            continue
        counts = sub[band_col].value_counts()
        lighter, darker = int(counts.get("lighter", 0)), int(counts.get("darker", 0))
        resolved = lighter + darker
        rows.append({
            "split": split, "images": len(sub),
            "groups": int(sub[group_col].nunique()),
            "lighter": lighter, "darker": darker,
            "uncertain": int(counts.get("uncertain", 0)),
            "classes": int(sub[class_col].nunique()),
            "darker_pct_of_resolved": round(darker / resolved * 100, 1) if resolved else float("nan"),
        })
    total = pd.DataFrame(rows).drop(columns=["split", "darker_pct_of_resolved"]).sum()
    res = int(total["lighter"] + total["darker"])
    rows.append({"split": "total", **{k: int(v) for k, v in total.items()},
                 "classes": int(df[class_col].nunique()),
                 "darker_pct_of_resolved": round(int(total["darker"]) / res * 100, 1) if res else float("nan")})
    return pd.DataFrame(rows)
