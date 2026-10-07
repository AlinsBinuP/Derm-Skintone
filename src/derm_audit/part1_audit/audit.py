"""Metadata-only audit of a dermatology benchmark (manuscript section 3).

Runs entirely on the released metadata file; no images are downloaded, so the
audit is reproducible by anyone with the CSV.
"""


from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

LIGHTER = (1, 2, 3)
DARKER = (4, 5, 6)
ADEQUACY_FLOOR = 20

__all__ = [
    "load_metadata",
    "assign_band",
    "representation",
    "category_association",
    "malignant_trend",
    "per_condition_counts",
    "coverage_summary",
    "run_audit",
]


def load_metadata(path, fitzpatrick_col="fitzpatrick_scale", condition_col="label",
                  category_col="three_partition_label", qc_col="qc"):
    """Load and clean the benchmark metadata.

    Drops records with a missing or unresolved Fitzpatrick type and records the
    dataset's own quality control flags as mislabelled (manuscript section 3.1).
    """
    df = pd.read_csv(path)
    missing = {fitzpatrick_col, condition_col}.difference(df.columns)
    if missing:
        raise KeyError(f"metadata is missing required columns: {sorted(missing)}")

    n_raw = len(df)
    fitz = pd.to_numeric(df[fitzpatrick_col], errors="coerce")
    keep = fitz.between(1, 6)
    if qc_col in df.columns:
        flagged = df[qc_col].astype(str).str.strip().str.lower()
        keep &= ~flagged.isin({"mislabelled", "mislabeled", "wrong", "1", "true"})

    out = df.loc[keep].copy()
    out["fitzpatrick"] = fitz.loc[keep].astype(int)
    out["condition"] = out[condition_col].astype(str).str.strip().str.lower()
    if category_col in out.columns:
        out["category"] = out[category_col].astype(str).str.strip().str.lower()
    out.attrs["n_raw"] = n_raw
    out.attrs["n_removed"] = n_raw - len(out)
    return out


def assign_band(fitzpatrick) -> pd.Series:
    """Collapse Fitzpatrick I-VI into the lighter and darker analysis bands."""
    f = pd.Series(fitzpatrick).astype(int)
    return pd.Series(np.where(f.isin(LIGHTER), "lighter", "darker"), index=f.index)


def representation(df) -> pd.DataFrame:
    """Image count and share per Fitzpatrick type (manuscript figure 1(a))."""
    counts = df["fitzpatrick"].value_counts().reindex(range(1, 7), fill_value=0)
    return pd.DataFrame(
        {"fitzpatrick": counts.index,
         "images": counts.to_numpy(),
         "percent": (counts / counts.sum() * 100).round(2).to_numpy()}
    )


def category_association(df) -> dict:
    """Pearson chi-square test of diagnostic category against band, with Cramer's V."""
    if "category" not in df.columns:
        raise KeyError("no 'category' column; pass category_col to load_metadata")
    table = pd.crosstab(df["band"], df["category"])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = int(table.to_numpy().sum())
    min_dim = min(table.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if n and min_dim else float("nan")
    return {"chi2": float(chi2), "dof": int(dof), "p": float(p),
            "cramers_v": cramers_v, "table": table}


def malignant_trend(df, malignant_value="malignant") -> dict:
    """Linear regression of malignant share on Fitzpatrick type (figure 1(b)).

    The regression has only six points, so the slope is reported with a 95%
    confidence interval (manuscript section 3.2).
    """
    share = (
        df.assign(is_mal=df["category"].eq(malignant_value))
        .groupby("fitzpatrick")["is_mal"].mean()
        .reindex(range(1, 7)) * 100.0
    )
    x = share.index.to_numpy(dtype=float)
    y = share.to_numpy(dtype=float)
    ok = ~np.isnan(y)
    res = stats.linregress(x[ok], y[ok])
    dof = int(ok.sum()) - 2
    t = stats.t.ppf(0.975, dof)
    return {
        "per_type_percent": share.round(2).to_dict(),
        "slope": float(res.slope),
        "slope_ci": (float(res.slope - t * res.stderr), float(res.slope + t * res.stderr)),
        "r": float(res.rvalue),
        "p": float(res.pvalue),
    }


def per_condition_counts(df) -> pd.DataFrame:
    """Images per condition in each band, with the lighter-to-darker ratio."""
    wide = (
        df.groupby(["condition", "band"]).size().unstack("band", fill_value=0)
        .reindex(columns=["lighter", "darker"], fill_value=0)
    )
    wide["ratio"] = np.where(wide["darker"] > 0,
                             wide["lighter"] / wide["darker"].replace(0, np.nan),
                             np.nan)
    return wide.reset_index()


def coverage_summary(counts: pd.DataFrame, floor: int = ADEQUACY_FLOOR) -> dict:
    """Adequacy of per-condition sampling and the two medians used downstream.

    Two medians are reported and must not be confused: the median over all
    conditions, and the median over adequately sampled conditions only, which is
    the quantity used in the projection (manuscript section 3.2).
    """
    total = len(counts)
    adequate = {b: int((counts[b] >= floor).sum()) for b in ("lighter", "darker")}
    both = counts[(counts["lighter"] > 0) & (counts["darker"] > 0)]
    return {
        "floor": floor,
        "n_conditions": total,
        "adequate": adequate,
        "adequate_percent": {b: round(adequate[b] / total * 100, 1) for b in adequate},
        "median_all": {b: float(counts[b].median()) for b in ("lighter", "darker")},
        "median_adequate": {
            b: float(counts.loc[counts[b] >= floor, b].median())
            for b in ("lighter", "darker")
        },
        "median_ratio": float(both["ratio"].median()) if len(both) else float("nan"),
        "ratio_iqr": (
            (float(both["ratio"].quantile(0.25)), float(both["ratio"].quantile(0.75)))
            if len(both) else (float("nan"), float("nan"))
        ),
        "percent_more_lighter": (
            round(float((both["lighter"] > both["darker"]).mean() * 100), 1)
            if len(both) else float("nan")
        ),
    }


def run_audit(path, floor: int = ADEQUACY_FLOOR, **load_kwargs) -> dict:
    """Run the full Part I audit and return every reported quantity."""
    df = load_metadata(path, **load_kwargs)
    df["band"] = assign_band(df["fitzpatrick"])
    counts = per_condition_counts(df)
    out = {
        "n_raw": df.attrs["n_raw"],
        "n_analysed": len(df),
        "representation": representation(df),
        "band_counts": df["band"].value_counts().to_dict(),
        "per_condition": counts,
        "coverage": coverage_summary(counts, floor=floor),
    }
    if "category" in df.columns:
        out["association"] = category_association(df)
        out["malignant_trend"] = malignant_trend(df)
    return out
