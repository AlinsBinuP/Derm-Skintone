"""Label harmonisation onto the 25-class taxonomy (manuscript section 4.3).

Five steps: inventory, fixed target classes, coding, expert resolution, freezing.
Every released record keeps its original label, the mapped label and the mapping
route, so the harmonisation is auditable end to end.
"""


from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROUTES = ("exact", "synonym", "code", "expert")
EXCLUSION_REASONS = ("out_of_scope", "too_rare", "non_specific", "unmappable")

__all__ = ["normalise_label", "LabelMap", "inventory_labels"]


def normalise_label(label: str) -> str:
    """Normalise for spelling, case, punctuation and whitespace (step 1)."""
    s = str(label).strip().lower()
    s = re.sub(r"[_/]+", " ", s)
    s = re.sub(r"[^a-z0-9 \-]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


class LabelMap:
    """A frozen, version-controlled source-to-class lookup table.

    The CSV must carry: source_dataset, source_label, target_class,
    broad_category, route, icd11, snomed, excluded, exclusion_reason.
    """

    REQUIRED = ["source_dataset", "source_label", "target_class", "broad_category",
                "route", "excluded"]

    def __init__(self, table: pd.DataFrame):
        missing = set(self.REQUIRED).difference(table.columns)
        if missing:
            raise KeyError(f"label map is missing columns: {sorted(missing)}")
        self.table = table.copy()
        self.table["_key"] = (
            self.table["source_dataset"].astype(str).str.lower().str.strip()
            + "||" + self.table["source_label"].map(normalise_label)
        )
        bad = set(self.table["route"].dropna().unique()).difference(ROUTES)
        if bad:
            raise ValueError(f"unknown mapping routes: {sorted(bad)}")
        self._lookup = self.table.set_index("_key").to_dict("index")

    @classmethod
    def from_csv(cls, path):
        # codes are identifiers, never numbers: keep them as written
        return cls(pd.read_csv(Path(path), dtype={"icd11": str, "snomed": str}))

    def map_label(self, source_dataset: str, source_label: str) -> dict:
        """Map one source label, returning the class, route and provenance."""
        key = f"{str(source_dataset).lower().strip()}||{normalise_label(source_label)}"
        row = self._lookup.get(key)
        if row is None:
            return {"target_class": None, "broad_category": None, "route": None,
                    "excluded": True, "exclusion_reason": "unmappable",
                    "source_label": source_label}
        return {"target_class": None if row["excluded"] else row["target_class"],
                "broad_category": None if row["excluded"] else row.get("broad_category"),
                "route": row.get("route"),
                "excluded": bool(row["excluded"]),
                "exclusion_reason": row.get("exclusion_reason"),
                "icd11": row.get("icd11"), "snomed": row.get("snomed"),
                "source_label": source_label}

    def apply(self, df, dataset_col="source_dataset", label_col="source_label"):
        """Map a whole dataframe, adding the mapped columns."""
        mapped = [self.map_label(ds, lbl)
                  for ds, lbl in zip(df[dataset_col], df[label_col])]
        return pd.concat([df.reset_index(drop=True),
                          pd.DataFrame(mapped).drop(columns=["source_label"])], axis=1)

    def summary(self) -> pd.DataFrame:
        """Manuscript table 6: classes, labels mapped and labels excluded."""
        t = self.table.copy()
        t["excluded"] = t["excluded"].astype(bool)
        kept = t[~t["excluded"]]
        per_category = (
            kept.groupby("broad_category")
                .agg(classes=("target_class", "nunique"),
                     mapped=("target_class", "size"))
                .reset_index()
        )
        dropped = (t[t["excluded"]].groupby("broad_category").size()
                   .rename("excluded").reset_index())
        out = per_category.merge(dropped, on="broad_category", how="left")
        out["excluded"] = out["excluded"].fillna(0).astype(int)
        total = pd.DataFrame([{
            "broad_category": "Total",
            "classes": int(kept["target_class"].nunique()),
            "mapped": int(len(kept)),
            "excluded": int(t["excluded"].sum()),
        }])
        return pd.concat([out, total], ignore_index=True)


def inventory_labels(frames: dict) -> pd.DataFrame:
    """Step 1: collect every distinct normalised label across the sources."""
    rows = []
    for name, df in frames.items():
        col = "source_label" if "source_label" in df.columns else df.columns[0]
        for label in df[col].dropna().unique():
            rows.append({"source_dataset": name, "source_label": label,
                         "normalised": normalise_label(label)})
    return pd.DataFrame(rows).drop_duplicates()
