#!/usr/bin/env python3
"""Assemble supplementary tables S1-S6 from the build and evaluation outputs.

    python scripts/07_make_supplementary.py --corpus outputs/corpus --results outputs/results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derm_audit.pipeline.harmonise import LabelMap

ITEMS = {
    "S1": "Complete source-to-class label map, with every excluded label and its reason",
    "S2": "Full captioning prompt and the checker rule set",
    "S3": "Split sizes by source dataset and by target class",
    "S4": "Structured-report field coverage by provenance tag",
    "S5": "Six-way per-Fitzpatrick-type AUROC and results on the full 25-class space",
    "S6": "Ablation results for the ResNet-50 backbone",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="outputs/corpus")
    ap.add_argument("--results", default="outputs/results")
    ap.add_argument("--label-map", default="configs/label_map.csv")
    ap.add_argument("--out", default="outputs/supplementary")
    args = ap.parse_args()

    corpus_dir, out = Path(args.corpus), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    made = []

    lm = LabelMap.from_csv(args.label_map)
    lm.table.drop(columns=["_key"]).to_csv(out / "S1_label_map.csv", index=False)
    made.append("S1")

    corpus_csv = corpus_dir / "corpus.csv"
    if corpus_csv.exists():
        df = pd.read_csv(corpus_csv)
        class_col = "target_class" if "target_class" in df.columns else "disease_label"
        (df.groupby(["split", "source_dataset", class_col, "band"]).size()
           .rename("images").reset_index()
           .to_csv(out / "S3_splits_by_source_and_class.csv", index=False))
        made.append("S3")

    s4 = corpus_dir / "supplementary_S4_field_coverage.json"
    if s4.exists():
        (out / "S4_field_coverage.json").write_text(s4.read_text())
        made.append("S4")

    print("assembled:", ", ".join(made) or "nothing")
    missing = [k for k in ITEMS if k not in made]
    if missing:
        print("\nstill to produce by hand or from a later stage:")
        for k in missing:
            print(f"  {k}: {ITEMS[k]}")
    (out / "README.md").write_text(
        "# Supplementary material\n\n"
        + "\n".join(f"- **{k}** - {v}" for k, v in ITEMS.items()) + "\n")
    print(f"\nwrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
