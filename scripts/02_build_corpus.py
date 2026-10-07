#!/usr/bin/env python3
"""Part II: build the harmonised, skin-tone-aware corpus.

Ingests the five training sources, harmonises labels onto the 25-class taxonomy,
removes duplicates, assigns group keys and splits, applies colour constancy,
estimates ITA with a confidence, and writes one provenance-tagged report per image.

DDI is never ingested here: it is an external test set only (section 4.8).

    python scripts/02_build_corpus.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derm_audit.pipeline.dedup import assign_group_keys
from derm_audit.pipeline.harmonise import LabelMap
from derm_audit.pipeline.reports import (build_report, coverage_by_provenance,
                                         validate_report, write_schema)
from derm_audit.pipeline.splits import (split_summary, stratified_group_split,
                                        verify_no_leakage)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--manifest", help="CSV of ingested images (overrides the config path)")
    ap.add_argument("--out", default="outputs/corpus")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest or cfg["paths"]["manifest"]
    print(f"[1/6] reading manifest {manifest_path}")
    df = pd.read_csv(manifest_path)
    if "ddi" in {str(s).lower() for s in df.get("source_dataset", [])}:
        raise SystemExit("DDI must not appear in the training manifest: it is external test only")
    print(f"      {len(df)} rows from {df['source_dataset'].nunique()} sources")

    print("[2/6] harmonising labels")
    lm = LabelMap.from_csv(cfg["paths"]["label_map"])
    df = lm.apply(df)
    before = len(df)
    df = df[~df["excluded"]].copy()
    print(f"      kept {len(df)} rows, excluded {before - len(df)} unmappable or out-of-scope")

    print("[3/6] assigning group keys")
    if "dup_cluster_loose" not in df.columns:
        df["dup_cluster_loose"] = range(len(df))
    df = assign_group_keys(df)
    print(f"      {df['group_key'].nunique()} groups "
          f"({(df['group_key_provenance'] == 'patient_or_case_id').mean():.0%} from patient or case ids)")

    print("[4/6] splitting")
    df = stratified_group_split(df, class_col="target_class",
                                fractions=tuple(cfg["split"]["fractions"]),
                                seed=cfg["split"]["seed"])
    leak = verify_no_leakage(df)
    if not leak["ok"]:
        raise SystemExit(f"leakage detected in {leak['n_leaking']} groups")
    print("      no group appears in more than one split")
    summary = split_summary(df, class_col="target_class")
    print(summary.to_string(index=False))

    print("[5/6] writing structured reports")
    reports = []
    for _, row in df.iterrows():
        r = build_report(
            case_id=row.get("case_id", row.name), source_dataset=row["source_dataset"],
            source_label=row["source_label"], disease_label=row["target_class"],
            analysis_band=row.get("band"), ita=row.get("ita_value"),
            ita_confidence=row.get("ita_confidence"),
            fitzpatrick_source=row.get("fitzpatrick_source"),
            anatomical_site=row.get("anatomical_site"), age_band=row.get("age_band"),
            sex=row.get("sex"), lesion_morphology=row.get("lesion_morphology"),
            diagnosis_certainty=row.get("diagnosis_certainty"),
        )
        problems = validate_report(r)
        if problems:
            raise SystemExit(f"provenance violation on {row.get('case_id')}: {problems}")
        reports.append(r)
    print(f"      {len(reports)} reports, all provenance checks passed")

    print("[6/6] writing outputs")
    df.to_csv(out / "corpus.csv", index=False)
    summary.to_csv(out / "table09_splits.csv", index=False)
    write_schema(out / "report_schema.json")
    (out / "supplementary_S4_field_coverage.json").write_text(
        json.dumps(coverage_by_provenance(reports), indent=2))
    (out / "supplementary_S1_label_map.csv").write_text(lm.table.to_csv(index=False))
    print(f"\nwrote {out}/corpus.csv and the supplementary tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
