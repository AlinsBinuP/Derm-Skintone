#!/usr/bin/env python3
"""External test on DDI, with the III-IV ambiguity handled explicitly.

DDI reports three skin-tone groups: FST I-II, III-IV and V-VI. The III-IV group
crosses the study's binary boundary, so the primary comparison uses I-II against
V-VI only and reports III-IV separately. Two sensitivity analyses assign III-IV
wholly to each band (manuscript section 4.8, table 18).

    python scripts/05_external_ddi.py --pred-dir outputs/ddi_predictions
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derm_audit.evaluation.metrics import macro_auroc_ovr

GROUPS = ("I-II", "III-IV", "V-VI")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", default="outputs/results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    by_setting = defaultdict(list)
    for path in sorted(Path(args.pred_dir).glob("*.npz")):
        m = re.match(r"(A\d)_seed(\d+)\.npz$", path.name)
        if m:
            by_setting[m.group(1)].append(np.load(path, allow_pickle=True))
    if not by_setting:
        raise SystemExit(f"no DDI prediction files in {args.pred_dir}")

    rows = []
    for name, runs in sorted(by_setting.items()):
        ref = runs[0]
        y, fst = ref["y_true"], ref["fst_group"]
        classes = sorted(set(np.unique(y[fst == "I-II"])) & set(np.unique(y[fst == "V-VI"])))

        per_group = {}
        for g in GROUPS:
            mask = fst == g
            if not mask.any():
                continue
            vals = [macro_auroc_ovr(y[mask], r["probs"][mask], classes) for r in runs]
            per_group[g] = (float(np.mean(vals)),
                            float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

        def gap(lighter_groups, darker_groups):
            lm = np.isin(fst, lighter_groups)
            dm = np.isin(fst, darker_groups)
            if not lm.any() or not dm.any():
                return float("nan")
            a = np.mean([macro_auroc_ovr(y[lm], r["probs"][lm], classes) for r in runs])
            b = np.mean([macro_auroc_ovr(y[dm], r["probs"][dm], classes) for r in runs])
            return float(a - b)

        row = {
            "setting": name, "n_classes_scored": len(classes),
            **{f"auroc_{g}": per_group.get(g, (float('nan'),))[0] for g in GROUPS},
            "delta_primary": gap(["I-II"], ["V-VI"]),
            "delta_III_IV_as_lighter": gap(["I-II", "III-IV"], ["V-VI"]),
            "delta_III_IV_as_darker": gap(["I-II"], ["III-IV", "V-VI"]),
        }
        rows.append(row)
        print(f"  {name}: primary delta (I-II vs V-VI) {row['delta_primary']:.3f} | "
              f"III-IV as lighter {row['delta_III_IV_as_lighter']:.3f} | "
              f"as darker {row['delta_III_IV_as_darker']:.3f}")

    (out / "table18_ddi_external.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {out}/table18_ddi_external.json")
    print("DDI is small: report these intervals as wide and treat the result as")
    print("confirming direction rather than estimating the population gap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
