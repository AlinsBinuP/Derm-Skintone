#!/usr/bin/env python3
"""Evaluate saved predictions: metrics, bootstrap intervals and the ablation tests.

Expects one .npz per setting and seed containing `probs`, `y_true`, `band` and
`group`. Produces manuscript tables 15 and 16.

    python scripts/04_evaluate.py --pred-dir outputs/predictions --out outputs/results
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derm_audit.evaluation.bootstrap import (bootstrap_metric, holm_adjust,
                                             paired_bootstrap_test)
from derm_audit.evaluation.metrics import (common_label_space, delta_only,
                                           evaluate)
from derm_audit.training.ablations import COMPARISONS


def load_predictions(pred_dir: Path) -> dict:
    """Group saved prediction files by ablation setting."""
    by_setting = defaultdict(list)
    for path in sorted(pred_dir.glob("*.npz")):
        m = re.match(r"(A\d)_seed(\d+)\.npz$", path.name)
        if not m:
            print(f"  skipping unrecognised file: {path.name}")
            continue
        by_setting[m.group(1)].append(np.load(path, allow_pickle=True))
    return dict(by_setting)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", default="outputs/results")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    preds = load_predictions(Path(args.pred_dir))
    if not preds:
        raise SystemExit(f"no prediction files found in {args.pred_dir}")
    print(f"loaded {sum(len(v) for v in preds.values())} runs "
          f"across {len(preds)} settings: {sorted(preds)}")

    ref = next(iter(preds.values()))[0]
    y_true, bands, groups = ref["y_true"], ref["band"], ref["group"]
    classes = common_label_space(y_true, bands)
    print(f"common label space: {len(classes)} classes present in both strata")

    def delta_fn(idx, run):
        return delta_only(y_true[idx], run["probs"][idx], bands[idx], classes)

    rows = []
    for name, runs in sorted(preds.items()):
        per_seed = [evaluate(y_true, r["probs"], bands, classes=classes) for r in runs]
        agg = {k: (float(np.mean([s[k] for s in per_seed])),
                   float(np.std([s[k] for s in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0)
               for k in ("macro_auroc", "macro_f1", "accuracy", "balanced_accuracy",
                         "auroc_lighter", "auroc_darker", "delta", "worst_group_f1")}
        boot = bootstrap_metric(delta_fn, runs, groups, bands,
                                n_boot=args.n_boot, seed=args.seed)
        rows.append({
            "setting": name, "n_seeds": len(runs),
            **{f"{k}_mean": v[0] for k, v in agg.items()},
            **{f"{k}_sd": v[1] for k, v in agg.items()},
            "delta_ci_low": boot["ci"][0], "delta_ci_high": boot["ci"][1],
        })
        print(f"  {name}: macro-AUROC {agg['macro_auroc'][0]:.3f}+/-{agg['macro_auroc'][1]:.3f}  "
              f"delta {agg['delta'][0]:.3f} [{boot['ci'][0]:.3f}, {boot['ci'][1]:.3f}]")

    table = pd.DataFrame(rows)
    table.to_csv(out / "table15_16_metrics.csv", index=False)

    print("\npaired bootstrap comparisons")
    comparisons = []
    for treat, comp, what in COMPARISONS:
        if treat not in preds or comp not in preds:
            continue
        res = paired_bootstrap_test(delta_fn, preds[comp], preds[treat], groups, bands,
                                    n_boot=args.n_boot, seed=args.seed)
        comparisons.append({"treatment": treat, "comparator": comp, "identifies": what,
                            "change_in_delta": res["diff"],
                            "ci_low": res["ci"][0], "ci_high": res["ci"][1],
                            "p_raw": res["p"]})
    if comparisons:
        adjusted = holm_adjust([c["p_raw"] for c in comparisons])
        for c, p in zip(comparisons, adjusted):
            c["p_holm"] = p
            print(f"  {c['treatment']} vs {c['comparator']}: "
                  f"{c['change_in_delta']:+.3f} "
                  f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]  "
                  f"p={c['p_raw']:.3f} (Holm {p:.3f})  <- {c['identifies']}")
        pd.DataFrame(comparisons).to_csv(out / "ablation_comparisons.csv", index=False)

    (out / "results.json").write_text(json.dumps(
        {"settings": rows, "comparisons": comparisons,
         "n_common_classes": len(classes), "n_boot": args.n_boot}, indent=2, default=str))
    print(f"\nwrote {out}/table15_16_metrics.csv and {out}/ablation_comparisons.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
