#!/usr/bin/env python3
"""Part I: audit a benchmark's metadata and project the implied performance gap.

Reproduces manuscript section 3, tables 3 and 4 and figures 1-3. Needs only the
released metadata CSV; no images are downloaded.

    python scripts/01_run_audit.py --metadata data/fitzpatrick17k.csv --out outputs/audit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derm_audit.part1_audit.audit import run_audit
from derm_audit.part1_audit.figures import (figure1_representation, figure2_coverage,
                                            figure3_projection)
from derm_audit.part1_audit.learning_curve import (aic, fit_curve, project_gap,
                                                   sensitivity_table)

# Anchor points: macro-F1 after training on random class-balanced subsets.
# Replace with your own measured values, reported to three decimals.
DEFAULT_ANCHORS_N = [20, 50, 100, 200, 500, 1000]
DEFAULT_ANCHORS_F = [0.37, 0.60, 0.70, 0.72, 0.72, 0.72]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True, help="benchmark metadata CSV")
    ap.add_argument("--out", default="outputs/audit")
    ap.add_argument("--floor", type=int, default=20, help="adequacy floor, images per condition")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--anchors", help="CSV with columns n,macro_f1 (overrides the defaults)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] auditing {args.metadata}")
    res = run_audit(args.metadata, floor=args.floor)
    cov = res["coverage"]
    print(f"      {res['n_analysed']} of {res['n_raw']} records retained")
    print(f"      bands: {res['band_counts']}")
    print(f"      conditions adequately sampled (floor {args.floor}): {cov['adequate']}")

    res["representation"].to_csv(out / "representation.csv", index=False)
    res["per_condition"].to_csv(out / "per_condition_counts.csv", index=False)

    print("[2/3] fitting the learning curve")
    if args.anchors:
        import pandas as pd
        a = pd.read_csv(args.anchors)
        anchors_n, anchors_f = a["n"].tolist(), a["macro_f1"].tolist()
    else:
        anchors_n, anchors_f = DEFAULT_ANCHORS_N, DEFAULT_ANCHORS_F
        print("      WARNING: using the built-in anchor points. Supply --anchors with")
        print("      your own measured values before reporting these numbers.")

    cf = fit_curve(anchors_n, anchors_f)
    n_lo = cov["median_adequate"]["lighter"]
    n_hi = cov["median_adequate"]["darker"]
    proj = project_gap(cf, n_lo, n_hi, n_boot=args.n_boot, seed=args.seed)
    ci = cf.ci()
    print(f"      a = {cf.params[0]:.3f} {tuple(round(v, 3) for v in ci['a'])}, "
          f"b = {cf.params[1]:.1f} {tuple(round(v, 1) for v in ci['b'])}")
    print(f"      projected macro-F1: lighter {proj['lighter']:.3f} (n={n_lo:.0f}), "
          f"darker {proj['darker']:.3f} (n={n_hi:.0f})")
    print(f"      gap = {proj['gap']:.3f}  95% interval "
          f"[{proj['ci'][0]:.3f}, {proj['ci'][1]:.3f}]")

    if float(abs(cf.residuals).max()) < 1e-3:
        print("      NOTE: the anchors lie almost exactly on the fitted curve, so the")
        print("      bootstrap interval is far tighter than real run-to-run variation.")
        print("      Report the anchors at full precision.")

    print("[3/3] sensitivity analysis")
    settings = [
        ("Main analysis: exponential, floor 20, adequately sampled median", "exponential", n_lo, n_hi),
        ("Inverse power law, floor 20", "power", n_lo, n_hi),
        ("Median over all conditions", "exponential",
         cov["median_all"]["lighter"], cov["median_all"]["darker"]),
    ]
    rows = sensitivity_table(anchors_n, anchors_f, settings, n_boot=args.n_boot, seed=args.seed)
    for r in rows:
        print(f"      {r['setting'][:52]:<52} gap={r['gap']:.3f} "
              f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]  AIC={r['aic']}")

    print("      rendering figures")
    if "malignant_trend" in res:
        figure1_representation(res["representation"], res["malignant_trend"],
                               out / "figure1_representation.png")
    figure2_coverage(cov, res["per_condition"], out / "figure2_coverage.png")
    figure3_projection(cf, anchors_n, anchors_f, n_lo, n_hi,
                       out / "figure3_projection.png", boot_ci=proj["ci"])

    summary = {
        "n_raw": res["n_raw"], "n_analysed": res["n_analysed"],
        "band_counts": res["band_counts"], "coverage": cov,
        "association": {k: v for k, v in res.get("association", {}).items() if k != "table"},
        "malignant_trend": res.get("malignant_trend"),
        "curve": {"a": float(cf.params[0]), "b": float(cf.params[1]),
                  "ci": {k: list(map(float, v)) for k, v in ci.items()},
                  "aic_exponential": aic(cf.residuals, 2)},
        "projection": {k: v for k, v in proj.items() if k != "draws"},
        "sensitivity": rows,
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out}/audit_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
