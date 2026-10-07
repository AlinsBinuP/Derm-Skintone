#!/usr/bin/env python3
"""Run the whole evaluation chain on synthetic data, with no images required.

Use this to check your installation, to see the expected file formats, and to
confirm the statistics before any real training. It writes prediction files in
exactly the layout scripts/04_evaluate.py expects.

    python scripts/00_demo_end_to_end.py --out outputs/demo
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Per-setting darker-stratum penalty, chosen so the resulting gaps resemble the
# ones in the paper. These are synthetic: they demonstrate the machinery only.
SETTING_PENALTY = {"A0": 0.065, "A1": 0.061, "A2": 0.057, "A3": 0.047,
                   "A4": 0.040, "A5": 0.032, "A6": 0.041}
SEEDS = (2026, 2027, 2028, 2029, 2030)


def make_predictions(out_dir: Path, n: int = 1800, n_classes: int = 12,
                     n_groups: int = 1400, darker_share: float = 0.384,
                     seeds=None):
    """Synthesise a test split and one prediction file per setting and seed."""
    rng = np.random.default_rng(2026)
    bands = np.where(rng.random(n) < (1 - darker_share), "lighter", "darker")
    groups = rng.integers(0, n_groups, n)
    y = rng.integers(0, n_classes, n)
    seeds = seeds or SEEDS
    for missing in (n_classes - 3, n_classes - 2, n_classes - 1):  # absent from darker
        y[(bands == "darker") & (y == missing)] = 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.npz"):      # never mix runs of different shapes
        stale.unlink()
    for setting, penalty in SETTING_PENALTY.items():
        for seed in seeds:
            r = np.random.default_rng(hash((setting, seed)) % (2**32))
            probs = r.random((n, n_classes))
            strength = np.where(bands == "lighter", 0.90, 0.90 - penalty * 6.0)
            probs[np.arange(n), y] += strength
            probs /= probs.sum(axis=1, keepdims=True)
            np.savez_compressed(out_dir / f"{setting}_seed{seed}.npz",
                                probs=probs.astype(np.float32), y_true=y,
                                band=bands, group=groups)
    return len(SETTING_PENALTY) * len(seeds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--n-boot", type=int, default=200,
                    help="kept low for speed; the paper uses 2000")
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="kept low for speed; the paper uses 5")
    args = ap.parse_args()

    out = Path(args.out)
    preds = out / "predictions"
    print("[1/3] writing synthetic predictions")
    written = make_predictions(preds, seeds=SEEDS[:args.n_seeds])
    print(f"      {written} files in {preds}")

    print("[2/3] checking the learning-curve projection")
    from derm_audit.part1_audit.learning_curve import fit_curve, project_gap
    cf = fit_curve([20, 50, 100, 200, 500, 1000], [0.37, 0.60, 0.70, 0.72, 0.72, 0.72])
    proj = project_gap(cf, 70, 47, n_boot=200)
    print(f"      a={cf.params[0]:.3f} b={cf.params[1]:.1f} "
          f"gap={proj['gap']:.3f} (paper reports 0.075)")

    print("[3/3] running scripts/04_evaluate.py")
    cmd = [sys.executable, str(ROOT / "scripts" / "04_evaluate.py"),
           "--pred-dir", str(preds), "--out", str(out / "results"),
           "--n-boot", str(args.n_boot)]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        return result.returncode

    print(f"\ndemo complete. Inspect {out}/results/ for the generated tables.")
    print("The numbers are synthetic: they show the machinery, not a finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
