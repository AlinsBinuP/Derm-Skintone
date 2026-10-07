#!/usr/bin/env python3
"""Run the controlled ablation ladder A0-A6 across five seeds.

Before training anything it checks that the requested per-class count N can
actually be drawn under balanced sampling, and refuses to proceed otherwise -
an unsatisfiable N silently breaks the A3-against-A1 comparison.

    python scripts/03_train_ablations.py --corpus outputs/corpus/corpus.csv --n-per-class 500
    python scripts/03_train_ablations.py --corpus outputs/corpus/corpus.csv --check-n
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derm_audit.training.ablations import LADDER, build_ablation, max_feasible_n
from derm_audit.training.models import SEEDS, TrainConfig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="outputs/ablations")
    ap.add_argument("--n-per-class", type=int)
    ap.add_argument("--backbone", default="vit_b_16", choices=["vit_b_16", "resnet50"])
    ap.add_argument("--settings", nargs="*", default=list(LADDER))
    ap.add_argument("--check-n", action="store_true",
                    help="report the largest feasible N and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the sampled sets and report composition, do not train")
    args = ap.parse_args()

    df = pd.read_csv(args.corpus)
    train_pool = df[df["split"] == "train"] if "split" in df.columns else df
    class_col = "target_class" if "target_class" in df.columns else "disease_label"
    pool = train_pool.rename(columns={class_col: "disease_label"})

    feas = max_feasible_n(pool)
    print(f"training pool: {len(pool)} images, {feas['n_classes']} classes, "
          f"{feas['total_lighter_available']} lighter / {feas['total_darker_available']} darker")
    print(f"largest per-class N a balanced draw can satisfy: {feas['max_n_per_class']} "
          f"(limited by class {feas['binding_class']!r})")
    if args.check_n:
        return 0

    if args.n_per_class is None:
        raise SystemExit("--n-per-class is required (see --check-n for the ceiling)")
    if args.n_per_class > feas["max_n_per_class"]:
        raise SystemExit(
            f"N={args.n_per_class} exceeds the feasible ceiling of {feas['max_n_per_class']}. "
            "A balanced setting needs N/2 darker images in every class."
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = (TrainConfig.vit_b16() if args.backbone == "vit_b_16"
              else TrainConfig.resnet50())

    manifest = []
    for name in args.settings:
        setting = LADDER[name]
        n = None if not setting.matched_size else args.n_per_class
        try:
            built = build_ablation(name, pool, n_per_class=n)
        except ValueError as exc:
            print(f"  {name}: SKIPPED - {exc}")
            continue
        print(f"  {name} {setting.label:<28} {built['n_images']:>6} images, "
              f"darker {built['darker_pct_of_resolved']:>5}%, "
              f"cc={setting.colour_constancy}, tone_loss={setting.tone_weighted_loss}")
        built["data"].to_csv(out / f"{name}_train.csv", index=False)
        manifest.append({"setting": name, "label": setting.label,
                         "n_images": built["n_images"],
                         "n_per_class": built["n_per_class"],
                         "darker_pct": built["darker_pct_of_resolved"],
                         "colour_constancy": setting.colour_constancy,
                         "tone_weighted_loss": setting.tone_weighted_loss,
                         "seeds": list(SEEDS)})

    (out / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out}/ablation_manifest.json")

    if args.dry_run:
        print("dry run: no training performed")
        return 0

    print("\nTraining requires torch and the image files. For each setting and each of")
    print(f"the five seeds {SEEDS}, train with the settings in table 10 and save the")
    print("test-split probabilities to outputs/predictions/<setting>_seed<k>.npz so that")
    print("scripts/04_evaluate.py can bootstrap them.")
    print(f"backbone={config.backbone} lr={config.learning_rate} batch={config.batch_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
