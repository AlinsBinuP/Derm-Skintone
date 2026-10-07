#!/usr/bin/env python3
"""Image-text experiment: contrastive pre-training then a linear probe.

Compares three text sources - class name only (control), template captions and
language-model captions - on the same split and metrics as the classifiers
(manuscript section 4.12, table 19).

    python scripts/06_image_text.py --features outputs/features --out outputs/results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derm_audit.evaluation.metrics import common_label_space, evaluate
from derm_audit.multimodal.clip_probe import TEXT_VARIANTS, linear_probe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True,
                    help="directory of <variant>_<split>.npz with `features` and `labels`")
    ap.add_argument("--out", default="outputs/results")
    ap.add_argument("--seeds", type=int, nargs="*", default=[2026, 2027, 2028, 2029, 2030])
    args = ap.parse_args()

    feat_dir, out = Path(args.features), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in TEXT_VARIANTS:
        train_f = feat_dir / f"{variant}_train.npz"
        test_f = feat_dir / f"{variant}_test.npz"
        if not train_f.exists() or not test_f.exists():
            print(f"  {variant}: missing feature files, skipping")
            continue

        tr, te = np.load(train_f), np.load(test_f)
        classes = common_label_space(te["labels"], te["band"])
        per_seed = []
        for seed in args.seeds:
            probs, _ = linear_probe(tr["features"], tr["labels"], te["features"], seed=seed)
            per_seed.append(evaluate(te["labels"], probs, te["band"], classes=classes))

        agg = {k: (float(np.mean([s[k] for s in per_seed])),
                   float(np.std([s[k] for s in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0)
               for k in ("macro_auroc", "macro_f1", "auroc_darker", "delta")}
        rows.append({"variant": variant, "n_seeds": len(per_seed),
                     **{f"{k}_mean": v[0] for k, v in agg.items()},
                     **{f"{k}_sd": v[1] for k, v in agg.items()}})
        print(f"  {variant:<16} macro-AUROC {agg['macro_auroc'][0]:.3f} "
              f"darker {agg['auroc_darker'][0]:.3f}  delta {agg['delta'][0]:.3f}")

    (out / "table19_image_text.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}/table19_image_text.json")
    print("The variants differ in wording as well as content, so a gain for the")
    print("language-model text does not on its own show the generated text is better.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
