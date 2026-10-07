# derm-skintone-audit

Code for **"A reproducible data-engineering pipeline for auditing and correcting skin-tone bias in dermatology image-recognition datasets"** (*Engineering Research Express*, manuscript ERX-121504).

The project asks how much of the performance disadvantage to darker skin is already present in the **training data**, before any model is built. It answers that in two parts:

- **Part I** — a metadata-only audit of a public benchmark that measures skin-tone imbalance and projects the macro-F1 gap it implies.
- **Part II** — a reproducible pipeline that harmonises five public sources into a 25-class corpus with skin-tone labels, provenance-tagged reports and captions, then isolates the effect of skin-tone rebalancing with controlled ablations.

## What is here

```
src/derm_audit/
  part1_audit/      audit statistics, learning-curve projection, figures
  pipeline/         colour constancy, ITA skin tone, harmonisation,
                    de-duplication, splitting, reports, captions
  training/         backbone settings and the A0-A6 ablation ladder
  evaluation/       metrics, group bootstrap, Holm adjustment
  multimodal/       CLIP-style image-text probe
scripts/            runnable entry points, 00-07
tests/              77 tests, including checks against the published numbers
configs/            default.yaml and the frozen label map
docs/               implementation guide and GitHub setup
```

## Install

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/derm-skintone-audit.git
cd derm-skintone-audit
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"        # add ",train" for torch and the training stack
```

Part I, the metrics and the bootstrap need no GPU and no images.

## Check it works

```bash
pytest tests -q                              # 77 tests
python scripts/00_demo_end_to_end.py         # synthetic end-to-end run, no data needed
```

The demo writes prediction files in the expected layout and runs the full evaluation chain, so you can confirm the statistics before any training.

## Reproduce the paper

```bash
# Part I: audit and projection
python scripts/01_run_audit.py --metadata data/fitzpatrick17k.csv --out outputs/audit

# Part II: build the corpus (DDI is never ingested here)
python scripts/02_build_corpus.py --config configs/default.yaml

# Check the per-class count a balanced ablation can actually draw, then train
python scripts/03_train_ablations.py --corpus outputs/corpus/corpus.csv --check-n
python scripts/03_train_ablations.py --corpus outputs/corpus/corpus.csv --n-per-class 500

# Evaluate with bootstrap intervals and Holm-adjusted comparisons
python scripts/04_evaluate.py --pred-dir outputs/predictions --out outputs/results
python scripts/05_external_ddi.py --pred-dir outputs/ddi_predictions
```

Full walkthrough: [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md).

## Three things the code enforces

**A balanced ablation cannot ask for more data than exists.** `max_feasible_n` reports the largest per-class count `N` a 50/50 draw can satisfy, and `sample_training_set` refuses an unsatisfiable `N` rather than silently returning an unbalanced set that would invalidate the A3-against-A1 comparison.

**The language model never writes a structured field.** It receives only fields tagged source-provided, computed or expert-validated, and `validate_report` rejects any report in which a structured field carries the language-model tag. Fields absent from the source stay `"not recorded"` rather than being inferred.

**Skin tone is a measured variable, not a label.** Every ITA estimate is stored with its raw angle, method and per-image confidence. A source-provided Fitzpatrick label always takes precedence, disagreements are flagged, and images with neither a source label nor a high-confidence estimate are marked `uncertain` and excluded from the stratified metrics.

## Data

No images or metadata are redistributed here. Obtain each source from its own provider under its own licence: Fitzpatrick17k, SD-198, PAD-UFES-20, Derm7pt, SCIN and (external test only) DDI. `configs/default.yaml` lists where the build expects to find them.

## Citation

See [`CITATION.cff`](CITATION.cff). GitHub renders a ready-made citation from it under "Cite this repository".

## Licence

MIT — see [`LICENSE`](LICENSE).
