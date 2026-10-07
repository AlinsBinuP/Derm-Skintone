# Implementation guide

How to run this work end to end, in the order the paper reports it. Each stage says what it needs, what it produces, and where it can mislead you.

---

## 0. Before anything: check the installation

```bash
pip install -e ".[dev]"
pytest tests -q                       # 77 tests
python scripts/00_demo_end_to_end.py  # synthetic run, no data needed
```

The demo fabricates predictions and runs the real evaluation chain over them. It proves the statistics work and shows the file layout the later stages expect. Its numbers are synthetic and mean nothing on their own.

---

## 1. Obtain the source data

Nothing is redistributed here. Get each source from its provider, under its own licence:

| Source | Role | Notes |
| --- | --- | --- |
| Fitzpatrick17k | Part I audit + training | metadata CSV is enough for Part I |
| SD-198 | training | no skin-tone metadata |
| PAD-UFES-20 | training | has patient ids |
| Derm7pt | training | has case ids, lesion morphology |
| SCIN | training | has case ids, estimated FST and Monk |
| DDI | **external test only** | never ingest into the corpus |

Point `configs/default.yaml` at where you put them.

> **DDI must stay out of the corpus.** `02_build_corpus.py` aborts if it sees DDI in the training manifest. Referee 2 asked specifically how DDI was separated; the answer has to be "it was never ingested", not "it was removed afterwards".

---

## 2. Part I — audit and projection

```bash
python scripts/01_run_audit.py \
  --metadata data/fitzpatrick17k.csv \
  --anchors data/anchor_points.csv \
  --out outputs/audit
```

This computes representation per Fitzpatrick type, the χ² association between diagnostic category and band with Cramér's V, the malignant-share trend with a 95% slope interval, per-condition coverage, and the two medians. It then fits `f(n) = a[1 − exp(−n/b)]`, projects the gap at each band's median adequately-sampled size, and runs the sensitivity analysis.

### The anchor points matter more than they look

`--anchors` is a CSV with columns `n,macro_f1`. Produce it by training on random class-balanced subsets at 20, 50, 100, 200, 500 and 1000 images per class and scoring each on the common test split.

**Report them to three decimals.** With values rounded to two decimals the curve fits almost exactly (R² = 1.0000, residual spread ≈ 0.0006), and the residual bootstrap then returns an interval far tighter than real run-to-run variation. The script warns you when it detects this. A referee checking your fit will find the same thing.

**Outputs**: `audit_summary.json`, `representation.csv`, `per_condition_counts.csv`.

---

## 3. Part II — build the corpus

```bash
python scripts/02_build_corpus.py --config configs/default.yaml --out outputs/corpus
```

In order: harmonise labels through the frozen map; drop unmappable and out-of-scope labels; assign group keys; split 70/10/20 with joint stratification by class and band; build one provenance-tagged report per image.

### Fill these config values first

`configs/default.yaml` ships with `TO_COMPLETE` markers for choices only you can make. They all appear in the paper, so they must match what you ran:

- `skin_tone.segmentation_model` — the network used to find healthy skin
- `skin_tone.confidence.*` — the three confidence thresholds
- `dedup.phash_hamming_threshold`, `embedding_model`, `cosine_threshold`
- `dedup.loose_hamming_threshold` — used for group keys, not for removal

### Why group keys are not optional

Removing duplicates does not make splits independent: the same patient can appear in several non-identical images. Where a source gives a patient or case id, that id is the group. Where it does not, the group is the near-duplicate cluster at a looser threshold. `verify_no_leakage` then asserts that no group spans two splits, and the build aborts if one does.

### Label map

`configs/label_map.csv` ships as a worked template, not a complete map. Every row records the source dataset, source label, target class, broad category, mapping route (`exact`, `synonym`, `code`, `expert`), ICD-11 and SNOMED codes, and for excluded labels a reason (`out_of_scope`, `too_rare`, `non_specific`, `unmappable`). The full map is supplementary table S1.

**Outputs**: `corpus.csv`, `table09_splits.csv`, `report_schema.json`, `supplementary_S1_label_map.csv`, `supplementary_S4_field_coverage.json`.

---

## 4. Captions

Two per image. The template grammar turns structured fields into sentences and cannot add anything absent from the record, so every image always has a safe caption. The language model is text-only, never sees the image, and receives only fields tagged S, C or E.

Four safeguards, in order: input restriction, instruction constraints, a rule-based checker, and a fallback of one regeneration then the template.

The checker rejects a caption that names a different body site, diagnosis or skin tone, uses a descriptor absent from the record, or contains a forbidden term (treatment, prognosis, age, sex, ethnicity). Negated mentions are allowed through a short negation window, so "with no scale or crust" passes.

```python
from derm_audit.pipeline.captions import generate_caption
result = generate_caption(report, lm_call=my_model)
# result["route"] is "lm", "lm_regenerated" or "template_fallback"
```

Report the counts for the three routes: the paper promises them.

---

## 5. Ablations

**Check what the pool can supply before you train anything:**

```bash
python scripts/03_train_ablations.py --corpus outputs/corpus/corpus.csv --check-n
```

A balanced setting needs `N/2` darker-band images in **every** class, so the scarcest class caps `N`. With a training split of 13 442 lighter and 8398 darker images across 25 classes, `N` cannot exceed roughly 670 per class however the images are distributed. `sample_training_set` raises rather than quietly returning an unbalanced set, because an unbalanced "balanced" arm would invalidate the A3-against-A1 comparison that the whole claim rests on.

```bash
python scripts/03_train_ablations.py --corpus outputs/corpus/corpus.csv --n-per-class 500
```

The ladder:

| Setting | Differs from | Isolates |
| --- | --- | --- |
| A0 | — | the original baseline |
| A1 | A0 | harmonisation and clean splitting |
| A2 | A1 | colour constancy alone |
| A3 | A1 | **skin-tone balancing alone** |
| A4 | A2 | balancing with colour constancy on |
| A5 | A4 | additional data |
| A6 | A2 | a loss-level correction instead |

A1–A4 share pool, label space, per-class count and test split, differing from their neighbour in exactly one factor. **A3 against A1 is the primary estimate.**

Train each setting with all five seeds (2026–2030) using the settings in table 10, and save the test-split probabilities as `outputs/predictions/<setting>_seed<k>.npz` with arrays `probs`, `y_true`, `band`, `group`.

---

## 6. Evaluate

```bash
python scripts/04_evaluate.py --pred-dir outputs/predictions --out outputs/results
python scripts/05_external_ddi.py --pred-dir outputs/ddi_predictions
```

Stratum AUROC is computed over the classes present in **both** strata, so the bands are scored on the same label space. Alongside the signed gap Δ, the absolute gap and worst-group AUROC and F1 are reported, so a model that becomes unfair in the other direction is not rewarded.

Intervals come from resampling the test set 2000 times **at the group level**, stratified by band, with every metric averaged over seeds, so the interval carries both test-set and seed variability. Differences use a paired bootstrap on the same resamples, with Holm adjustment across the six comparisons.

On DDI, the FST III–IV group straddles the binary boundary, so the primary comparison uses I–II against V–VI only, III–IV is reported separately, and two sensitivity analyses assign it to each band in turn.

---

## 7. Image–text experiment

A CLIP-style dual encoder is trained contrastively on image–caption pairs in three variants — class name only (control), template captions, language-model captions — then the image encoder is frozen and evaluated with a linear probe on the same split and metrics.

The three variants differ in wording as well as content, so a gain for language-model text over template text does not by itself show the generated text is better.

---

## Reporting checklist

Before submitting, confirm every one of these appears in the paper:

- [ ] Anchor points at three decimals, with the fitted `a` and `b` and their intervals
- [ ] The adequacy floor and the sensitivity analysis around it
- [ ] Segmentation network, fallback rule, and the three ITA confidence thresholds
- [ ] De-duplication thresholds, embedding model, and how many records each step removed
- [ ] The per-class count `N`, and that it is within the feasible ceiling
- [ ] Split sizes by source and by class (supplementary table S3)
- [ ] Caption route counts: first attempt, after regeneration, template fallback
- [ ] Inter-rater agreement for each judgement, with intervals
- [ ] DDI: cross-source matches removed, images and classes retained
- [ ] Repository URL and archived DOI

---

## Troubleshooting

**`balanced sampling at N=… is not satisfiable`** — working as intended. Run `--check-n` and lower `N`, or report the shortfall explicitly.

**`leakage detected in N groups`** — a group spans two splits. Usually a missing patient id, so the image fell back to cluster grouping. Check `group_key_provenance`.

**`provenance violation on …`** — a field carries a tag it is not allowed. Most often a lesion-morphology value written by the language model; those must come from a source or from expert review, or stay `"not recorded"`.

**Bootstrap is slow** — use `delta_only` rather than full `evaluate` inside resampling, which is what `04_evaluate.py` does. Drop `--n-boot` while developing and restore 2000 for the reported run.

**AUROC is `nan` for a class** — it is absent or constant in that stratum. Expected for rare classes; it is why stratum AUROC uses the common label space.
