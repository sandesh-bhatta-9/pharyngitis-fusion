# Analysis decisions

The study was planned before modelling (research plan, October 3, 2026), but the hypotheses were not formally frozen
before the first results were seen. All analyses are therefore reported as exploratory. This file records the decisions
that shape the results, so readers can check them against the code.

## Data

- Labels: physician consensus from photographs and symptoms (no microbiology). Majority vote of 3-9 diagnoses.
- Ties (100 patients): counted as bacterial in the main analysis; non-bacterial and excluded as sensitivity analyses.
- Exclusions: patient 666 (no image); 19/23, 218/256, 301/305 (identical photos, different records). n = 735.
- Site: no city column; the EXIF phone model is used as a proxy (Samsung SM-G998B 382, Xiaomi 2201117SG 290, unknown 63).

## Evaluation

- 5-fold cross-validation x 5 repeats, stratified on label x phone; folds in `data/processed/folds.csv`.
- Inner 3-fold CV for hyperparameters, epochs (fusion) and the decision threshold (max balanced accuracy).
- Fine-tuned CNNs: one repeat (5 folds) and a 15% validation split, for compute reasons.
- Primary metric: AUC. Model comparisons: Nadeau-Bengio corrected resampled t-test on fold AUCs (one-sided),
  DeLong per repeat, Holm across H1-H3.

## Hypotheses (gated fusion, DINOv2 features)

- H1: better than the best symptom-only model.
- H2: better than the best image-only model.
- H3: at least as good as the best standard fusion model.
- The "best" comparator in each group is chosen by its own cross-validated AUC, which makes the tests conservative.

## Shortcut analyses (added after the phone imbalance was found in the data audit)

Phone-only, image-size and photo-statistics baselines; phone identification; AUC within phone; models trained and
tested within one phone; cross-phone transfer; large images only; shuffled-image control.

## Addendum (2026-10-05): deconfounded evaluation and shortcut mitigation

Added after the main results were known, and therefore exploratory. Code: `src/mitigate.py`, run by
`scripts/run_mitigation.sh`.

- **Deconfounded AUC:** AUC with each patient weighted by P(y) / P(y | phone group), estimated on the whole cohort, so the
  label is independent of the phone and any phone-only score gets 0.5. Reported per fold and per repeat (pooled), with a
  weighted patient-level bootstrap CI.
- **Mitigation (frozen DINOv2 and ConvNeXt embeddings, same outer folds and nested tuning as `img_lr`):** Shades-of-Gray
  colour constancy, phone as covariate, reweighting by P(y) / P(y | phone), per-phone standardisation, LEACE, LEACE +
  reweighting, colour constancy + LEACE + reweighting, and a one-hidden-layer network with a gradient-reversal phone
  head (lambda in {0, 0.3, 1, 3, 10}, fixed in advance; lambda = 0 is the control).
- **Leakage probe:** linear probe identifying Xiaomi vs Samsung from the transformed test features in every outer fold.
- **Tests:** corrected resampled t-test (two-sided, uncorrected) on fold-level pooled and deconfounded AUC against the
  reference model of the same backbone.

## Addendum 2 (2026-10-06): metric validation, adjusted AUC, end-to-end mitigation, site comparison

Also added after the main results, in response to an internal review; exploratory.

- **Covariate-adjusted AUC** (Janes & Pepe 2008) reported next to the deconfounded AUC for every model.
- **Simulation:** two sites, marker with true AUC 0.60, site signature (SMD 3), prevalence gap 0-0.4, 500 replicates;
  target = AUC of the same model in 20,000 unconfounded patients from the same sites.
- **Resampling sweep:** 200 patients per phone, overall prevalence 0.25, gap 0-0.4, 20 resamples, stratified 5-fold CV.
- **End-to-end mitigation:** fine-tuned ConvNeXt-Tiny (1 repeat) with ERM, GroupDRO (eta = 0.01) or a gradient-reversal
  phone head (lambda ramped to 1); early stopping on validation deconfounded AUC for all three; leakage probe on pooled
  backbone features.
- **Site comparison:** raters per patient, agreement, ties, prevalence under each tie rule, symptoms (chi-square,
  Bonferroni over 20 symptoms).
