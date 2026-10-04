# Pharyngitis fusion

Code for *Does the throat photo add information? Symptom-aware multimodal fusion for bacterial versus
non-bacterial pharyngitis on PGUPharyngitis*. Every model uses the same 5-fold × 5-repeat outer CV with nested
inner CV. Every model saves its out-of-fold predictions, and all tables, tests and figures are computed from
those files.

## 1. Setup (once)

```bash
brew install miniforge          # or download Miniforge from github.com/conda-forge/miniforge
conda init zsh                  # then open a new terminal
cd ~/pharyngitis-fusion
conda env create -f environment.yml
conda activate pharyngitis
python -c "import torch; print(torch.backends.mps.is_available())"   # must print True
```

In VS Code, open this folder, press `Cmd+Shift+P` → **Python: Select Interpreter** → `pharyngitis`.
Run all commands from VS Code's terminal (`Ctrl+\``) in the project root.

## 2. Check the code works (5 minutes, fake data)

```bash
python -m src.make_synthetic --n 300
bash scripts/run_all.sh configs/synthetic.yaml
```

The synthetic numbers mean nothing. This run only checks that every step works on your machine.

## 3. Get the real data

1. Download PGUPharyngitis from Figshare (the link is in the data paper) and accept its licence.
2. Put the Excel file at `data/raw/PGUPharyngitis.xlsx` and the images in `data/raw/images/`.
3. Inspect the columns, then edit `configs/default.yaml` (`columns`, `centor`, `cv.stratify_meta`):

```bash
python -m src.prepare_data --inspect
```

4. Build the clean dataset and read the audit report (`data/processed/audit.md`) before going on:

```bash
python -m src.prepare_data
python -m src.make_folds
```

5. Commit `configs/default.yaml`, `data/processed/folds.csv` and `paper/analysis_plan.md` **before** running E2.
   That commit is your pre-registration.

## 4. Run the experiments

Run `bash scripts/run_all.sh` (main experiments), then `bash scripts/run_extra.sh` (site-shortcut experiments, tie-rule sensitivity, fine-tuning on one repeat, final metrics and paper figures). Together they reproduce every number and figure in the paper, in about 3 hours on an M-series MacBook. Or step by step:

| Step | Command | Output | Plan |
| --- | --- | --- | --- |
| Feature cache | `python -m src.extract_features` | `data/features/*.npz` | §3 |
| Image fine-tuning | `python -m src.finetune_image --model densenet121` | `results/oof/ft_densenet121.csv` | E1 |
| Baselines | `python -m src.baselines` | `sym_*`, `centor`, `img_lr_*`, `late_*`, `early_*`, `stack_*` | E1, E2 |
| Proposed model | `python -m src.fusion --backbone convnext_tiny` | `gated_convnext_tiny` | E2 |
| Ablations | `python -m src.fusion --backbone convnext_tiny --fusion concat` (see `--help`) | `concat_*`, `*__hard`, `*__noAux`, … | E3 |
| Knock-outs | `--drop-symptoms fever cough --tag noFeverCough` on baselines and fusion | `*__noFeverCough` | E5 |
| Shuffled control | `python -m src.fusion --backbone convnext_tiny --shuffle-images --tag shuffled` | `*__shuffled` | E7 |
| Metrics | `python -m src.evaluate` | `results/metrics/summary.md` | all |
| Hypotheses H1–H3 | `python -m src.stats --hypotheses` | `results/metrics/hypotheses.csv` | §5 |
| Label dependence | `python -m src.label_dependence strata/residual/knockouts …` | `label_dependence_*.csv` | E5, E7 |
| Explainability | `python -m src.explain shap/gate/gradcam …` | `figures/` | E6 |
| Figures | `python -m src.figures roc/calibration/dca/ablation/subgroups …` | `figures/` | paper |
| Site shortcut | `python -m src.site_shortcut size/within/cross/lowlevel/largeimg` | `results/metrics/site_*.csv` | shortcut |
| Tie-rule sensitivity | any script with `--config configs/tie_nonbacterial.yaml` or `tie_drop.yaml` | `results/tie_*/` | sensitivity |

Every script explains its options with `--help`. To look at results quickly, add `--repeats 1` to the
training scripts (5 folds instead of 25).

## 5. Methods notes for the paper

- **Splits:** stratified on label × city; `folds.csv` is shared by every model.
- **Nested CV:** hyperparameters, epoch count and the decision threshold are chosen on 3-fold inner CV inside
  each outer training part. Imputers and scalers are fitted on training data only.
- **Fine-tuning exception (E1):** one repeat (5 folds); a stratified 15% validation split replaces inner CV for cost reasons.
- **Threshold:** maximises balanced accuracy on inner out-of-fold predictions.
- **Image features:** frozen backbones, embeddings cached for the centre crop, a horizontal flip (TTA) and
  10 random training augmentations (a new one is sampled each epoch).
- **Gated fusion:** image MLP + symptom MLP → 128-d each; scalar gate g = σ(W[h_img; h_sym]); auxiliary
  image-only and symptom-only heads (λ ∈ {0, 0.3}); modality dropout 0.2; soft-label BCE weighted by class.
- **Statistics:** Nadeau–Bengio corrected resampled t-test on fold AUCs (primary), DeLong per repeat,
  Holm correction across H1–H3, patient-level bootstrap CIs.
- **Reporting:** AUC is reported both pooled per repeat and as the mean of fold AUCs, plus the AUC within each phone model.
- **Site shortcut:** the phone model is read from EXIF (no city column exists); phone-only, image-size and photo-statistics
  baselines, phone identification, within-phone and cross-phone evaluation (`src/site_shortcut.py`).
- **Labels:** majority vote, ties = bacterial; ties = non-bacterial and ties dropped as sensitivity analyses.
- **Exclusions:** patient 666 (no image) and three pairs sharing an identical photo (19/23, 218/256, 301/305).

## Layout

```
configs/      default.yaml (real data), synthetic.yaml (smoke test)
src/          all code; run as python -m src.<script>
scripts/      run_all.sh
data/         raw/ (never edited), processed/, features/   (not in git, except folds.csv)
results/      oof/ (predictions), metrics/, models/
figures/      PNG (300 dpi) + PDF
paper/        analysis_plan.md, drafts
```
