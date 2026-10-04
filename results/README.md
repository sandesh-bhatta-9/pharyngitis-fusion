# Results

Everything here is produced by `scripts/run_all.sh` and `scripts/run_extra.sh`. All tables and figures are
computed from the out-of-fold predictions, without retraining.

## `oof/` - out-of-fold predictions, one CSV per model

Columns: `patient_id, repeat, fold, y` (majority-vote label), `y_soft` (share of bacterial votes), `agreement`,
`p` (predicted probability), `thr` (decision threshold chosen on inner CV). Fusion models add `p_img`, `p_sym`
(auxiliary heads) and `g` (gate value).

| Prefix | Model |
| --- | --- |
| `sym_lr`, `sym_lgbm`, `sym_mlp` | Symptoms, age and sex only |
| `centor` | Modified McIsaac score (fever, no cough, age) |
| `phone`, `size`, `lowlevel` | Shortcut baselines: phone model, original image size, 18 photo statistics |
| `img_lr_<backbone>` | Logistic regression on frozen DINOv2 or ConvNeXt embeddings |
| `ft_<model>` | Fine-tuned CNN (one repeat, 5 folds) |
| `late_`, `early_`, `stack_<backbone>` | Standard fusion |
| `gated_`, `concat_`, `film_<backbone>` | Proposed fusion network and its variants |
| `...__<tag>` | Ablations and controls: `hard`, `noAux`, `noModDrop`, `noFeverCough`, `shuffled` |

## `metrics/`

| File | Content |
| --- | --- |
| `summary.md`, `summary.csv` | Main table: AUC (pooled and fold mean), PR-AUC, balanced accuracy, F1, sensitivity, specificity, Brier, calibration |
| `fold_auc.csv`, `per_repeat.csv` | AUC per fold and all metrics per repeat |
| `hypotheses_*.csv`, `comparisons.csv` | Corrected t-tests, DeLong tests and Holm correction |
| `label_dependence_*.csv` | Unanimous vs split votes, AUC within phone, residual test, knock-outs |
| `site_*.csv` | Phone identification, within-phone and cross-phone results, large images, photo statistics |
| `gate_*.csv`, `shap_importance.csv`, `subgroups_*.csv` | Explainability and subgroups |

## `tie_nonbacterial/`, `tie_drop/`

The same analyses with tied votes counted as non-bacterial or excluded.
