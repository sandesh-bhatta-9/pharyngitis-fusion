# Pre-registered analysis plan

Fill this in and commit it **before** running E2 on the real data. Record the commit hash in the Methods.

- Date frozen:
- Commit:

## Data
- Labelling protocol (what each physician saw):
- Tie rule:
- Exclusions:

## Primary outcome and metric
- Outcome: majority-vote bacterial vs. non-bacterial (expert consensus, not culture).
- Primary metric: AUC; unit of analysis for tests = fold AUC over 5 × 5 repeated CV.

## Hypotheses (one-sided, α = 0.05, Holm across H1–H3)
- H1: proposed fusion > best symptom-only model (sym_lr, sym_lgbm, sym_mlp, centor)
- H2: proposed fusion > best image-only model (img_lr_*, ft_*)
- H3: proposed fusion ≥ best standard fusion (late_*, early_*, stack_*), reported with CI
- Proposed model name:
- Minimum effect of interest: ΔAUC ≥ 0.02

## Secondary and exploratory
- Secondary: balanced accuracy, macro F1, Brier, calibration slope, net benefit (thresholds 0.1–0.5).
- Exploratory (labelled as such): E3 ablations, E4 subgroups, E5 knock-outs and strata, E6, E7.

## Pivot rule
- If symptom-only fold-mean AUC ≥ 0.90, the paper leads with the label-dependence analysis (E5/E7).
