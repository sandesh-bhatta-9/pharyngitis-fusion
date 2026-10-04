"""Compute every metric from the saved out-of-fold predictions (no retraining).

For each model and repeat, predictions from the 5 test folds are pooled; metrics are then summarised as
mean ± SD over repeats. AUC 95% CIs come from a patient-level bootstrap within each repeat, averaged.

Outputs:
  results/metrics/summary.csv, summary.md    main table (E2, E3)
  results/metrics/per_repeat.csv             one row per model x repeat
  results/metrics/fold_auc.csv               one row per model x repeat x fold (used by stats.py)

Usage:
    python -m src.evaluate
    python -m src.evaluate --models sym_lgbm gated_convnext_tiny --n-boot 2000
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             roc_auc_score)

from .common import add_config_arg, list_models, load_config, load_oof
from .stats import fold_aucs


def calibration(y, p):
    """Calibration slope and intercept from a logistic recalibration of y on logit(p)."""
    z = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    m = LogisticRegression(C=np.inf, max_iter=1000).fit(z, y)
    return m.coef_[0, 0], m.intercept_[0]


def net_benefit(y, p, thresholds):
    n = len(y)
    out = []
    for t in thresholds:
        pred = p >= t
        tp, fp = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def bootstrap_auc(y, p, n_boot, rng):
    vals = []
    n = len(y)
    while len(vals) < n_boot:
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n:
            vals.append(roc_auc_score(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])


def metrics(g: pd.DataFrame) -> dict:
    y, p = g["y"].to_numpy(), g["p"].to_numpy()
    yhat = (p >= g["thr"].to_numpy()).astype(int)  # each fold uses its own inner-CV threshold
    tp, tn = np.sum((yhat == 1) & (y == 1)), np.sum((yhat == 0) & (y == 0))
    slope, intercept = calibration(y, p)
    return {
        "auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        "bal_acc": balanced_accuracy_score(y, yhat), "f1_macro": f1_score(y, yhat, average="macro"),
        "sensitivity": tp / max(y.sum(), 1), "specificity": tn / max((y == 0).sum(), 1),
        "accuracy": (yhat == y).mean(), "brier": brier_score_loss(y, p),
        "cal_slope": slope, "cal_intercept": intercept,
    }


SHOW = ["auc", "pr_auc", "bal_acc", "f1_macro", "sensitivity", "specificity", "brier", "cal_slope", "accuracy"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument("--models", nargs="*", help="default: every file in results/oof")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg["seed"])
    out_dir = cfg["paths"]["metrics"]
    out_dir.mkdir(parents=True, exist_ok=True)

    per_rep, folds, summary = [], [], []
    for name in args.models or list_models(cfg):
        oof = load_oof(cfg, name)
        rows = []
        for r, g in oof.groupby("repeat"):
            m = metrics(g)
            m["auc_lo"], m["auc_hi"] = bootstrap_auc(g["y"].to_numpy(), g["p"].to_numpy(), args.n_boot, rng)
            rows.append({"model": name, "repeat": r, **m})
        rep = pd.DataFrame(rows)
        fa = fold_aucs(oof).rename("auc").reset_index().assign(model=name)
        per_rep.append(rep)
        folds.append(fa)
        s = {"model": name, "repeats": len(rep), "auc_fold_mean": fa["auc"].mean(), "auc_fold_sd": fa["auc"].std()}
        for c in SHOW + ["auc_lo", "auc_hi", "cal_intercept"]:
            s[c] = rep[c].mean()
            s[c + "_sd"] = rep[c].std(ddof=1) if len(rep) > 1 else 0.0
        summary.append(s)

    per_rep, folds = pd.concat(per_rep), pd.concat(folds)
    summ = pd.DataFrame(summary).sort_values("auc", ascending=False)
    per_rep.to_csv(out_dir / "per_repeat.csv", index=False)
    folds.to_csv(out_dir / "fold_auc.csv", index=False)
    summ.to_csv(out_dir / "summary.csv", index=False)

    fmt = lambda r, c: f"{r[c]:.3f} ± {r[c + '_sd']:.3f}"  # noqa: E731
    lines = ["| Model | AUC pooled [95% CI] | AUC fold-mean | PR-AUC | Bal. acc. | Macro F1 | Sens. | Spec. | Brier | Cal. slope | Acc. |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in summ.iterrows():
        lines.append(f"| {r['model']} | {fmt(r, 'auc')} [{r['auc_lo']:.3f}, {r['auc_hi']:.3f}] | "
                     f"{r['auc_fold_mean']:.3f} ± {r['auc_fold_sd']:.3f} | "
                     + " | ".join(fmt(r, c) for c in SHOW[1:]) + " |")
    note = ("\nMetrics: pooled out-of-fold predictions per repeat, mean ± SD over repeats. "
            "AUC fold-mean: mean ± SD of the per-fold AUCs (the unit of the corrected t-test). "
            "A pooled AUC well below the fold-mean signals calibration drift between folds. "
            "Thresholds chosen on inner CV (balanced accuracy).\n")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n" + note)
    print("\n".join(lines))
    print(f"\nWrote {out_dir}/summary.csv, summary.md, per_repeat.csv, fold_auc.csv")


if __name__ == "__main__":
    main()
