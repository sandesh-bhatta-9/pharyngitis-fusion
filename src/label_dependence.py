"""E5 / E7 analyses that ask whether the photo adds information beyond the symptoms.

Subcommands:
  strata      AUC on unanimous vs. split-vote patients, and Spearman correlation with the vote share.
  residual    Do image embeddings predict what the symptom model gets wrong? (permutation test)
  knockouts   Fusion minus symptom-only AUC for each symptom knock-out tag.

Usage:
    python -m src.label_dependence strata --models sym_lgbm img_lr_convnext_tiny gated_convnext_tiny
    python -m src.label_dependence residual --backbone convnext_tiny --sym-model sym_lr --perms 500
    python -m src.label_dependence knockouts --fusion gated_convnext_tiny --symptom sym_lr \
        --tags noFeverCough noCentor
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import TAG_SEP, add_config_arg, load_config, load_data, load_features, load_oof
from .stats import fold_aucs


def auc_or_nan(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def strata(cfg, models):
    rows = []
    for m in models:
        oof = load_oof(cfg, m)
        per = []
        for _, g in oof.groupby("repeat"):
            u = g["agreement"] == 1
            per.append({"auc_all": auc_or_nan(g["y"], g["p"]),
                        "auc_unanimous": auc_or_nan(g.loc[u, "y"], g.loc[u, "p"]),
                        "auc_split": auc_or_nan(g.loc[~u, "y"], g.loc[~u, "p"]),
                        "spearman_vote_share": spearmanr(g["p"], g["y_soft"]).statistic})
        per = pd.DataFrame(per)
        n_u = int((oof.groupby("patient_id")["agreement"].first() == 1).sum())
        rows.append({"model": m, **per.mean().round(4).to_dict(), "n_unanimous": n_u,
                     "n_split": oof["patient_id"].nunique() - n_u})
    return pd.DataFrame(rows)


def residual_test(cfg, backbone, sym_model, perms, seed):
    """Correlation between symptom-model residuals and their CV prediction from image embeddings."""
    df, _ = load_data(cfg)
    F, _, _ = load_features(cfg, backbone, df)
    oof = load_oof(cfg, sym_model)
    oof = oof[oof["repeat"] == 0].set_index("patient_id").loc[df["patient_id"]]
    resid = df["y"].to_numpy() - oof["p"].to_numpy()  # out-of-fold residuals: no leakage from the symptom model
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 9)))
    kf = KFold(5, shuffle=True, random_state=seed)

    def stat(target):
        pred = np.zeros_like(target)
        for tr, te in kf.split(F):
            pred[te] = model.fit(F[tr], target[tr]).predict(F[te])
        return np.corrcoef(pred, target)[0, 1]

    observed = stat(resid)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(resid)) for _ in range(perms)])
    p = (1 + np.sum(null >= observed)) / (1 + perms)
    return pd.DataFrame([{"backbone": backbone, "symptom_model": sym_model, "r_observed": observed,
                          "null_mean": null.mean(), "null_95pct": np.percentile(null, 95), "p_perm": p,
                          "perms": perms}])


def knockouts(cfg, fusion, symptom, tags):
    rows = []
    for tag in [""] + tags:
        sfx = f"{TAG_SEP}{tag}" if tag else ""
        fa, sa = fold_aucs(load_oof(cfg, fusion + sfx)), fold_aucs(load_oof(cfg, symptom + sfx))
        common = fa.index.intersection(sa.index)
        rows.append({"knockout": tag or "none", "auc_fusion": fa[common].mean(), "auc_symptom": sa[common].mean(),
                     "delta_auc": (fa[common] - sa[common]).mean(), "folds": len(common)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("strata")
    s.add_argument("--models", nargs="+", required=True)
    r = sub.add_parser("residual")
    r.add_argument("--backbone", required=True)
    r.add_argument("--sym-model", default="sym_lr")
    r.add_argument("--perms", type=int, default=500)
    k = sub.add_parser("knockouts")
    k.add_argument("--fusion", required=True)
    k.add_argument("--symptom", required=True)
    k.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = cfg["paths"]["metrics"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "strata":
        res = strata(cfg, args.models)
    elif args.cmd == "residual":
        res = residual_test(cfg, args.backbone, args.sym_model, args.perms, cfg["seed"])
    else:
        res = knockouts(cfg, args.fusion, args.symptom, args.tags)
    path = out_dir / f"label_dependence_{args.cmd}.csv"
    res.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.round(4).to_string(index=False))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
