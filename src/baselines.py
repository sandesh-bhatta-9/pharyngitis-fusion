"""Baselines on the shared folds (groups A-C of the plan), with nested inner CV.

Models (name in results/oof/):
  sym_lr, sym_lgbm, sym_mlp        symptoms + age + gender only
  centor                           McIsaac/Centor score from the mapped items (no fitting)
  img_lr_<bb>                      logistic regression on frozen image embeddings (+ flip TTA)
  late_<bb>                        mean of sym_lr and img_lr_<bb> probabilities
  early_<bb>                       one logistic regression on [tabular, PCA(image)]
  stack_<bb>                       logistic regression on inner out-of-fold [sym_lr, img_lr] logits

Hyperparameters are chosen by inner CV (roc_auc); the decision threshold maximises balanced accuracy
on inner out-of-fold predictions. Nothing is fitted on the outer test fold.

Usage:
    python -m src.baselines
    python -m src.baselines --models sym_lr sym_lgbm --drop-symptoms fever cough --tag noFeverCough
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import logit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import (TAG_SEP, add_config_arg, backbone_key, choose_threshold, inner_cv, load_config, load_data,
                     load_features, oof_frame, outer_splits, save_oof, set_seed, tab_array, tab_transformer)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*did not converge.*")

ALL = ["sym_lr", "sym_lgbm", "sym_mlp", "centor", "img_lr", "late", "early", "stack"]


def lr(C=1.0):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=5000)


def symptom_models(n_bin):
    tt = tab_transformer(n_bin)
    return {
        "sym_lr": (Pipeline([("prep", tt), ("clf", lr())]), {"clf__C": [0.01, 0.1, 1, 10]}),
        "sym_lgbm": (Pipeline([("prep", tt), ("clf", LGBMClassifier(
            learning_rate=0.05, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            class_weight="balanced", n_jobs=1, verbose=-1, random_state=0))]),
            {"clf__num_leaves": [4, 8], "clf__n_estimators": [100, 300]}),
        "sym_mlp": (Pipeline([("prep", tt), ("clf", MLPClassifier((64, 64), max_iter=1000, random_state=0))]),
                    {"clf__alpha": [1e-3, 1e-1, 1.0]}),
    }


def fit_select(est, grid, X, y, cv):
    """Grid search on inner CV; return the refitted best model and its inner out-of-fold probabilities."""
    gs = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1).fit(X, y)
    inner = cross_val_predict(clone(gs.best_estimator_), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return gs.best_estimator_, inner


def proba(model, X):
    return model.predict_proba(X)[:, 1]


def centor_score(df, cfg):
    items = cfg.get("centor", {}) or {}
    score, used = np.zeros(len(df)), []
    for item in ("fever", "exudate", "nodes"):
        col = items.get(item)
        if col and col in df:
            score += df[col].fillna(0).to_numpy()
            used.append(item)
    if items.get("cough") and items["cough"] in df:
        score += 1 - df[items["cough"]].fillna(1).to_numpy()
        used.append("no cough")
    age = df["age"].to_numpy()
    score += ((age >= 3) & (age <= 14)).astype(float) - (age >= 45).astype(float)
    return score, used


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument("--models", nargs="*", default=ALL, choices=ALL)
    ap.add_argument("--backbones", nargs="*", help="default: config features.backbones")
    ap.add_argument("--drop-symptoms", nargs="*", default=[], help="E5 knock-out: remove these inputs")
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. noFeverCough")
    ap.add_argument("--repeats", type=int, help="run only the first N repeats (quick checks)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    df, cols = load_data(cfg)
    y = df["y"].to_numpy()
    X_tab, n_bin, _ = tab_array(df, cols, args.drop_symptoms)
    suffix = f"{TAG_SEP}{args.tag}" if args.tag else ""
    backbones = [backbone_key(b) for b in (args.backbones or cfg["features"]["backbones"])]
    need_img = any(m in args.models for m in ("img_lr", "late", "early", "stack"))
    feats = {bb: load_features(cfg, bb, df) for bb in backbones} if need_img else {}

    c_score, c_used = centor_score(df, cfg)
    if "centor" in args.models and not c_used:
        print("centor: no items mapped in config 'centor'; skipping")
        args.models = [m for m in args.models if m != "centor"]
    elif "centor" in args.models and len(c_used) < 4:
        print(f"centor: modified score, items available = {c_used} + age")

    out = {}
    add = lambda name, frame: out.setdefault(name + suffix, []).append(frame)  # noqa: E731
    sym_defs = symptom_models(n_bin)

    for r, k, tr, te in outer_splits(cfg, df, args.repeats):
        cv = inner_cv(cfg, r, k)
        print(f"repeat {r} fold {k}")
        sym = {}
        for name, (est, grid) in sym_defs.items():
            if name in args.models or (name == "sym_lr" and need_img):
                model, inner = fit_select(est, grid, X_tab[tr], y[tr], cv)
                sym[name] = (proba(model, X_tab[te]), inner)
                if name in args.models:
                    add(name, oof_frame(df, te, r, k, sym[name][0], choose_threshold(y[tr], inner)))
        if "centor" in args.models:
            s = (c_score + 1) / 6.0  # McIsaac range -1..5 mapped to 0..1 (a rank score, not a probability)
            add("centor", oof_frame(df, te, r, k, s[te], choose_threshold(y[tr], s[tr])))

        for bb, (F, F_flip, _) in feats.items():
            img_est = Pipeline([("sc", StandardScaler()), ("clf", lr())])
            img, img_inner = fit_select(img_est, {"clf__C": [1e-3, 1e-2, 1e-1, 1]}, F[tr], y[tr], cv)
            p_img = (proba(img, F[te]) + proba(img, F_flip[te])) / 2
            p_sym, sym_inner = sym["sym_lr"]
            if "img_lr" in args.models:
                add(f"img_lr_{bb}", oof_frame(df, te, r, k, p_img, choose_threshold(y[tr], img_inner)))
            if "late" in args.models:
                add(f"late_{bb}", oof_frame(df, te, r, k, (p_sym + p_img) / 2,
                                            choose_threshold(y[tr], (sym_inner + img_inner) / 2)))
            if "stack" in args.models:
                eps = 1e-6
                Z_tr = logit(np.clip(np.c_[sym_inner, img_inner], eps, 1 - eps))
                Z_te = logit(np.clip(np.c_[p_sym, p_img], eps, 1 - eps))
                meta = lr().fit(Z_tr, y[tr])
                meta_inner = cross_val_predict(lr(), Z_tr, y[tr], cv=cv, method="predict_proba")[:, 1]
                add(f"stack_{bb}", oof_frame(df, te, r, k, proba(meta, Z_te), choose_threshold(y[tr], meta_inner),
                                             w_sym=meta.coef_[0, 0], w_img=meta.coef_[0, 1]))
            if "early" in args.models:
                X = np.hstack([X_tab, F])
                X_flip = np.hstack([X_tab, F_flip])
                img_cols = list(range(X_tab.shape[1], X.shape[1]))
                grid = {"clf__C": [0.01, 0.1, 1], "prep__img__pca__n_components": [16, 64]}
                est = Pipeline([("prep", tab_transformer(n_bin, img_cols, img_pca=16)), ("clf", lr())])
                model, inner = fit_select(est, grid, X[tr], y[tr], cv)
                p = (proba(model, X[te]) + proba(model, X_flip[te])) / 2
                add(f"early_{bb}", oof_frame(df, te, r, k, p, choose_threshold(y[tr], inner)))

    print("\nOut-of-fold results")
    for name, frames in out.items():
        save_oof(cfg, name, frames)


if __name__ == "__main__":
    main()
