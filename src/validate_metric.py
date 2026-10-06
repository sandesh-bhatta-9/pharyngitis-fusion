"""Validation of the deconfounded AUC (ground truth known by construction).

Subcommands:
  sweep     Semi-synthetic, on the real data. Patients with a known phone are subsampled (200 per phone, overall
            prevalence 0.25) so that the bacterial rate differs between phones by delta = 0, 0.1, ..., 0.4 (Xiaomi
            higher, as in the full data, where delta = 0.29). At delta = 0 the phone carries no label information, so
            the pooled AUC there is the shortcut-free reference. Each subsample is evaluated with stratified 5-fold
            CV for phone-only, image (DINOv2 + logistic regression), image after LEACE, and symptoms.
  simulate  Fully synthetic. Two sites, a true marker X with known AUC (0.60) and a site signature S that identifies
            the site. For each delta, a logistic regression is trained on [X, S] in a confounded sample and scored
            (i) on a confounded test sample with every metric and (ii) on a large unconfounded test sample from the
            same sites (the target that the deconfounded AUC estimates).

Usage:
    python -m src.validate_metric sweep --resamples 20
    python -m src.validate_metric simulate --reps 500
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shortcut_audit import LeaceEraser, adjusted_auc, balance_weights, within_site_auc
from .common import add_config_arg, load_config, load_data, load_features, set_seed, tab_array, tab_transformer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*did not converge.*")
DELTAS = [0.0, 0.1, 0.2, 0.3, 0.4]


def metrics(y, p, site):
    w = balance_weights(y, site)
    within = within_site_auc(y, p, site)
    return {"auc": roc_auc_score(y, p), "auc_deconf": roc_auc_score(y, p, sample_weight=w),
            "auc_adjusted": adjusted_auc(y, p, site), "auc_within": np.mean(list(within.values()))}


def lr_model(n_bin=None):
    if n_bin is None:
        est, grid = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))]), \
            {"clf__C": [1e-3, 1e-2, 1e-1, 1]}
    else:
        est, grid = Pipeline([("prep", tab_transformer(n_bin)),
                              ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))]), \
            {"clf__C": [0.01, 0.1, 1, 10]}
    return GridSearchCV(est, grid, scoring="roc_auc", cv=3, n_jobs=-1)


def subsample(rng, y, phone, delta, n_per=200, prev=0.25):
    """Indices with P(y | Xiaomi) = prev + delta/2 and P(y | Samsung) = prev - delta/2, n_per patients per phone."""
    idx = []
    for code, rate in (("2201117SG", prev + delta / 2), ("SM-G998B", prev - delta / 2)):
        n_pos = int(round(n_per * rate))
        pos = np.flatnonzero((phone == code) & (y == 1))
        neg = np.flatnonzero((phone == code) & (y == 0))
        if n_pos > len(pos) or n_per - n_pos > len(neg):
            raise ValueError(f"not enough patients for delta={delta} on {code}")
        idx += list(rng.choice(pos, n_pos, replace=False)) + list(rng.choice(neg, n_per - n_pos, replace=False))
    return np.array(idx)


def cmd_sweep(cfg, resamples, backbone):
    df, cols = load_data(cfg)
    y_all = df["y"].to_numpy()
    phone_all = df["exif_phone"].fillna("unknown").to_numpy()
    F_all, Ff_all, _ = load_features(cfg, backbone, df)
    X_tab, n_bin, _ = tab_array(df, cols)
    rows = []
    for delta in DELTAS:
        for rep in range(resamples):
            rng = np.random.default_rng(cfg["seed"] + 1000 * rep + int(delta * 100))
            idx = subsample(rng, y_all, phone_all, delta)
            y, ph = y_all[idx], phone_all[idx]
            F, Ff, T = F_all[idx], Ff_all[idx], X_tab[idx]
            preds = {m: np.zeros(len(idx)) for m in ("phone", "image", "image_leace", "symptoms")}
            strat = np.char.add(y.astype(str), ph)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(F, strat):
                rate = {v: y[tr][ph[tr] == v].mean() for v in np.unique(ph[tr])}
                preds["phone"][te] = [rate[v] for v in ph[te]]
                m = lr_model().fit(F[tr], y[tr])
                preds["image"][te] = (m.predict_proba(F[te])[:, 1] + m.predict_proba(Ff[te])[:, 1]) / 2
                e = LeaceEraser().fit(F[tr], (ph[tr] == "2201117SG").astype(float))
                m = lr_model().fit(e.transform(F[tr]), y[tr])
                preds["image_leace"][te] = (m.predict_proba(e.transform(F[te]))[:, 1]
                                            + m.predict_proba(e.transform(Ff[te]))[:, 1]) / 2
                preds["symptoms"][te] = lr_model(n_bin).fit(T[tr], y[tr]).predict_proba(T[te])[:, 1]
            for name, p in preds.items():
                rows.append({"delta": delta, "resample": rep, "model": name, **metrics(y, p, ph)})
        cur = pd.DataFrame(rows)
        cur = cur[cur["delta"] == delta].groupby("model")[["auc", "auc_deconf", "auc_adjusted"]].mean()
        print(f"delta {delta:.1f}\n{cur.round(3).to_string()}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(cfg["paths"]["metrics"] / "validate_sweep.csv", index=False)
    summ = res.groupby(["model", "delta"]).agg(["mean", "std"]).drop(columns="resample")
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(cfg["paths"]["metrics"] / "validate_sweep_summary.csv", index=False)
    print(summ.round(3).to_string())


def sim_sample(rng, n, delta, prev, share_b, shift, kappa):
    site = (rng.random(n) < share_b).astype(int)  # 1 = site B
    p_a, p_b = prev - delta * share_b, prev + delta * (1 - share_b)  # site B has the higher prevalence
    y = (rng.random(n) < np.where(site == 1, p_b, p_a)).astype(int)
    x = rng.normal(shift * y, 1.0)
    s = rng.normal(kappa * site, 1.0)
    return np.c_[x, s], y, np.where(site == 1, "B", "A")


def cmd_simulate(cfg, reps, n=700, prev=0.28, share_b=0.45, true_auc=0.60, kappa=3.0):
    shift = np.sqrt(2) * norm.ppf(true_auc)
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for delta in DELTAS:
        for r in range(reps):
            X_tr, y_tr, _ = sim_sample(rng, n, delta, prev, share_b, shift, kappa)
            X_te, y_te, s_te = sim_sample(rng, n, delta, prev, share_b, shift, kappa)
            X_big, y_big, _ = sim_sample(rng, 20000, 0.0, prev, share_b, shift, kappa)
            model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
            scores = {"marker only": (X_te[:, 0], X_big[:, 0]), "site only": (X_te[:, 1], X_big[:, 1]),
                      "learned (marker + site)": (model.decision_function(X_te), model.decision_function(X_big))}
            for name, (p, p_big) in scores.items():
                rows.append({"delta": delta, "rep": r, "score": name, **metrics(y_te, p, s_te),
                             "auc_target": roc_auc_score(y_big, p_big)})
    res = pd.DataFrame(rows)
    res.to_csv(cfg["paths"]["metrics"] / "validate_simulation.csv", index=False)
    g = res.groupby(["score", "delta"])
    summ = g[["auc", "auc_deconf", "auc_adjusted", "auc_within", "auc_target"]].mean()
    for m in ("auc", "auc_deconf", "auc_adjusted", "auc_within"):
        summ[f"{m}_sd"] = g[m].std()
        summ[f"{m}_bias"] = g.apply(lambda d, m=m: (d[m] - d["auc_target"]).mean(), include_groups=False)
    summ.reset_index().to_csv(cfg["paths"]["metrics"] / "validate_simulation_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summ.round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--resamples", type=int, default=20)
    s.add_argument("--backbone", default="vit_small_patch14_dinov2")
    s = sub.add_parser("simulate")
    s.add_argument("--reps", type=int, default=500)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    cfg["paths"]["metrics"].mkdir(parents=True, exist_ok=True)
    if args.cmd == "sweep":
        cmd_sweep(cfg, args.resamples, args.backbone)
    else:
        cmd_simulate(cfg, args.reps)


if __name__ == "__main__":
    main()
