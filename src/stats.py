"""Statistical tests for model comparison.

  - Nadeau-Bengio corrected resampled t-test on fold-level AUC differences (main test; accounts for the
    overlap between training sets in repeated CV).
  - DeLong test on pooled out-of-fold predictions, run per repeat.
  - Holm correction across the pre-registered hypotheses.

Usage:
    python -m src.stats --hypotheses                      # H1-H3 for config hypotheses.proposed
    python -m src.stats --pair gated_convnext_tiny sym_lgbm
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.metrics import roc_auc_score

from .common import TAG_SEP, add_config_arg, list_models, load_config, load_oof


# ---------------------------------------------------------------- tests

def corrected_ttest(diffs, n_train, n_test, one_sided=True):
    """Nadeau & Bengio (2003) corrected resampled t-test. Returns (mean diff, t, p)."""
    d = np.asarray(diffs, float)
    J = len(d)
    var = d.var(ddof=1)
    se = np.sqrt((1 / J + n_test / n_train) * var) if var > 0 else np.inf
    t = d.mean() / se
    p = st.t.sf(t, J - 1) if one_sided else 2 * st.t.sf(abs(t), J - 1)
    return d.mean(), t, p


def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong(y, p1, p2):
    """Fast DeLong test (Sun & Xu 2014) for two correlated AUCs. Returns (auc1, auc2, z, two-sided p)."""
    y = np.asarray(y)
    order = np.argsort(-y, kind="stable")
    m = int(y.sum())
    preds = np.vstack([p1, p2])[:, order]
    pos, neg = preds[:, :m], preds[:, m:]
    n = neg.shape[1]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return aucs[0], aucs[1], z, 2 * st.norm.sf(abs(z))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


# ---------------------------------------------------------------- comparisons

def fold_aucs(oof):
    return oof.groupby(["repeat", "fold"]).apply(lambda g: roc_auc_score(g["y"], g["p"]), include_groups=False)


def compare(cfg, a: str, b: str, one_sided=True) -> dict:
    A, B = load_oof(cfg, a), load_oof(cfg, b)
    fa, fb = fold_aucs(A), fold_aucs(B)
    common = fa.index.intersection(fb.index)
    if len(common) < 2:
        raise ValueError(f"{a} and {b} share fewer than 2 folds")
    n = A[A["repeat"] == A["repeat"].iloc[0]].shape[0]
    K = A["fold"].nunique()
    diff, t, p = corrected_ttest(fa[common] - fb[common], n * (K - 1) / K, n / K, one_sided)
    M = A.merge(B, on=["patient_id", "repeat"], suffixes=("_a", "_b"))
    dl = [delong(g["y_a"].to_numpy(), g["p_a"].to_numpy(), g["p_b"].to_numpy())[3] for _, g in M.groupby("repeat")]
    return {"model_a": a, "model_b": b, "auc_a": fa[common].mean(), "auc_b": fb[common].mean(),
            "delta_auc": diff, "t": t, "p_corrected_t": p, "folds": len(common),
            "delong_p_median": float(np.median(dl)), "delong_sig_repeats": f"{sum(x < 0.05 for x in dl)}/{len(dl)}"}


def best_of(cfg, prefixes, exclude=()):
    cands = [m for m in list_models(cfg) if m.startswith(tuple(prefixes)) and TAG_SEP not in m and m not in exclude]
    if not cands:
        return None
    return max(cands, key=lambda m: fold_aucs(load_oof(cfg, m)).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"), action="append", help="test A > B (one-sided)")
    ap.add_argument("--hypotheses", action="store_true", help="run pre-registered H1-H3 with Holm correction")
    ap.add_argument("--proposed", help="override config hypotheses.proposed")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = cfg["paths"]["metrics"]
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.hypotheses:
        prop = args.proposed or cfg["hypotheses"]["proposed"]
        targets = {"H1 vs best symptom-only": best_of(cfg, ["sym_", "centor"]),
                   "H2 vs best image-only": best_of(cfg, ["img_", "ft_"]),
                   "H3 vs best standard fusion": best_of(cfg, ["late_", "early_", "stack_"])}
        for h, other in targets.items():
            if other:
                rows.append({"hypothesis": h, **compare(cfg, prop, other)})
        res = pd.DataFrame(rows)
        res["p_holm"] = holm(res["p_corrected_t"])
        path = out_dir / "hypotheses.csv"
    else:
        for a, b in args.pair or []:
            rows.append(compare(cfg, a, b))
        res = pd.DataFrame(rows)
        path = out_dir / "comparisons.csv"
    if res.empty:
        raise SystemExit("Nothing to compare: pass --pair A B or --hypotheses")
    res.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.round(4).to_string(index=False))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
