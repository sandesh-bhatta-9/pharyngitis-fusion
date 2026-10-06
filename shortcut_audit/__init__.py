"""shortcut_audit: check whether a classifier's AUC comes from recognising the acquisition site.

Given labels, predicted scores and the site (hospital, device, scanner...) of each patient:

    from shortcut_audit import audit
    res = audit(y, p, site)          # dict: pooled, deconfounded, adjusted, within-site and site-only AUC

Metrics
  pooled AUC         ordinary AUC
  deconfounded AUC   AUC with patient i weighted by P(y_i) / P(y_i | site_i), so that the label is independent of the
                     site. Any score that depends only on the site gets exactly 0.5. Pairs from different sites count,
                     so a score shifted by site is penalised.
  adjusted AUC       covariate-adjusted AUC of Janes & Pepe (Am J Epidemiol 2008): within-site AUCs averaged over the
                     sites of the positive cases. Ignores all pairs from different sites, so a site offset is invisible.
  within-site AUC    AUC inside each site
  site-only AUC      AUC of the site's own prevalence as a score: how strongly the site predicts the label

Tools
  balance_weights    the weights P(y) / P(y | site)
  LeaceEraser        least-squares linear concept erasure (Belrose et al., NeurIPS 2023)
  site_probe_auc     how well a linear probe recovers the site from features (leakage check)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

__all__ = ["audit", "balance_weights", "deconfounded_auc", "adjusted_auc", "within_site_auc", "site_only_auc",
           "LeaceEraser", "site_probe_auc"]
__version__ = "1.0.0"


def _arrays(y, p, site):
    y, p, site = np.asarray(y).astype(int), np.asarray(p, float), np.asarray(site).astype(str)
    if not (len(y) == len(p) == len(site)):
        raise ValueError("y, p and site must have the same length")
    return y, p, site


def balance_weights(y, site):
    """w_i = P(y_i) / P(y_i | site_i). Under these weights every site has the overall prevalence, and each site keeps
    its total weight (= its number of patients)."""
    y, site = np.asarray(y).astype(int), np.asarray(site).astype(str)
    w = np.ones(len(y))
    for s in np.unique(site):
        m = site == s
        for c in (0, 1):
            pc, pcs = (y == c).mean(), (y[m] == c).mean()
            if pcs > 0:
                w[m & (y == c)] = pc / pcs
    return w


def deconfounded_auc(y, p, site, weights=None):
    y, p, site = _arrays(y, p, site)
    w = balance_weights(y, site) if weights is None else np.asarray(weights, float)
    return roc_auc_score(y, p, sample_weight=w)


def within_site_auc(y, p, site, min_per_class=1):
    """{site: AUC} for sites with at least `min_per_class` positives and negatives."""
    y, p, site = _arrays(y, p, site)
    out = {}
    for s in np.unique(site):
        m = site == s
        if y[m].sum() >= min_per_class and (1 - y[m]).sum() >= min_per_class:
            out[s] = roc_auc_score(y[m], p[m])
    return out


def adjusted_auc(y, p, site):
    """Covariate-adjusted AUC (Janes & Pepe 2008): sum over sites of P(site | y = 1) x AUC within the site."""
    y, p, site = _arrays(y, p, site)
    within = within_site_auc(y, p, site)
    cases = {s: int(y[site == s].sum()) for s in within}
    total = sum(cases.values())
    return float(sum(cases[s] * a for s, a in within.items()) / total) if total else float("nan")


def site_only_auc(y, site):
    """AUC of the site's prevalence used as the score (in-sample): the strength of site-label confounding."""
    y, site = np.asarray(y).astype(int), np.asarray(site).astype(str)
    rate = {s: y[site == s].mean() for s in np.unique(site)}
    return roc_auc_score(y, np.array([rate[s] for s in site]))


def _boot(y, p, site, fn, n_boot, rng):
    vals, n = [], len(y)
    while len(vals) < n_boot:
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n:
            v = fn(y[i], p[i], site[i])
            if np.isfinite(v):
                vals.append(v)
    return np.percentile(vals, [2.5, 97.5])


def audit(y, p, site, n_boot=1000, seed=0):
    """All metrics, with patient-level bootstrap 95% CIs for pooled, deconfounded and adjusted AUC (weights are
    re-estimated in every resample)."""
    y, p, site = _arrays(y, p, site)
    rng = np.random.default_rng(seed)
    fns = {"pooled": lambda a, b, c: roc_auc_score(a, b), "deconfounded": deconfounded_auc,
           "adjusted": adjusted_auc}
    res = {}
    for name, fn in fns.items():
        res[f"auc_{name}"] = float(fn(y, p, site))
        if n_boot:
            res[f"auc_{name}_lo"], res[f"auc_{name}_hi"] = map(float, _boot(y, p, site, fn, n_boot, rng))
    res.update({f"auc_within_{s}": float(a) for s, a in within_site_auc(y, p, site).items()})
    res["auc_site_only"] = float(site_only_auc(y, site))
    res["n"], res["n_pos"], res["n_sites"] = len(y), int(y.sum()), len(np.unique(site))
    return res


class LeaceEraser:
    """Least-squares concept erasure (Belrose et al., NeurIPS 2023). Fit on training features X and the concept Z
    (n x k, e.g. one-hot site without one column); transform removes all linear information about Z."""

    def fit(self, X, Z):
        X, Z = np.asarray(X, float), np.asarray(Z, float).reshape(len(X), -1)
        self.mu = X.mean(0)
        Xc, Zc = X - self.mu, Z - Z.mean(0)
        S = Xc.T @ Xc / len(X)
        s, U = np.linalg.eigh(S)
        keep = s > s.max() * 1e-8
        U, s = U[:, keep], s[keep]
        W, W_inv = (U / np.sqrt(s)) @ U.T, (U * np.sqrt(s)) @ U.T
        Q, R = np.linalg.qr(W @ (Xc.T @ Zc / len(X)))
        Q = Q[:, np.abs(np.diag(R)) > 1e-10]
        self.M = W_inv @ Q @ Q.T @ W
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        return X - (X - self.mu) @ self.M.T


def site_probe_auc(F_train, site_train, F_test, site_test, positive, C=0.1):
    """Leakage probe: standardised logistic regression trained to tell `positive` from the other sites on training
    features, scored (AUC) on test features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    probe.fit(F_train, np.asarray(site_train).astype(str) == positive)
    return roc_auc_score(np.asarray(site_test).astype(str) == positive, probe.predict_proba(F_test)[:, 1])
