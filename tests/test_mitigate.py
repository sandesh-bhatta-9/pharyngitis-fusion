import numpy as np
from sklearn.metrics import roc_auc_score

from src.mitigate import Leace, balance_weights, onehot  # re-exported from shortcut_audit


def _cohort(seed=0, n=600):
    rng = np.random.default_rng(seed)
    g = rng.choice(["a", "b", "u"], n, p=[0.5, 0.4, 0.1])
    y = (rng.random(n) < np.where(g == "a", 0.45, np.where(g == "b", 0.15, 0.3))).astype(int)
    return rng, g, y


def test_site_only_score_has_deconfounded_auc_one_half():
    _, g, y = _cohort()
    w = balance_weights(y, g)
    rate = {v: y[g == v].mean() for v in np.unique(g)}
    p = np.array([rate[v] for v in g])
    assert roc_auc_score(y, p) > 0.6
    assert abs(roc_auc_score(y, p, sample_weight=w) - 0.5) < 1e-12


def test_weights_make_label_independent_of_site_and_keep_group_sizes():
    _, g, y = _cohort(1)
    w = balance_weights(y, g)
    for v in np.unique(g):
        m = g == v
        assert np.isclose(np.average(y[m], weights=w[m]), y.mean())
        assert np.isclose(w[m].sum(), m.sum())


def test_leace_removes_linear_site_information():
    rng, g, _ = _cohort(2)
    Z = onehot(g, ["a", "b", "u"])[:, 1:]
    X = rng.normal(size=(len(g), 50)) + Z @ rng.normal(size=(2, 50)) * 2
    Xe = Leace().fit(X, Z).transform(X)
    cov = (Xe - Xe.mean(0)).T @ (Z - Z.mean(0)) / len(g)
    assert np.abs(cov).max() < 1e-10
