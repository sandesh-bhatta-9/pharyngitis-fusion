import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.stats import corrected_ttest, delong, holm


def test_delong_aucs_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p1 = y + rng.normal(0, 1, 300)
    p2 = y + rng.normal(0, 1.5, 300)
    a1, a2, _, p = delong(y, p1, p2)
    assert a1 == pytest.approx(roc_auc_score(y, p1))
    assert a2 == pytest.approx(roc_auc_score(y, p2))
    assert 0 <= p <= 1


def test_delong_identical_predictions_not_significant():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = y + rng.normal(0, 1, 200)
    _, _, z, pval = delong(y, p, p.copy())
    assert z == 0 and pval == 1.0


def test_delong_handles_ties():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    p1 = np.array([0.1, 0.4, 0.4, 0.8, 0.4, 0.9, 0.2, 0.4])
    p2 = np.array([0.2, 0.2, 0.6, 0.6, 0.3, 0.7, 0.2, 0.5])
    a1, a2, _, _ = delong(y, p1, p2)
    assert a1 == pytest.approx(roc_auc_score(y, p1))
    assert a2 == pytest.approx(roc_auc_score(y, p2))


def test_corrected_ttest_is_more_conservative_than_plain():
    rng = np.random.default_rng(2)
    d = rng.normal(0.02, 0.03, 25)
    _, t_corr, p_corr = corrected_ttest(d, n_train=588, n_test=147)
    t_plain = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    assert abs(t_corr) < abs(t_plain)
    assert 0 <= p_corr <= 1


def test_corrected_ttest_one_sided_direction():
    better = np.full(10, 0.05) + np.linspace(-0.01, 0.01, 10)
    _, _, p_better = corrected_ttest(better, 400, 100)
    _, _, p_worse = corrected_ttest(-better, 400, 100)
    assert p_better < 0.05 < p_worse


def test_holm_known_values():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(holm([0.5, 0.9]), [1.0, 1.0])
