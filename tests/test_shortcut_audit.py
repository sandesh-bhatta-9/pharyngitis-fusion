import numpy as np
import pytest

from shortcut_audit import adjusted_auc, audit, deconfounded_auc, site_only_auc, within_site_auc


def _data(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    site = rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2])
    y = (rng.random(n) < np.select([site == "A", site == "B"], [0.4, 0.15], 0.25)).astype(int)
    return rng, site, y


def test_site_only_score_gets_one_half_on_both_metrics():
    _, site, y = _data()
    p = np.select([site == "A", site == "B"], [0.9, 0.1], 0.5)
    assert site_only_auc(y, site) > 0.6
    assert deconfounded_auc(y, p, site) == pytest.approx(0.5, abs=1e-12)
    assert adjusted_auc(y, p, site) == pytest.approx(0.5, abs=1e-12)


def test_site_offset_lowers_deconfounded_but_not_adjusted_auc():
    rng, site, y = _data(1)
    marker = rng.normal(0.7 * y, 1.0)
    shifted = marker + np.where(site == "A", 2.0, 0.0)
    assert adjusted_auc(y, shifted, site) == pytest.approx(adjusted_auc(y, marker, site))
    assert deconfounded_auc(y, shifted, site) < deconfounded_auc(y, marker, site) - 0.05


def test_adjusted_auc_is_case_weighted_mean_of_within_site_aucs():
    rng, site, y = _data(2)
    p = rng.normal(y, 1.0)
    within = within_site_auc(y, p, site)
    cases = {s: y[site == s].sum() for s in within}
    expected = sum(cases[s] * within[s] for s in within) / sum(cases.values())
    assert adjusted_auc(y, p, site) == pytest.approx(expected)


def test_audit_returns_cis_that_contain_the_estimate():
    rng, site, y = _data(3)
    res = audit(y, rng.normal(y, 1.0), site, n_boot=200)
    for k in ("pooled", "deconfounded", "adjusted"):
        assert res[f"auc_{k}_lo"] <= res[f"auc_{k}"] <= res[f"auc_{k}_hi"]
    assert res["n_sites"] == 3
