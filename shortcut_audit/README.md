# shortcut-audit

Checks whether a classifier's AUC comes from recognising the acquisition site (hospital, scanner, phone) instead of
the disease. Written for the paper *Recognising the phone, not the throat* (this repository), usable with any
predictions.

```bash
pip install "git+https://github.com/sandesh-bhatta-9/pharyngitis-fusion"
```

```python
from shortcut_audit import audit
res = audit(y, p, site)      # y: 0/1 labels, p: scores, site: site of each patient
```

or from the command line, for a CSV with one row per prediction:

```bash
python -m shortcut_audit predictions.csv --y y --p p --site site --group repeat
```

| Output | Meaning |
| --- | --- |
| `auc_pooled` | Ordinary AUC |
| `auc_deconfounded` | AUC with each patient weighted by P(y) / P(y \| site), so the label is independent of the site. A score that uses only the site gets 0.5; a score shifted by site is penalised |
| `auc_adjusted` | Covariate-adjusted AUC (Janes & Pepe 2008): within-site AUCs averaged over the sites of the positive cases |
| `auc_within_<site>` | AUC inside each site |
| `auc_site_only` | AUC of the site's prevalence as a score: how strongly the site alone predicts the label |

Pooled, deconfounded and adjusted AUC come with patient-level bootstrap 95% CIs (`_lo`, `_hi`).

Also included: `balance_weights` (the weights above, e.g. for training), `LeaceEraser` (linear concept erasure,
Belrose et al. 2023) and `site_probe_auc` (how well a linear probe recovers the site from features).

**Reading the numbers.** If pooled AUC is well above deconfounded AUC, part of the performance comes from the site. If
`auc_site_only` is near the pooled AUC, the site alone explains it. Report deconfounded AUC next to pooled AUC for
any multi-site dataset where prevalence differs between sites.
