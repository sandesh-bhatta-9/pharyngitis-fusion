"""Command line: python -m shortcut_audit predictions.csv --y y --p p --site site [--group repeat]

Reads a CSV of predictions and prints pooled, deconfounded, adjusted, within-site and site-only AUC. With --group,
metrics are computed per group (e.g. per cross-validation repeat) and averaged.
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import audit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--y", default="y", help="label column (0/1)")
    ap.add_argument("--p", default="p", help="score column")
    ap.add_argument("--site", default="site", help="site / device column")
    ap.add_argument("--group", help="optional column to compute metrics per group and average (e.g. repeat)")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    df[args.site] = df[args.site].fillna("unknown")
    groups = df.groupby(args.group) if args.group else [(None, df)]
    rows = [audit(g[args.y], g[args.p], g[args.site], n_boot=args.n_boot) for _, g in groups]
    res = pd.DataFrame(rows).mean(numeric_only=True)
    width = max(len(k) for k in res.index)
    for k, v in res.items():
        print(f"{k:<{width}}  {v:.4f}" if isinstance(v, float) and not k.startswith("n") else f"{k:<{width}}  {v:g}")


if __name__ == "__main__":
    main()
