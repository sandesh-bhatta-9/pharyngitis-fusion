"""Create the fixed outer folds (5-fold x 5 repeats) used by every model: data/processed/folds.csv."""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .common import add_config_arg, load_config, load_data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_arg(ap)
    args = ap.parse_args()
    cfg = load_config(args.config)
    df, _ = load_data(cfg)
    K, R = cfg["cv"]["outer_folds"], cfg["cv"]["repeats"]

    key = df["y"].astype(str)
    strat = [c for c in cfg["cv"].get("stratify_meta", []) if c in df.columns]
    if strat:
        cand = key + "_" + df[strat].astype(str).agg("_".join, axis=1)
        if cand.value_counts().min() >= K:
            key = cand
            print(f"Stratifying on label x {strat}")
        else:
            print(f"Some label x {strat} groups have < {K} patients; stratifying on label only")

    rows = []
    for r in range(R):
        skf = StratifiedKFold(K, shuffle=True, random_state=cfg["seed"] + r)
        for k, (_, te) in enumerate(skf.split(df, key)):
            rows += [{"patient_id": pid, "repeat": r, "fold": k} for pid in df["patient_id"].iloc[te]]
    out = pd.DataFrame(rows)
    path = cfg["paths"]["processed"] / "folds.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {path}: {R} repeats x {K} folds, {len(df)} patients")
    print(out.merge(df[["patient_id", "y"]]).groupby(["repeat", "fold"])["y"].agg(["size", "mean"]).head(K).round(3))


if __name__ == "__main__":
    main()
