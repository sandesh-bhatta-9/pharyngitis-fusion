"""Phone / site shortcut experiments.

The two phones were used in two cities with very different bacterial prevalence (Xiaomi 44%, Samsung 15%),
so a model can score well by recognising the phone. These analyses separate that shortcut from real
throat-image signal.

Subcommands:
  size      `size` baseline: logistic regression on the original image width, height and aspect ratio
            (shared folds, saved to results/oof/size.csv) + how well image embeddings identify the phone.
  within    models trained AND tested inside one phone (5-fold x 5-repeat CV within each phone).
  cross     train on all patients of one phone, test on the other phone (both directions).
  largeimg  within-phone AUC of saved models restricted to images with original short side >= --min-side px.
  lowlevel  "low-level" baseline: colour, brightness, darkness and edge statistics of each photo (no deep features);
            label AUC overall and within phone, and how well these statistics identify the phone.

Usage:
    python -m src.site_shortcut size
    python -m src.site_shortcut within --backbones vit_small_patch14_dinov2 convnext_tiny
    python -m src.site_shortcut cross --backbones vit_small_patch14_dinov2 convnext_tiny
    python -m src.site_shortcut largeimg --models img_lr_vit_small_patch14_dinov2 phone sym_lgbm
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import lr, symptom_models
from .common import (add_config_arg, backbone_key, choose_threshold, load_config, load_data, load_features,
                     load_oof, oof_frame, outer_splits, save_oof, tab_array, tab_transformer)
from .evaluate import bootstrap_auc

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*did not converge.*")
PHONES = {"SM-G998B": "Samsung S21 Ultra", "2201117SG": "Xiaomi"}


def img_model():
    return GridSearchCV(Pipeline([("sc", StandardScaler()), ("clf", lr())]),
                        {"clf__C": [1e-3, 1e-2, 1e-1, 1]}, scoring="roc_auc", cv=3, n_jobs=-1)


def sym_model(n_bin):
    est, grid = symptom_models(n_bin)["sym_lr"]
    return GridSearchCV(est, grid, scoring="roc_auc", cv=3, n_jobs=-1)


def early_model(n_bin, d_img):
    img_cols = list(range(n_bin + 1, n_bin + 1 + d_img))
    return GridSearchCV(Pipeline([("prep", tab_transformer(n_bin, img_cols, img_pca=32)), ("clf", lr())]),
                        {"clf__C": [0.01, 0.1, 1]}, scoring="roc_auc", cv=3, n_jobs=-1)


def size_features(df):
    wh = df["orig_size"].str.split("x", expand=True).astype(float)
    w, h = wh[0].to_numpy(), wh[1].to_numpy()
    return np.c_[np.log(w), np.log(h), np.log(w / h), np.log(w * h)]


def cmd_size(cfg, backbones):
    df, _ = load_data(cfg)
    y = df["y"].to_numpy()
    X = size_features(df)
    frames = []
    for r, k, tr, te in outer_splits(cfg, df):
        m = GridSearchCV(Pipeline([("sc", StandardScaler()), ("clf", lr())]), {"clf__C": [0.01, 0.1, 1, 10]},
                         scoring="roc_auc", cv=3).fit(X[tr], y[tr])
        inner = cross_val_predict(m.best_estimator_, X[tr], y[tr], cv=3, method="predict_proba")[:, 1]
        frames.append(oof_frame(df, te, r, k, m.predict_proba(X[te])[:, 1], choose_threshold(y[tr], inner)))
    print("Label from original image size only:")
    save_oof(cfg, "size", frames)

    known = df["exif_phone"].isin(PHONES).to_numpy()
    is_xiaomi = (df["exif_phone"] == "2201117SG").to_numpy()[known]
    rows = []
    skf = StratifiedKFold(5, shuffle=True, random_state=cfg["seed"])
    for name, F in [("image size", X)] + [(bb, load_features(cfg, bb, df)[0]) for bb in backbones]:
        p = cross_val_predict(Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.1, max_iter=5000))]),
                              F[known], is_xiaomi, cv=skf, method="predict_proba")[:, 1]
        rows.append({"features": name, "auc_identify_phone": roc_auc_score(is_xiaomi, p), "n": int(known.sum())})
    return pd.DataFrame(rows)


def cv_within(X, y, make, seed, repeats=5, folds=5):
    aucs = []
    for r in range(repeats):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(X, y):
            p[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, p))
    return np.mean(aucs), np.std(aucs, ddof=1)


def cmd_within(cfg, backbones):
    df, cols = load_data(cfg)
    X_tab, n_bin, _ = tab_array(df, cols)
    feats = {bb: load_features(cfg, bb, df)[0] for bb in backbones}
    rows = []
    for code, phone in PHONES.items():
        m = (df["exif_phone"] == code).to_numpy()
        y = df["y"].to_numpy()[m]
        sets = {"symptoms (LR)": (X_tab[m], lambda: sym_model(n_bin))}
        for bb, F in feats.items():
            sets[f"image {bb} (LR)"] = (F[m], img_model)
            sets[f"image {bb} + symptoms (LR)"] = (np.hstack([X_tab[m], F[m]]),
                                                   lambda d=F.shape[1]: early_model(n_bin, d))
        for name, (X, make) in sets.items():
            mu, sd = cv_within(X, y, make, cfg["seed"])
            rows.append({"phone": phone, "n": int(m.sum()), "bacterial": int(y.sum()), "model": name,
                         "auc_mean": mu, "auc_sd": sd})
            print(f"  {phone:18s} {name:48s} AUC {mu:.3f} ± {sd:.3f}")
    return pd.DataFrame(rows)


def cmd_cross(cfg, backbones, n_boot):
    df, cols = load_data(cfg)
    X_tab, n_bin, _ = tab_array(df, cols)
    y_all = df["y"].to_numpy()
    feats = {bb: load_features(cfg, bb, df) for bb in backbones}
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for a, b in [("SM-G998B", "2201117SG"), ("2201117SG", "SM-G998B")]:
        tr = (df["exif_phone"] == a).to_numpy()
        te = (df["exif_phone"] == b).to_numpy()
        preds = {"symptoms (LR)": sym_model(n_bin).fit(X_tab[tr], y_all[tr]).predict_proba(X_tab[te])[:, 1]}
        for bb, (F, F_flip, _) in feats.items():
            m = img_model().fit(F[tr], y_all[tr])
            preds[f"image {bb} (LR)"] = (m.predict_proba(F[te])[:, 1] + m.predict_proba(F_flip[te])[:, 1]) / 2
        for name, p in preds.items():
            lo, hi = bootstrap_auc(y_all[te], p, n_boot, rng)
            rows.append({"train_phone": PHONES[a], "test_phone": PHONES[b], "model": name,
                         "auc": roc_auc_score(y_all[te], p), "auc_lo": lo, "auc_hi": hi,
                         "n_train": int(tr.sum()), "n_test": int(te.sum())})
            print(f"  {PHONES[a]:>18s} -> {PHONES[b]:18s} {name:40s} AUC {rows[-1]['auc']:.3f} [{lo:.3f}, {hi:.3f}]")
    return pd.DataFrame(rows)


def lowlevel_features(cfg, df):
    """Global photo statistics that capture lighting, colour cast, zoom/framing (dark cavity share) and sharpness."""
    from PIL import Image, ImageFilter
    rows = []
    for path in df["image"]:
        im = Image.open(cfg["paths"]["processed"] / path).convert("RGB")
        a = np.asarray(im, dtype=np.float32) / 255.0
        hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
        grey = a.mean(axis=2)
        edges = np.asarray(im.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        rows.append([*a.reshape(-1, 3).mean(0), *a.reshape(-1, 3).std(0), *hsv.reshape(-1, 3).mean(0),
                     *hsv.reshape(-1, 3).std(0), (grey < 0.15).mean(), (grey > 0.85).mean(),
                     ((r - g) > 0.25).mean(), edges.mean(), (edges > 0.2).mean()])
    return np.array(rows)


def cmd_lowlevel(cfg):
    df, _ = load_data(cfg)
    y = df["y"].to_numpy()
    X = lowlevel_features(cfg, df)
    make = lambda: GridSearchCV(Pipeline([("sc", StandardScaler()), ("clf", lr())]),  # noqa: E731
                                {"clf__C": [0.01, 0.1, 1, 10]}, scoring="roc_auc", cv=3)
    frames = []
    for r, k, tr, te in outer_splits(cfg, df):
        m = make().fit(X[tr], y[tr])
        inner = cross_val_predict(m.best_estimator_, X[tr], y[tr], cv=3, method="predict_proba")[:, 1]
        frames.append(oof_frame(df, te, r, k, m.predict_proba(X[te])[:, 1], choose_threshold(y[tr], inner)))
    print("Label from low-level photo statistics:")
    save_oof(cfg, "lowlevel", frames)
    known = df["exif_phone"].isin(PHONES).to_numpy()
    is_xiaomi = (df["exif_phone"] == "2201117SG").to_numpy()[known]
    p = cross_val_predict(make(), X[known], is_xiaomi, cv=StratifiedKFold(5, shuffle=True, random_state=cfg["seed"]),
                          method="predict_proba")[:, 1]
    rows = [{"analysis": "identify phone (Xiaomi vs Samsung)", "auc": roc_auc_score(is_xiaomi, p), "n": int(known.sum())}]
    for code, name in PHONES.items():
        msk = (df["exif_phone"] == code).to_numpy()
        mu, sd = cv_within(X[msk], y[msk], make, cfg["seed"])
        rows.append({"analysis": f"label, trained and tested within {name}", "auc": mu, "auc_sd": sd, "n": int(msk.sum())})
    return pd.DataFrame(rows)


def cmd_largeimg(cfg, models, min_side):
    df, _ = load_data(cfg)
    wh = df["orig_size"].str.split("x", expand=True).astype(float)
    big = df.loc[wh.min(axis=1) >= min_side, "patient_id"]
    phone = df.set_index("patient_id")["exif_phone"]
    rows = []
    for m in models:
        o = load_oof(cfg, m)
        for subset, oo in [("all images", o), (f"short side >= {min_side}px", o[o["patient_id"].isin(big)])]:
            for code, name in PHONES.items():
                g = oo[oo["patient_id"].map(phone) == code]
                aucs = [roc_auc_score(s["y"], s["p"]) for _, s in g.groupby("repeat") if 0 < s["y"].sum() < len(s)]
                rows.append({"model": m, "subset": subset, "phone": name, "n": g["patient_id"].nunique(),
                             "auc_within_phone": np.mean(aucs) if aucs else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("size", "within", "cross"):
        s = sub.add_parser(name)
        s.add_argument("--backbones", nargs="*", help="default: config features.backbones")
        if name == "cross":
            s.add_argument("--n-boot", type=int, default=1000)
    sub.add_parser("lowlevel")
    s = sub.add_parser("largeimg")
    s.add_argument("--models", nargs="+", required=True)
    s.add_argument("--min-side", type=int, default=500)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = cfg["paths"]["metrics"]
    out.mkdir(parents=True, exist_ok=True)
    bbs = [backbone_key(b) for b in (getattr(args, "backbones", None) or cfg["features"]["backbones"])]
    if args.cmd == "size":
        res = cmd_size(cfg, bbs)
    elif args.cmd == "within":
        res = cmd_within(cfg, bbs)
    elif args.cmd == "cross":
        res = cmd_cross(cfg, bbs, args.n_boot)
    elif args.cmd == "lowlevel":
        res = cmd_lowlevel(cfg)
    else:
        res = cmd_largeimg(cfg, args.models, args.min_side)
    path = out / f"site_{args.cmd}.csv"
    res.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.round(3).to_string(index=False))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
