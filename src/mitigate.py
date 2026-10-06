"""Shortcut mitigation: can the phone/site shortcut be removed while keeping real throat-image signal?

Every method is an image-only model on frozen embeddings, trained on the shared outer folds (5 x 5) and
compared with the plain logistic regression (`base`, = img_lr). Phone group = EXIF phone, unknown phone is
its own group. Nothing is fitted on the outer test fold; phone statistics come from the training fold.

Methods:
  base        logistic regression on the embeddings (reference)
  sog         same, on embeddings of colour-constancy-corrected photos (Shades of Gray, p = 6)
  center      per-phone standardisation of the embeddings (training-fold mean/SD of each phone group)
  leace       linear concept erasure of the phone group (LEACE, Belrose et al. 2023)
  reweight    training weights P(y) / P(y | phone), so label and phone are independent in training
  adjust      phone as a covariate during training, averaged out at prediction (image part only)
  leace_rw    leace + reweight
  sog_leace_rw  sog + leace + reweight
  mlp         one-hidden-layer network (lambda = 0, the control for the adversarial models)
  adv<l>      same network with a gradient-reversal phone head, adversarial weight lambda = l

Metrics (summary):
  auc               pooled out-of-fold AUC, mean over repeats
  auc_deconf        AUC with patients weighted by P(y) / P(y | phone), i.e. as if every phone had the same
                    prevalence; the phone-only model scores exactly 0.5 on this metric
  auc_<phone>       AUC within each phone; auc_within = mean of the two
  phone_id_auc      how well a linear probe identifies the phone from the (transformed) test features

Subcommands:
  extract   cache Shades-of-Gray embeddings: data/features/<bb>__sog.npz
  run       fit the methods; OOF files go to results/mitigation/oof/
  cross     train on one phone, test on the other, for base / sog / center
  audit     deconfounded and within-phone AUC of main and mitigation models + corrected t-tests vs base

Usage:
    python -m src.mitigate extract
    python -m src.mitigate run
    python -m src.mitigate cross
    python -m src.mitigate audit
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import (add_config_arg, backbone_key, inner_cv, load_config, load_data, load_features, load_oof,
                     oof_frame, outer_splits, save_oof, set_seed)
from .evaluate import bootstrap_auc
from .stats import corrected_ttest
from shortcut_audit import LeaceEraser as Leace, adjusted_auc, balance_weights  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*did not converge.*")
PHONES = {"SM-G998B": "Samsung", "2201117SG": "Xiaomi"}
LR_METHODS = ["base", "sog", "center", "leace", "reweight", "adjust", "leace_rw", "sog_leace_rw"]
ADV_LAMBDAS = [0.0, 0.3, 1.0, 3.0, 10.0]
C_GRID = {"clf__C": [1e-3, 1e-2, 1e-1, 1]}


def lr(C=1.0):  # same as baselines.lr; not imported, because LightGBM's OpenMP crashes next to torch on macOS
    return LogisticRegression(C=C, class_weight="balanced", max_iter=5000)


def mit_cfg(cfg):
    cfg["paths"]["oof"] = cfg["paths"]["results"] / "mitigation" / "oof"
    return cfg


def groups(df):
    return df["exif_phone"].fillna("unknown").to_numpy()


def onehot(g, levels):
    return (g[:, None] == np.asarray(levels)[None, :]).astype(float)


# ---------------------------------------------------------------- colour constancy features

def shades_of_gray(a, p=6):
    """Shades-of-Gray colour constancy (Finlayson & Trezzi 2004) on an RGB float image in [0, 1]."""
    e = np.power(np.mean(np.power(a.reshape(-1, 3), p), axis=0), 1 / p)
    e = e / np.sqrt(np.sum(e ** 2))
    return np.clip(a / (e * np.sqrt(3)), 0, 1)


def cmd_extract(cfg, backbones):
    import torch
    from PIL import Image

    from .common import torch_device
    from .extract_features import create_backbone, embed, transforms
    df, _ = load_data(cfg)
    device = torch_device()
    test_tf, flip_tf, _ = transforms(cfg["image"]["crop"])
    images = []
    for path in df["image"]:
        a = np.asarray(Image.open(cfg["paths"]["processed"] / path).convert("RGB"), dtype=np.float32) / 255.0
        images.append(Image.fromarray((shades_of_gray(a) * 255).round().astype(np.uint8)))
    for name in backbones:
        print(f"{name} (Shades of Gray) on {device}")
        model = create_backbone(name, True, cfg["image"]["crop"]).eval().to(device)
        test = embed(model, images, test_tf, device, cfg["features"]["batch_size"])
        flip = embed(model, images, flip_tf, device, cfg["features"]["batch_size"])
        path = cfg["paths"]["features"] / f"{backbone_key(name)}__sog.npz"
        np.savez(path, ids=df["patient_id"].to_numpy(str), test=test, flip=flip, aug=test[None],
                 backbone=name, pretrained=True)
        print(f"  saved {path}")
        del model
        if device.type == "mps":
            torch.mps.empty_cache()


# ---------------------------------------------------------------- feature transforms (fitted on training rows)

class GroupCenter:
    def fit(self, X, g):
        self.glob = (X.mean(0), X.std(0) + 1e-6)
        self.stats = {v: (X[g == v].mean(0), X[g == v].std(0) + 1e-6) for v in np.unique(g) if (g == v).sum() > 5}
        return self

    def transform(self, X, g):
        out = np.empty_like(X)
        for v in np.unique(g):
            mu, sd = self.stats.get(v, self.glob)
            out[g == v] = (X[g == v] - mu) / sd
        return out


def phone_probe_auc(F_tr, g_tr, F_te, g_te):
    """Linear probe: identify Xiaomi vs Samsung on known-phone patients."""
    k_tr, k_te = np.isin(g_tr, list(PHONES)), np.isin(g_te, list(PHONES))
    probe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.1, max_iter=5000))])
    probe.fit(F_tr[k_tr], g_tr[k_tr] == "2201117SG")
    return roc_auc_score(g_te[k_te] == "2201117SG", probe.predict_proba(F_te[k_te])[:, 1])


# ---------------------------------------------------------------- adversarial network

def fit_adv(X, y, g, X_te_list, lam, seed, aug=None, epochs=100):
    import torch
    from torch import nn

    class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, l):
            ctx.l = l
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad):
            return -ctx.l * grad, None

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    levels = sorted(np.unique(g))
    gi = np.array([levels.index(v) for v in g])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    enc = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Dropout(0.3))
    head = nn.Linear(128, 1)
    adv = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(levels)))
    params = [*enc.parameters(), *head.parameters(), *adv.parameters()]
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-3)
    pos_w = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), dtype=torch.float32)
    bce, ce = nn.BCEWithLogitsLoss(pos_weight=pos_w), nn.CrossEntropyLoss()
    yt, gt = torch.tensor(y, dtype=torch.float32), torch.tensor(gi)
    steps, n = 0, len(y)
    total = epochs * int(np.ceil(n / 64))
    for ep in range(epochs):
        view = X if aug is None else aug[rng.integers(0, len(aug))]
        Xt = torch.tensor((view - mu) / sd, dtype=torch.float32)
        enc.train()
        for idx in torch.randperm(n).split(64):
            ramp = 2 / (1 + np.exp(-10 * steps / total)) - 1  # DANN schedule: 0 -> 1
            h = enc(Xt[idx])
            loss = bce(head(h).squeeze(1), yt[idx])
            if lam > 0:
                loss = loss + ce(adv(GradReverse.apply(h, lam * ramp)), gt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
    enc.eval()
    with torch.no_grad():
        out = []
        for Xe in X_te_list:
            h = enc(torch.tensor((Xe - mu) / sd, dtype=torch.float32))
            out.append((torch.sigmoid(head(h).squeeze(1)).numpy(), h.numpy()))
    return out


# ---------------------------------------------------------------- run

def fit_lr(X_tr, y, X_te_list, cv, w=None):
    est = Pipeline([("sc", StandardScaler()), ("clf", lr())])
    gs = GridSearchCV(est, C_GRID, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(X_tr, y, **({"clf__sample_weight": w} if w is not None else {}))
    return [gs.predict_proba(X)[:, 1] for X in X_te_list]


def cmd_run(cfg, backbones, methods, repeats):
    df, _ = load_data(cfg)
    y_all, g_all = df["y"].to_numpy(), groups(df)
    levels = sorted(np.unique(g_all))
    out, probe_rows = {}, []
    for bb in backbones:
        F, F_flip, aug = load_features(cfg, bb, df)
        S, S_flip = (load_features(cfg, f"{bb}__sog", df)[:2]
                     if any(m.startswith("sog") for m in methods) else (None, None))
        for r, k, tr, te in outer_splits(cfg, df, repeats):
            cv = inner_cv(cfg, r, k)
            y, g, gte = y_all[tr], g_all[tr], g_all[te]
            w = balance_weights(y, g)
            res, feats = {}, {}
            for m in methods:
                if m in ("base", "reweight"):
                    p, pf = fit_lr(F[tr], y, [F[te], F_flip[te]], cv, w if m == "reweight" else None)
                    feats["base"] = (F[tr], F[te])
                elif m == "sog":
                    p, pf = fit_lr(S[tr], y, [S[te], S_flip[te]], cv)
                    feats[m] = (S[tr], S[te])
                elif m == "center":
                    c = GroupCenter().fit(F[tr], g)
                    p, pf = fit_lr(c.transform(F[tr], g), y, [c.transform(F[te], gte), c.transform(F_flip[te], gte)], cv)
                    feats[m] = (c.transform(F[tr], g), c.transform(F[te], gte))
                elif m in ("leace", "leace_rw", "sog_leace_rw"):
                    A, A_flip = (S, S_flip) if m.startswith("sog") else (F, F_flip)
                    e = Leace().fit(A[tr], onehot(g, levels)[:, 1:])
                    p, pf = fit_lr(e.transform(A[tr]), y, [e.transform(A[te]), e.transform(A_flip[te])], cv,
                                   w if m.endswith("_rw") else None)
                    feats["leace" if m == "leace" else m] = (e.transform(A[tr]), e.transform(A[te]))
                elif m == "adjust":
                    O_tr, O_mean = onehot(g, levels), onehot(g, levels).mean(0)
                    O_te = np.repeat(O_mean[None], len(te), 0)
                    p, pf = fit_lr(np.hstack([F[tr], O_tr]), y, [np.hstack([F[te], O_te]), np.hstack([F_flip[te], O_te])], cv)
                elif m.startswith("adv") or m == "mlp":
                    lam = 0.0 if m == "mlp" else float(m[3:])
                    (p, h_te), (pf, _), (_, h_tr) = fit_adv(F[tr], y, g, [F[te], F_flip[te], F[tr]], lam,
                                                            cfg["seed"] + 100 * r + k, aug=aug[:, tr])
                    feats[m] = (h_tr, h_te)
                else:
                    raise ValueError(m)
                res[m] = (p + pf) / 2
            for m, p in res.items():
                out.setdefault(f"mit_{m}_{bb}", []).append(oof_frame(df, te, r, k, p, np.nan))
            for m, (A_tr, A_te) in feats.items():
                probe_rows.append({"backbone": bb, "method": m, "repeat": r, "fold": k,
                                   "phone_id_auc": phone_probe_auc(A_tr, g, A_te, gte)})
            print(f"{bb} repeat {r} fold {k}: " + " ".join(f"{m}={roc_auc_score(y_all[te], p):.3f}"
                                                          for m, p in res.items()), flush=True)
    for name, frames in out.items():
        save_oof(cfg, name, frames)
    probes = pd.DataFrame(probe_rows)
    path = cfg["paths"]["metrics"] / "mitigation_phone_probe.csv"
    probes.to_csv(path, index=False)
    print(probes.groupby(["backbone", "method"])["phone_id_auc"].mean().round(3).to_string())


# ---------------------------------------------------------------- cross-phone transfer

def cmd_cross(cfg, backbones, n_boot):
    df, _ = load_data(cfg)
    y_all, g_all = df["y"].to_numpy(), groups(df)
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for bb in backbones:
        F, F_flip, _ = load_features(cfg, bb, df)
        S, S_flip, _ = load_features(cfg, f"{bb}__sog", df)
        for a, b in [("SM-G998B", "2201117SG"), ("2201117SG", "SM-G998B")]:
            tr, te = g_all == a, g_all == b
            sets = {"base": (F, F_flip, False), "sog": (S, S_flip, False), "center": (F, F_flip, True),
                    "sog_center": (S, S_flip, True)}
            for m, (A, A_flip, center) in sets.items():
                A_tr, A_te, A_tf = A[tr], A[te], A_flip[te]
                if center:  # each phone standardised with its own statistics (unsupervised, no labels used)
                    mu_a, sd_a = A_tr.mean(0), A_tr.std(0) + 1e-6
                    mu_b, sd_b = A_te.mean(0), A_te.std(0) + 1e-6
                    A_tr, A_te, A_tf = (A_tr - mu_a) / sd_a, (A_te - mu_b) / sd_b, (A_tf - mu_b) / sd_b
                p, pf = fit_lr(A_tr, y_all[tr], [A_te, A_tf], 3)  # same inner CV as site_shortcut cross
                p = (p + pf) / 2
                lo, hi = bootstrap_auc(y_all[te], p, n_boot, rng)
                rows.append({"backbone": bb, "method": m, "train": PHONES[a], "test": PHONES[b],
                             "auc": roc_auc_score(y_all[te], p), "auc_lo": lo, "auc_hi": hi})
                print(f"  {bb:26s} {m:11s} {PHONES[a]:>8s} -> {PHONES[b]:8s} AUC {rows[-1]['auc']:.3f} [{lo:.3f}, {hi:.3f}]")
    res = pd.DataFrame(rows)
    res.to_csv(cfg["paths"]["metrics"] / "mitigation_cross.csv", index=False)


# ---------------------------------------------------------------- audit metrics

def shortcut_metrics(oof, g_map, w_map, n_boot, rng):
    """Per fold: plain, deconfounded and within-phone AUC; per repeat: the same on pooled predictions."""
    oof = oof.assign(g=oof["patient_id"].map(g_map), w=oof["patient_id"].map(w_map))
    folds, reps = [], []
    for (r, k), f in oof.groupby(["repeat", "fold"]):
        folds.append({"repeat": r, "fold": k, "auc": roc_auc_score(f["y"], f["p"]),
                      "auc_deconf": roc_auc_score(f["y"], f["p"], sample_weight=f["w"]),
                      "auc_adjusted": adjusted_auc(f["y"], f["p"], f["g"])})
    for r, f in oof.groupby("repeat"):
        y, p, w = f["y"].to_numpy(), f["p"].to_numpy(), f["w"].to_numpy()
        row = {"repeat": r, "auc": roc_auc_score(y, p), "auc_deconf": roc_auc_score(y, p, sample_weight=w)}
        boot, n = [], len(y)
        while len(boot) < n_boot:
            i = rng.integers(0, n, n)
            if 0 < y[i].sum() < n:
                boot.append(roc_auc_score(y[i], p[i], sample_weight=w[i]))
        row["auc_deconf_lo"], row["auc_deconf_hi"] = np.percentile(boot, [2.5, 97.5])
        row["auc_adjusted"] = adjusted_auc(y, p, f["g"].to_numpy())
        for code, name in PHONES.items():
            s = f[f["g"] == code]
            row[f"auc_{name}"] = roc_auc_score(s["y"], s["p"])
        reps.append(row)
    reps = pd.DataFrame(reps)
    reps["auc_within"] = reps[[f"auc_{n}" for n in PHONES.values()]].mean(1)
    return pd.DataFrame(folds), reps


def cmd_audit(cfg_main, cfg_mit, main_models, pairs, n_boot):
    df, _ = load_data(cfg_main)
    g = groups(df)
    g_map = dict(zip(df["patient_id"], g))
    w_map = dict(zip(df["patient_id"], balance_weights(df["y"].to_numpy(), g)))
    n = len(df)
    sources = [(m, cfg_main) for m in main_models if (cfg_main["paths"]["oof"] / f"{m}.csv").exists()]
    sources += [(p.stem, cfg_mit) for p in sorted(cfg_mit["paths"]["oof"].glob("mit_*.csv"))]
    sources += [(p.stem, cfg_mit) for p in sorted(cfg_mit["paths"]["oof"].glob("ftmit_*.csv"))]
    ft_path = cfg_main["paths"]["metrics"] / "mitigation_finetune_probe.csv"
    ft_probes = pd.read_csv(ft_path) if ft_path.exists() else pd.DataFrame()
    probes_path = cfg_main["paths"]["metrics"] / "mitigation_phone_probe.csv"
    probes = pd.read_csv(probes_path) if probes_path.exists() else pd.DataFrame()
    summary, fold_tab = [], {}
    for name, c in sources:
        folds, reps = shortcut_metrics(load_oof(c, name), g_map, w_map, n_boot, np.random.default_rng(cfg_main["seed"]))
        fold_tab[name] = folds.set_index(["repeat", "fold"])
        row = {"model": name, "repeats": len(reps)}
        for col in ["auc", "auc_deconf", "auc_deconf_lo", "auc_deconf_hi", "auc_adjusted", "auc_Samsung", "auc_Xiaomi",
                    "auc_within"]:
            row[col], row[col + "_sd"] = reps[col].mean(), reps[col].std(ddof=1) if len(reps) > 1 else np.nan
        if name.startswith("mit_") and not probes.empty:
            meth, bb = name[4:].split("_", 1)[0], None
            for b in probes["backbone"].unique():
                if name.endswith("_" + b):
                    meth, bb = name[4:-len(b) - 1], b
            hit = probes[(probes["backbone"] == bb) & (probes["method"] == ("base" if meth == "reweight" else meth))]
            row["phone_id_auc"] = hit["phone_id_auc"].mean() if len(hit) else np.nan
        if name.startswith("ftmit_") and not ft_probes.empty:
            meth, key = name[6:].split("_", 1)
            hit = ft_probes[(ft_probes["method"] == meth) & (ft_probes["model"] == key)]
            row["phone_id_auc"] = hit["phone_id_auc"].mean() if len(hit) else np.nan
        summary.append(row)
    summary = pd.DataFrame(summary)

    tests = []
    for name in fold_tab:
        if name.startswith("ftmit_") and not name.startswith("ftmit_erm_"):
            ref = "ftmit_erm_" + name.split("_", 2)[2]  # end-to-end methods vs ERM with the same selection rule
        elif name.startswith("mit_") and not name.startswith("mit_base_"):
            ref = "mit_base_" + next(b for b in ("vit_small_patch14_dinov2", "convnext_tiny") if name.endswith(b))
        else:
            continue
        if ref not in fold_tab:
            continue
        a, b = fold_tab[name], fold_tab[ref]
        common = a.index.intersection(b.index)
        for metric in ("auc", "auc_deconf", "auc_adjusted"):
            d, t, p = corrected_ttest(a.loc[common, metric] - b.loc[common, metric], n * 4 / 5, n / 5, one_sided=False)
            tests.append({"model": name, "vs": ref, "metric": metric, "delta": d, "t": t, "p_two_sided": p})
    for a_name, b_name in pairs:  # main-model comparisons on the deconfounded metric
        if a_name not in fold_tab or b_name not in fold_tab:
            continue
        a, b = fold_tab[a_name], fold_tab[b_name]
        common = a.index.intersection(b.index)
        if len(common) < 2:
            continue
        for metric in ("auc", "auc_deconf", "auc_adjusted"):
            d, t, p = corrected_ttest(a.loc[common, metric] - b.loc[common, metric], n * 4 / 5, n / 5, one_sided=False)
            tests.append({"model": a_name, "vs": b_name, "metric": metric, "delta": d, "t": t, "p_two_sided": p})
    tests = pd.DataFrame(tests)
    out = cfg_main["paths"]["metrics"]
    summary.to_csv(out / "mitigation_summary.csv", index=False)
    tests.to_csv(out / "mitigation_tests.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.drop(columns=[c for c in summary if c.endswith("_sd")]).round(3).to_string(index=False))
        print()
        print(tests.round(4).to_string(index=False))


MAIN_MODELS = ["phone", "size", "lowlevel", "sym_lr", "sym_lgbm", "img_lr_vit_small_patch14_dinov2",
               "img_lr_convnext_tiny", "stack_vit_small_patch14_dinov2", "late_vit_small_patch14_dinov2",
               "early_vit_small_patch14_dinov2", "gated_vit_small_patch14_dinov2", "gated_convnext_tiny",
               "ft_densenet121", "ft_mobilenetv3_large_100", "ft_convnext_tiny"]
DINO = "vit_small_patch14_dinov2"
PAIRS = [(f"img_lr_{DINO}", "sym_lgbm"), (f"img_lr_{DINO}", "lowlevel"), (f"stack_{DINO}", f"img_lr_{DINO}"),
         (f"gated_{DINO}", f"img_lr_{DINO}"), ("gated_convnext_tiny", "img_lr_convnext_tiny"), ("sym_lgbm", "phone"),
         (f"img_lr_{DINO}", "phone")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "run", "cross"):
        s = sub.add_parser(name)
        s.add_argument("--backbones", nargs="*", help="default: config features.backbones")
        if name == "run":
            s.add_argument("--methods", nargs="*",
                           default=LR_METHODS + ["mlp"] + [f"adv{l:g}" for l in ADV_LAMBDAS if l > 0])
            s.add_argument("--repeats", type=int)
        if name == "cross":
            s.add_argument("--n-boot", type=int, default=1000)
    s = sub.add_parser("audit")
    s.add_argument("--models", nargs="*", default=MAIN_MODELS)
    s.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    cfg["paths"]["metrics"].mkdir(parents=True, exist_ok=True)
    names = getattr(args, "backbones", None) or cfg["features"]["backbones"]
    if args.cmd == "extract":
        cmd_extract(cfg, names)
    elif args.cmd == "run":
        cmd_run(mit_cfg(cfg), [backbone_key(b) for b in names], args.methods, args.repeats)
    elif args.cmd == "cross":
        cmd_cross(cfg, [backbone_key(b) for b in names], args.n_boot)
    else:
        cmd_audit(load_config(args.config), mit_cfg(load_config(args.config)), args.models, PAIRS, args.n_boot)


if __name__ == "__main__":
    main()
