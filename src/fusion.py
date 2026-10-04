"""Proposed gated cross-modal fusion (group D) on cached image embeddings, with nested inner CV.

Per outer fold:
  1. For each grid point (lr, dropout, aux_weight), train on each inner fold for epochs_max epochs and record
     the validation AUC after every epoch.
  2. Pick the grid point and epoch count with the best mean (smoothed) inner AUC; pick the threshold from
     inner out-of-fold predictions at that epoch.
  3. Retrain on the full outer training part for that many epochs (same cosine schedule) and predict the
     test fold with flip TTA. The gate value g, and the image-only / symptom-only auxiliary probabilities,
     are saved with every prediction.

Usage:
    python -m src.fusion --backbone convnext_tiny                     # -> gated_convnext_tiny
    python -m src.fusion --backbone convnext_tiny --fusion concat     # E3: no gate
    python -m src.fusion --backbone convnext_tiny --hard-labels        # E3: no soft labels
    python -m src.fusion --backbone convnext_tiny --aux 0              # E3: no auxiliary heads
    python -m src.fusion --backbone convnext_tiny --modality-dropout 0 # E3: no modality dropout
    python -m src.fusion --backbone convnext_tiny --shuffle-images     # E7: shuffled-image control
    python -m src.fusion --backbone convnext_tiny --drop-symptoms fever cough --tag noFeverCough   # E5
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .common import (TAG_SEP, add_config_arg, backbone_key, choose_threshold, inner_cv, load_config, load_data,
                     load_features, oof_frame, outer_splits, save_oof, set_seed, tab_array, tab_transformer)


class FusionNet(nn.Module):
    def __init__(self, d_img, d_tab, proj=128, dropout=0.3, fusion="gated"):
        super().__init__()
        self.fusion = fusion
        self.img = nn.Sequential(nn.LayerNorm(d_img), nn.Dropout(dropout), nn.Linear(d_img, proj), nn.GELU())
        self.sym = nn.Sequential(nn.Linear(d_tab, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, proj), nn.GELU())
        if fusion == "gated":
            self.gate = nn.Linear(2 * proj, 1)
        elif fusion == "film":
            self.film = nn.Linear(proj, 2 * proj)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * proj if fusion == "concat" else proj, 1))
        self.aux_img = nn.Linear(proj, 1)
        self.aux_sym = nn.Linear(proj, 1)

    def forward(self, xi, xs, keep_img=None, keep_sym=None):
        hi, hs = self.img(xi), self.sym(xs)
        if keep_img is not None:
            hi, hs = hi * keep_img, hs * keep_sym
        g = None
        if self.fusion == "gated":
            g = torch.sigmoid(self.gate(torch.cat([hi, hs], -1)))
            z = g * hi + (1 - g) * hs
        elif self.fusion == "concat":
            z = torch.cat([hi, hs], -1)
        else:  # film: symptoms scale and shift the image representation
            gamma, beta = self.film(hs).chunk(2, -1)
            z = hi * (1 + gamma) + beta
        return self.head(z).squeeze(-1), self.aux_img(hi).squeeze(-1), self.aux_sym(hs).squeeze(-1), g


@torch.no_grad()
def predict(model, xi, xs):
    model.eval()
    logit, ai, as_, g = model(torch.as_tensor(xi, dtype=torch.float32), torch.as_tensor(xs, dtype=torch.float32))
    g = g.squeeze(-1).numpy() if g is not None else np.full(len(xi), np.nan)
    return torch.sigmoid(logit).numpy(), torch.sigmoid(ai).numpy(), torch.sigmoid(as_).numpy(), g


def train(hp, opts, Xi_aug, Xs, y_target, y_hard, epochs, seed, val=None):
    """Train for `epochs` epochs of a cosine schedule spanning epochs_max. Returns (model, val probs per epoch)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    K, n, d_img = Xi_aug.shape
    model = FusionNet(d_img, Xs.shape[1], opts["proj_dim"], hp["dropout"], opts["fusion"])
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=opts["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=opts["epochs_max"])
    n_pos = max(y_hard.sum(), 1)
    w = torch.as_tensor(np.where(y_hard == 1, n / (2 * n_pos), n / (2 * max(n - n_pos, 1))), dtype=torch.float32)
    yt = torch.as_tensor(y_target, dtype=torch.float32)
    xs_all = torch.as_tensor(Xs, dtype=torch.float32)
    md, bs, aux = opts["modality_dropout"], opts["batch_size"], hp["aux_weight"]
    history = []
    for _ in range(epochs):
        model.train()
        xi_all = torch.as_tensor(Xi_aug[rng.integers(K, size=n), np.arange(n)], dtype=torch.float32)
        perm = rng.permutation(n)
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            if len(idx) < 2:
                continue
            u = torch.as_tensor(rng.random(len(idx)), dtype=torch.float32).unsqueeze(-1)
            keep_img, keep_sym = (u >= md).float(), (u <= 1 - md).float()  # never drop both
            logit, ai, as_, _ = model(xi_all[idx], xs_all[idx], keep_img, keep_sym)
            bce = lambda z: (w[idx] * F.binary_cross_entropy_with_logits(z, yt[idx], reduction="none")).mean()  # noqa
            loss = bce(logit) + aux * (bce(ai) + bce(as_))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if val is not None:
            history.append(predict(model, *val)[0])
    return model, history


def smooth(curve):
    pad = np.pad(curve, 1, mode="edge")
    return (pad[:-2] + pad[1:-1] + pad[2:]) / 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument("--backbone", required=True, help="feature cache name, e.g. convnext_tiny")
    ap.add_argument("--fusion", default="gated", choices=["gated", "concat", "film"])
    ap.add_argument("--hard-labels", action="store_true", help="train on 0/1 labels instead of vote share")
    ap.add_argument("--aux", type=float, help="fix the auxiliary loss weight instead of searching it")
    ap.add_argument("--modality-dropout", type=float, help="override config fusion.modality_dropout")
    ap.add_argument("--shuffle-images", action="store_true", help="E7 control: break image-patient pairing")
    ap.add_argument("--drop-symptoms", nargs="*", default=[])
    ap.add_argument("--tag", default="")
    ap.add_argument("--name", help="output name (default: <fusion>_<backbone>[__tag])")
    ap.add_argument("--repeats", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    torch.set_num_threads(4)
    df, cols = load_data(cfg)
    fc = cfg["fusion"]
    opts = {"proj_dim": fc["proj_dim"], "fusion": args.fusion, "weight_decay": fc["weight_decay"],
            "epochs_max": fc["epochs_max"], "batch_size": fc["batch_size"],
            "modality_dropout": fc["modality_dropout"] if args.modality_dropout is None else args.modality_dropout}
    grid = dict(fc["grid"])
    if args.aux is not None:
        grid["aux_weight"] = [args.aux]
    grid_points = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]

    bb = backbone_key(args.backbone)
    Fi, Fi_flip, Fi_aug = load_features(cfg, bb, df)
    if args.shuffle_images:
        perm = np.random.default_rng(cfg["seed"]).permutation(len(df))
        Fi, Fi_flip, Fi_aug = Fi[perm], Fi_flip[perm], Fi_aug[:, perm]
    X_tab, n_bin, _ = tab_array(df, cols, args.drop_symptoms)
    y = df["y"].to_numpy()
    y_target = y.astype(float) if args.hard_labels else df["y_soft"].to_numpy()
    name = args.name or f"{args.fusion}_{bb}" + (f"{TAG_SEP}{args.tag}" if args.tag else "")
    E = opts["epochs_max"]

    frames = []
    for r, k, tr, te in outer_splits(cfg, df, args.repeats):
        seed = cfg["seed"] + 100 * r + k
        best = None
        for hp in grid_points:
            curves, oof_ep = [], np.zeros((E, len(tr)))
            for a, b in inner_cv(cfg, r, k).split(tr, y[tr]):
                ia, ib = tr[a], tr[b]
                prep = tab_transformer(n_bin).fit(X_tab[ia])
                _, hist = train(hp, opts, Fi_aug[:, ia], prep.transform(X_tab[ia]), y_target[ia], y[ia], E, seed,
                                val=(Fi[ib], prep.transform(X_tab[ib])))
                oof_ep[:, b] = np.array(hist)
                curves.append([roc_auc_score(y[ib], p) for p in hist])
            sm = smooth(np.mean(curves, axis=0))
            ep = int(np.argmax(sm)) + 1
            if best is None or sm[ep - 1] > best[0]:
                best = (sm[ep - 1], hp, ep, choose_threshold(y[tr], oof_ep[ep - 1]))
        score, hp, ep, thr = best
        prep = tab_transformer(n_bin).fit(X_tab[tr])
        model, _ = train(hp, opts, Fi_aug[:, tr], prep.transform(X_tab[tr]), y_target[tr], y[tr], ep, seed)
        xs_te = prep.transform(X_tab[te])
        a, b = predict(model, Fi[te], xs_te), predict(model, Fi_flip[te], xs_te)
        p, p_img, p_sym, g = [(u + v) / 2 for u, v in zip(a, b)]
        print(f"repeat {r} fold {k}: inner AUC {score:.3f}, epochs {ep}, {hp}, test AUC {roc_auc_score(y[te], p):.3f}")
        frames.append(oof_frame(df, te, r, k, p, thr, p_img=p_img, p_sym=p_sym, g=g, epochs=ep,
                                hparams=json.dumps(hp)))
    save_oof(cfg, name, frames)


if __name__ == "__main__":
    main()
