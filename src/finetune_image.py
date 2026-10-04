"""E1: reproduce the published image-only baselines by end-to-end fine-tuning on the Apple GPU.

Cost compromise (state it in the Methods): full nested inner CV is too slow for fine-tuning, so each outer
training part holds out a stratified 15% validation split for early stopping and the threshold.
Frozen-feature models (baselines.py, fusion.py) use the full nested inner CV.

Usage:
    python -m src.finetune_image --model densenet121              # -> results/oof/ft_densenet121.csv
    python -m src.finetune_image --model densenet121 --repeats 1  # quicker: 5 folds only
    python -m src.finetune_image --model convnext_tiny.fb_in22k_ft_in1k --save-full   # model for Grad-CAM
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .common import (add_config_arg, backbone_key, choose_threshold, load_config, load_data, oof_frame,
                     outer_splits, save_oof, set_seed, torch_device)
from .extract_features import create_backbone, transforms


def batches(images, idx, tf, bs, y=None, shuffle=False, rng=None):
    order = rng.permutation(idx) if shuffle else idx
    for b in range(0, len(order), bs):
        j = order[b:b + bs]
        x = torch.stack([tf(images[i]) for i in j])
        yield (x, torch.as_tensor(y[j], dtype=torch.float32)) if y is not None else x


@torch.no_grad()
def predict(model, images, idx, tfs, bs, device):
    model.eval()
    out = []
    for tf in tfs:  # TTA: mean over the given transforms
        ps = [torch.sigmoid(model(x.to(device)).squeeze(-1)).float().cpu().numpy() for x in batches(images, idx, tf, bs)]
        out.append(np.concatenate(ps))
    return np.mean(out, axis=0)


def fit(name, images, y, fit_idx, val_idx, ft, crop, device, pretrained, seed, epochs=None):
    """Train with early stopping on val AUC (or a fixed number of epochs when val_idx is None)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    test_tf, flip_tf, train_tf = transforms(crop)
    model = create_backbone(name, pretrained, crop, num_classes=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=ft["lr"], weight_decay=ft["weight_decay"])
    n_ep = epochs or ft["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    pos_weight = torch.tensor(float((y[fit_idx] == 0).sum() / max((y[fit_idx] == 1).sum(), 1)),
                              dtype=torch.float32, device=device)
    best, best_auc, best_ep, wait = None, -1.0, n_ep, 0
    for ep in range(1, n_ep + 1):
        model.train()
        for x, t in batches(images, fit_idx, train_tf, ft["batch_size"], y, shuffle=True, rng=rng):
            loss = F.binary_cross_entropy_with_logits(model(x.to(device)).squeeze(-1), t.to(device),
                                                      pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if val_idx is None:
            continue
        auc = roc_auc_score(y[val_idx], predict(model, images, val_idx, [test_tf], ft["batch_size"], device))
        if auc > best_auc:
            best, best_auc, best_ep, wait = copy.deepcopy(model.state_dict()), auc, ep, 0
        else:
            wait += 1
            if wait >= ft["patience"]:
                break
    if best is not None:
        model.load_state_dict(best)
    return model, best_ep, best_auc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument("--model", required=True, help="timm model name")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--no-pretrained", action="store_true", help="random weights; smoke tests only")
    ap.add_argument("--save-full", action="store_true",
                    help="after CV, train on all patients (median best epoch) and save it for Grad-CAM")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    df, _ = load_data(cfg)
    device, ft, crop = torch_device(), cfg["finetune"], cfg["image"]["crop"]
    test_tf, flip_tf, _ = transforms(crop)
    images = [Image.open(cfg["paths"]["processed"] / p).convert("RGB") for p in df["image"]]
    for im in images:
        im.load()
    y = df["y"].to_numpy()
    key = backbone_key(args.model)
    print(f"Fine-tuning {args.model} on {device}")

    frames, best_eps = [], []
    for r, k, tr, te in outer_splits(cfg, df, args.repeats):
        seed = cfg["seed"] + 100 * r + k
        fit_idx, val_idx = train_test_split(tr, test_size=ft["val_frac"], stratify=y[tr], random_state=seed)
        model, ep, vauc = fit(args.model, images, y, fit_idx, val_idx, ft, crop, device, not args.no_pretrained, seed)
        thr = choose_threshold(y[val_idx], predict(model, images, val_idx, [test_tf], ft["batch_size"], device))
        p = predict(model, images, te, [test_tf, flip_tf], ft["batch_size"], device)
        best_eps.append(ep)
        print(f"repeat {r} fold {k}: best epoch {ep}, val AUC {vauc:.3f}, test AUC {roc_auc_score(y[te], p):.3f}")
        frames.append(oof_frame(df, te, r, k, p, thr, epochs=ep))
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    save_oof(cfg, f"ft_{key}", frames)

    if args.save_full:
        n_ep = int(np.median(best_eps))
        all_idx = np.arange(len(df))
        model, _, _ = fit(args.model, images, y, all_idx, None, ft, crop, device, not args.no_pretrained,
                          cfg["seed"], epochs=n_ep)
        out = cfg["paths"]["models"]
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"name": args.model, "state_dict": model.cpu().state_dict(), "epochs": n_ep},
                   out / f"{key}_full.pt")
        print(f"Saved {out / f'{key}_full.pt'} (trained on all patients for {n_ep} epochs; for explanation only)")


if __name__ == "__main__":
    main()
