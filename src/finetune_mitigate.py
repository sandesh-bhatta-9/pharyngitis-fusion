"""End-to-end shortcut mitigation: fine-tuned CNN with ERM, GroupDRO or a gradient-reversal phone head.

Same outer folds (one repeat), image settings and optimiser as finetune_image.py. Early stopping and the epoch choice
use the deconfounded AUC on the 15% validation split, so that model selection does not reward the phone shortcut.
`erm` (plain cross-entropy) with the same selection rule is the control.

  erm       class-weighted binary cross-entropy
  gdro      group DRO (Sagawa et al., ICLR 2020) over the phone x label groups, step size eta = 0.01
  adv       gradient-reversal phone head on the pooled features (Ganin et al. 2016), lambda ramped to 1

After training, a linear probe identifies the phone from the pooled features of the test patients (trained on the
training patients' features). Outputs: results/mitigation/oof/ftmit_<method>_<model>.csv and
results/metrics/mitigation_finetune_probe.csv.

Usage:
    python -m src.finetune_mitigate --method gdro --model convnext_tiny.fb_in22k_ft_in1k --repeats 1
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

from shortcut_audit import balance_weights, site_probe_auc
from .common import (add_config_arg, backbone_key, load_config, load_data, oof_frame, outer_splits, save_oof,
                     set_seed, torch_device)
from .extract_features import create_backbone, transforms
from .finetune_image import batches


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class Net(nn.Module):
    def __init__(self, name, crop, n_groups):
        super().__init__()
        self.backbone = create_backbone(name, True, crop, num_classes=0)
        d = self.backbone.num_features
        self.head = nn.Linear(d, 1)
        self.adv = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, n_groups))
        self.lam = 0.0

    def forward(self, x, with_phone=False):
        h = self.backbone(x)
        logit = self.head(h).squeeze(-1)
        return (logit, self.adv(GradReverse.apply(h, self.lam))) if with_phone else logit


@torch.no_grad()
def run(model, images, idx, tfs, bs, device, features=False):
    model.eval()
    out = []
    for tf in tfs:
        chunks = []
        for x in batches(images, idx, tf, bs):
            x = x.to(device)
            chunks.append((model.backbone(x) if features else torch.sigmoid(model(x))).float().cpu().numpy())
        out.append(np.concatenate(chunks))
    return np.mean(out, axis=0)


def fit(method, name, images, y, phone, fit_idx, val_idx, ft, crop, device, seed, eta=0.01, lam_max=1.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    test_tf, _, train_tf = transforms(crop)
    levels = sorted(np.unique(phone))
    ph = np.array([levels.index(v) for v in phone])
    groups = ph * 2 + y  # phone x label
    model = Net(name, crop, len(levels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=ft["lr"], weight_decay=ft["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ft["epochs"])
    pos_weight = torch.tensor(float((y[fit_idx] == 0).sum() / max((y[fit_idx] == 1).sum(), 1)), device=device)
    q = torch.ones(2 * len(levels), device=device) / (2 * len(levels))
    w_val = balance_weights(y[val_idx], phone[val_idx])
    steps_per_ep = int(np.ceil(len(fit_idx) / ft["batch_size"]))
    best, best_auc, best_ep, wait, step = None, -1.0, ft["epochs"], 0, 0
    for ep in range(1, ft["epochs"] + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(fit_idx)
        for b in range(0, len(order), ft["batch_size"]):
            j = order[b:b + ft["batch_size"]]
            x = torch.stack([train_tf(images[i]) for i in j]).to(device)
            t = torch.as_tensor(y[j], dtype=torch.float32, device=device)
            if method == "adv":
                model.lam = lam_max * (2 / (1 + np.exp(-10 * step / (ft["epochs"] * steps_per_ep))) - 1)
                logit, ph_logit = model(x, with_phone=True)
                loss = F.binary_cross_entropy_with_logits(logit, t, pos_weight=pos_weight) + \
                    F.cross_entropy(ph_logit, torch.as_tensor(ph[j], device=device))
            elif method == "gdro":
                per = F.binary_cross_entropy_with_logits(model(x), t, reduction="none")
                gj = torch.as_tensor(groups[j], device=device)
                g_loss = torch.zeros_like(q)
                present = torch.zeros_like(q)
                for g in gj.unique():
                    g_loss[g] = per[gj == g].mean()
                    present[g] = 1.0
                with torch.no_grad():
                    q = q * torch.exp(eta * g_loss * present)
                    q = q / q.sum()
                loss = (q * g_loss).sum()
            else:
                loss = F.binary_cross_entropy_with_logits(model(x), t, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        sched.step()
        p_val = run(model, images, val_idx, [test_tf], ft["batch_size"], device)
        auc = roc_auc_score(y[val_idx], p_val, sample_weight=w_val)
        print(f"    epoch {ep} loss {loss.item():.3f} val deconf AUC {auc:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if auc > best_auc:
            best = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            best_auc, best_ep, wait = auc, ep, 0
        else:
            wait += 1
            if wait >= ft["patience"]:
                break
    model.load_state_dict(best)
    return model, best_ep, best_auc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument("--method", choices=["erm", "gdro", "adv"], required=True)
    ap.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    torch.set_num_threads(args.threads)
    df, _ = load_data(cfg)
    device, ft, crop = torch_device(), cfg["finetune"], cfg["image"]["crop"]
    test_tf, flip_tf, _ = transforms(crop)
    images = [Image.open(cfg["paths"]["processed"] / p).convert("RGB") for p in df["image"]]
    for im in images:
        im.load()
    y = df["y"].to_numpy()
    phone = df["exif_phone"].fillna("unknown").to_numpy()
    key = backbone_key(args.model)
    print(f"Fine-tuning {args.model} with {args.method} on {device}")
    frames, probes = [], []
    for r, k, tr, te in outer_splits(cfg, df, args.repeats):
        seed = cfg["seed"] + 100 * r + k
        strat = np.char.add(y[tr].astype(str), phone[tr])
        fit_idx, val_idx = train_test_split(tr, test_size=ft["val_frac"], stratify=strat, random_state=seed)
        model, ep, vauc = fit(args.method, args.model, images, y, phone, fit_idx, val_idx, ft, crop, device, seed)
        p = run(model, images, te, [test_tf, flip_tf], ft["batch_size"], device)
        H_tr = run(model, images, tr, [test_tf], ft["batch_size"], device, features=True)
        H_te = run(model, images, te, [test_tf], ft["batch_size"], device, features=True)
        known_tr, known_te = phone[tr] != "unknown", phone[te] != "unknown"
        probe = site_probe_auc(H_tr[known_tr], phone[tr][known_tr], H_te[known_te], phone[te][known_te], "2201117SG")
        probes.append({"method": args.method, "model": key, "repeat": r, "fold": k, "best_epoch": ep,
                       "phone_id_auc": probe})
        print(f"repeat {r} fold {k}: epoch {ep}, val deconf AUC {vauc:.3f}, test AUC {roc_auc_score(y[te], p):.3f}, "
              f"phone probe {probe:.3f}", flush=True)
        frames.append(oof_frame(df, te, r, k, p, np.nan, epochs=ep))
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    cfg["paths"]["oof"] = cfg["paths"]["results"] / "mitigation" / "oof"
    save_oof(cfg, f"ftmit_{args.method}_{key}", frames)
    path = cfg["paths"]["metrics"] / "mitigation_finetune_probe.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[~((old["method"] == args.method) & (old["model"] == key))]
    pd.concat([old, pd.DataFrame(probes)], ignore_index=True).to_csv(path, index=False)


if __name__ == "__main__":
    main()
