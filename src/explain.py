"""E6 explainability.

Subcommands:
  shap      SHAP summary for LightGBM on symptoms + age + gender (fitted on all patients; explanation only).
  gate      Gate value g of the fusion model by vote agreement and by number of positive symptoms.
  gradcam   Grad-CAM grid from a CNN saved by `finetune_image --save-full`.

Usage:
    python -m src.explain shap
    python -m src.explain gate --model gated_convnext_tiny
    python -m src.explain gradcam --model convnext_tiny
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

from .common import add_config_arg, load_config, load_data, load_oof  # noqa: E402
from .figures import style, INK, SERIES  # noqa: E402


def run_shap(cfg, out):
    import warnings
    import shap
    warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")
    from lightgbm import LGBMClassifier
    df, cols = load_data(cfg)
    names = cols["binary"] + ["age"]
    X = df[names]
    model = LGBMClassifier(num_leaves=4, n_estimators=200, learning_rate=0.05, min_child_samples=10,
                           class_weight="balanced", verbose=-1, random_state=0).fit(X, df["y"])
    sv = shap.TreeExplainer(model).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
    imp.rename("mean_abs_shap").to_csv(cfg["paths"]["metrics"] / "shap_importance.csv")
    plt.figure()
    shap.summary_plot(sv, X, show=False, max_display=15, plot_size=(6.5, 5))
    plt.tight_layout()
    plt.savefig(out / "shap_summary.png", dpi=300)
    plt.savefig(out / "shap_summary.pdf")
    print(imp.head(10).round(4).to_string())


def run_gate(cfg, model, out):
    df, cols = load_data(cfg)
    oof = load_oof(cfg, model)
    if oof["g"].isna().all():
        raise SystemExit(f"{model} has no gate values (not a gated model)")
    g = oof.groupby("patient_id").agg(g=("g", "mean"), agreement=("agreement", "first"), y=("y", "first"))
    n_pos = df.set_index("patient_id")[cols["symptoms"]].sum(axis=1)
    g["n_symptoms"] = n_pos.loc[g.index]
    split, unan = g.loc[g["agreement"] < 1, "g"], g.loc[g["agreement"] == 1, "g"]
    mw = mannwhitneyu(split, unan) if len(split) and len(unan) else None
    rho = spearmanr(g["g"], g["n_symptoms"]).statistic
    summary = pd.DataFrame([{
        "model": model, "g_mean": g["g"].mean(), "g_sd": g["g"].std(),
        "g_split_median": split.median(), "g_unanimous_median": unan.median(),
        "mannwhitney_p": mw.pvalue if mw else np.nan, "spearman_g_vs_n_symptoms": rho}])
    summary.to_csv(cfg["paths"]["metrics"] / f"gate_{model}.csv", index=False)
    print(summary.round(4).to_string(index=False))

    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    bins = np.linspace(0, 1, 26)
    ax.hist(unan, bins=bins, histtype="step", lw=1.6, color=SERIES[0], label=f"Unanimous (n={len(unan)})")
    ax.hist(split, bins=bins, histtype="step", lw=1.6, color=SERIES[1], label=f"Split vote (n={len(split)})")
    ax.set_xlabel("Gate value g (1 = relies on image)")
    ax.set_ylabel("Patients")
    ax.legend(frameon=False)
    ax = axes[1]
    grp = g.groupby("n_symptoms")["g"]
    ax.plot(grp.median().index, grp.median().values, "o-", color=SERIES[0], ms=4)
    ax.set_xlabel("Number of positive symptoms")
    ax.set_ylabel("Median g")
    fig.tight_layout()
    fig.savefig(out / f"gate_{model}.png", dpi=300)
    fig.savefig(out / f"gate_{model}.pdf")


def run_gradcam(cfg, key, n_each, out):
    import torch
    from PIL import Image
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
    from .extract_features import create_backbone, transforms

    ckpt = torch.load(cfg["paths"]["models"] / f"{key}_full.pt", map_location="cpu")
    crop = cfg["image"]["crop"]
    model = create_backbone(ckpt["name"], False, crop, num_classes=1)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if hasattr(model, "stages"):          # ConvNeXt
        layer = model.stages[-1]
    elif hasattr(model, "features"):      # DenseNet
        layer = model.features
    elif hasattr(model, "layer4"):        # ResNet
        layer = model.layer4
    elif hasattr(model, "blocks"):        # MobileNetV3 / EfficientNet
        layer = model.blocks[-1]
    else:
        raise SystemExit("Unknown architecture for Grad-CAM; use a CNN (ConvNeXt, DenseNet, ResNet, MobileNet)")

    df, _ = load_data(cfg)
    test_tf, _, _ = transforms(crop)
    imgs = [Image.open(cfg["paths"]["processed"] / p).convert("RGB") for p in df["image"]]
    x = torch.stack([test_tf(im) for im in imgs])
    with torch.no_grad():
        p = torch.sigmoid(model(x).squeeze(-1)).numpy()
    # Most confident correct cases of each class.
    y = df["y"].to_numpy()
    pick = list(np.argsort(-np.where(y == 1, p, -1))[:n_each]) + list(np.argsort(np.where(y == 0, p, 2))[:n_each])
    cam = GradCAM(model=model, target_layers=[layer])
    maps = cam(input_tensor=x[pick], targets=[BinaryClassifierOutputTarget(1)] * len(pick))
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    style()
    fig, axes = plt.subplots(2, n_each, figsize=(1.6 * n_each, 3.6))
    for ax, i, m in zip(axes.flat, pick, maps):
        rgb = np.clip(x[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)
        ax.imshow(show_cam_on_image(rgb.astype(np.float32), m, use_rgb=True))
        ax.set_title(f"{'Bact.' if y[i] else 'Non-bact.'}  p={p[i]:.2f}", fontsize=7, color=INK)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / f"gradcam_{key}.png", dpi=300)
    fig.savefig(out / f"gradcam_{key}.pdf")
    pd.DataFrame({"patient_id": df["patient_id"].to_numpy()[pick], "y": y[pick], "p": p[pick]}).to_csv(
        cfg["paths"]["metrics"] / f"gradcam_{key}_cases.csv", index=False)
    print(f"Grad-CAM for {len(pick)} images (top row bacterial, bottom row non-bacterial; red = high attention). "
          f"Ask a clinician to rate anatomical plausibility.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("shap")
    g = sub.add_parser("gate")
    g.add_argument("--model", required=True)
    c = sub.add_parser("gradcam")
    c.add_argument("--model", required=True, help="key of results/models/<key>_full.pt, e.g. convnext_tiny")
    c.add_argument("--n", type=int, default=6, help="images per class")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = cfg["paths"]["figures"]
    out.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["metrics"].mkdir(parents=True, exist_ok=True)
    if args.cmd == "shap":
        run_shap(cfg, out)
    elif args.cmd == "gate":
        run_gate(cfg, args.model, out)
    else:
        run_gradcam(cfg, args.model, args.n, out)
    print(f"Figures in {out}")


if __name__ == "__main__":
    main()
