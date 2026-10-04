"""Paper figures from the saved out-of-fold predictions (PNG at 300 dpi + PDF).

Subcommands:
  roc          mean ROC over repeats with ±1 SD band (<= 4 models)
  calibration  reliability curves (<= 4 models)
  dca          decision curves: net benefit vs. treat-all and treat-none (<= 4 models)
  ablation     AUC with 95% CI per model, as a dot plot (E3)
  subgroups    forest plot of AUC by subgroup for one model (E4)
  shortcut     paper Figure 3: AUC overall vs within phone (A) and cross-phone transfer (B)

Usage:
    python -m src.figures roc --models gated_convnext_tiny sym_lgbm img_lr_convnext_tiny
    python -m src.figures ablation --models gated_convnext_tiny concat_convnext_tiny gated_convnext_tiny__hard
    python -m src.figures subgroups --model gated_convnext_tiny
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from .common import add_config_arg, load_config, load_data, load_oof  # noqa: E402
from .stats import fold_aucs  # noqa: E402

# Validated categorical order (blue, orange, aqua, yellow); text never wears series colour.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight",
    })


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {out / name}.png/.pdf")


def check_count(models, cap=4):
    if len(models) > cap:
        raise SystemExit(f"Use at most {cap} models per panel so colours stay distinguishable; split into panels.")


def fig_roc(cfg, models, out, labels):
    check_count(models)
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], ls="--", color=REF, lw=1)
    for c, m in zip(SERIES, models):
        oof = load_oof(cfg, m)
        tprs, aucs = [], []
        for _, g in oof.groupby("repeat"):
            fpr, tpr, _ = roc_curve(g["y"], g["p"])
            tprs.append(np.interp(grid, fpr, tpr))
            aucs.append(roc_auc_score(g["y"], g["p"]))
        mu, sd = np.mean(tprs, 0), np.std(tprs, 0)
        ax.fill_between(grid, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=c, alpha=0.15, lw=0)
        ax.plot(grid, mu, color=c, label=f"{labels.get(m, m)}  AUC {np.mean(aucs):.3f}")
    ax.set(xlabel="1 − specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.01), aspect="equal")
    ax.legend(loc="lower right", fontsize=7)
    save(fig, out, "roc")


def fig_calibration(cfg, models, out, labels, bins=10):
    check_count(models)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], ls="--", color=REF, lw=1)
    for c, m in zip(SERIES, models):
        oof = load_oof(cfg, m)
        q = pd.qcut(oof["p"], bins, duplicates="drop")
        cal = oof.groupby(q, observed=True).agg(p=("p", "mean"), y=("y", "mean"))
        ax.plot(cal["p"], cal["y"], "o-", color=c, ms=4, label=labels.get(m, m))
    ax.set(xlabel="Predicted probability", ylabel="Observed bacterial fraction", xlim=(0, 1), ylim=(0, 1),
           aspect="equal")
    ax.legend(loc="upper left", fontsize=7)
    save(fig, out, "calibration")


def fig_dca(cfg, models, out, labels):
    from .evaluate import net_benefit
    check_count(models)
    th = np.linspace(0.05, 0.6, 56)
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    first = load_oof(cfg, models[0])
    prev = first.groupby("repeat")["y"].mean().mean()
    ax.plot(th, prev - (1 - prev) * th / (1 - th), color=REF, lw=1.2, label="Treat all")
    ax.axhline(0, color=INK, lw=0.8, label="Treat none")
    for c, m in zip(SERIES, models):
        oof = load_oof(cfg, m)
        nb = np.mean([net_benefit(g["y"].to_numpy(), g["p"].to_numpy(), th) for _, g in oof.groupby("repeat")], 0)
        ax.plot(th, nb, color=c, label=labels.get(m, m))
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", ylim=(-0.05, max(prev, 0.05) * 1.1))
    ax.legend(fontsize=7)
    save(fig, out, "decision_curve")


def fig_ablation(cfg, models, out, labels):
    """Dot + 95% CI (t-interval over fold AUCs); AUC does not start at zero, so no bars."""
    from scipy import stats as st
    rows = []
    for m in models:
        a = fold_aucs(load_oof(cfg, m)).to_numpy()
        half = st.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0
        rows.append((labels.get(m, m), a.mean(), half))
    fig, ax = plt.subplots(figsize=(4.4, 0.35 * len(rows) + 0.8))
    ys = np.arange(len(rows))[::-1]
    for y_, (lab, mu, h) in zip(ys, rows):
        ax.errorbar(mu, y_, xerr=h, fmt="o", color=SERIES[0], ms=5, capsize=0, elinewidth=1.4)
        ax.text(mu + h + 0.004, y_, f"{mu:.3f}", va="center", fontsize=7, color=MUTED)
    ax.axvline(rows[0][1], color=REF, ls="--", lw=1)
    ax.set_yticks(ys, [r[0] for r in rows], color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUC (mean of fold AUCs, 95% CI)")
    save(fig, out, "ablation")


def fig_subgroups(cfg, model, out, n_boot=500):
    df, cols = load_data(cfg)
    oof = load_oof(cfg, model).merge(df.drop(columns=["y", "y_soft", "agreement"]), on="patient_id")
    oof["age band"] = pd.cut(oof["age"], [-1, 14, 44, 200], labels=["≤14", "15–44", "≥45"])
    oof["gender"] = oof["gender"].map({0: "Female", 1: "Male"})
    groups = ["gender", "age band"] + [c for c in cols["meta"] if c in oof]
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for gcol in groups:
        for val, sub in oof.groupby(gcol, observed=True):
            aucs, boots = [], []
            for _, g in sub.groupby("repeat"):
                y, p = g["y"].to_numpy(), g["p"].to_numpy()
                if 0 < y.sum() < len(y):
                    aucs.append(roc_auc_score(y, p))
                    for _ in range(n_boot // max(sub["repeat"].nunique(), 1)):
                        i = rng.integers(0, len(y), len(y))
                        if 0 < y[i].sum() < len(y):
                            boots.append(roc_auc_score(y[i], p[i]))
            if aucs:
                n = sub["patient_id"].nunique()
                rows.append({"group": gcol, "value": str(val), "n": n, "auc": np.mean(aucs),
                             "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5)})
    res = pd.DataFrame(rows)
    res.to_csv(cfg["paths"]["metrics"] / f"subgroups_{model}.csv", index=False)
    overall = np.mean([roc_auc_score(g["y"], g["p"]) for _, g in load_oof(cfg, model).groupby("repeat")])
    fig, ax = plt.subplots(figsize=(4.4, 0.3 * len(res) + 0.8))
    ys = np.arange(len(res))[::-1]
    ax.errorbar(res["auc"], ys, xerr=[res["auc"] - res["lo"], res["hi"] - res["auc"]], fmt="s", color=SERIES[0],
                ms=4, capsize=0, elinewidth=1.4)
    ax.axvline(overall, color=REF, ls="--", lw=1)
    ax.set_yticks(ys, [f"{r.group}: {r.value} (n={r.n})" for r in res.itertuples()], color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUC (95% bootstrap CI); dashed = overall")
    save(fig, out, f"subgroups_{model}")
    print(res.round(3).to_string(index=False))


SHORTCUT_MODELS = [  # (oof name, label), ordered for the figure
    ("stack_vit_small_patch14_dinov2", "Stacked fusion (DINOv2)"),
    ("img_lr_vit_small_patch14_dinov2", "Image, frozen DINOv2"),
    ("ft_convnext_tiny", "Image, fine-tuned ConvNeXt"),
    ("ft_densenet121", "Image, fine-tuned DenseNet121"),
    ("phone", "Phone only"),
    ("lowlevel", "Photo statistics"),
    ("sym_lgbm", "Symptoms (LightGBM)"),
]


def fig_shortcut(cfg, out):
    """Panel A: pooled AUC vs within-phone AUC per model. Panel B: train on one phone, test on the other."""
    m = cfg["paths"]["metrics"]
    summ = pd.read_csv(m / "summary.csv").set_index("model")
    strata = pd.read_csv(m / "label_dependence_strata.csv").set_index("model")
    cross = pd.read_csv(m / "site_cross.csv")
    rows = [(lab, summ.loc[k, "auc"], strata.loc[k, "auc_within_phone"]) for k, lab in SHORTCUT_MODELS
            if k in summ.index and k in strata.index]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1.35, 1]})
    ys = np.arange(len(rows))[::-1]
    for y_, (lab, tot, wit) in zip(ys, rows):
        a.plot([wit, tot], [y_, y_], color=GRID, lw=2.2, zorder=1)
        a.scatter(tot, y_, color=SERIES[0], s=34, zorder=3, label="Overall" if y_ == ys[0] else None)
        a.scatter(wit, y_, color=SERIES[1], s=34, zorder=3, label="Within phone" if y_ == ys[0] else None)
    a.axvline(0.5, color=REF, ls="--", lw=1)
    a.set_yticks(ys, [r[0] for r in rows], color=INK)
    a.grid(axis="y", visible=False)
    a.set_xlim(0.45, 0.75)
    a.set_xlabel("AUC")
    a.set_title("A  Performance overall and within one phone", loc="left", fontsize=8.5)
    a.legend(loc="lower right", fontsize=7)

    names = {"symptoms (LR)": "Symptoms", "image vit_small_patch14_dinov2 (LR)": "Image DINOv2",
             "image convnext_tiny (LR)": "Image ConvNeXt"}
    cross = cross[cross["model"].isin(names)]
    groups = [("Samsung S21 Ultra", "Xiaomi"), ("Xiaomi", "Samsung S21 Ultra")]
    width = 0.26
    for gi, (tr, te) in enumerate(groups):
        sub = cross[(cross["train_phone"] == tr) & (cross["test_phone"] == te)].set_index("model")
        for mi, (key, lab) in enumerate(names.items()):
            if key not in sub.index:
                continue
            r = sub.loc[key]
            x = gi + (mi - 1) * width
            b.errorbar(x, r["auc"], yerr=[[r["auc"] - r["auc_lo"]], [r["auc_hi"] - r["auc"]]], fmt="o",
                       color=(SERIES[2], SERIES[3], MUTED)[mi], ms=5, elinewidth=1.4, capsize=0,
                       label=lab if gi == 0 else None)
    b.axhline(0.5, color=REF, ls="--", lw=1)
    b.set_xticks([0, 1], ["Train Samsung\ntest Xiaomi", "Train Xiaomi\ntest Samsung"], color=INK)
    b.set_xlim(-0.6, 1.6)
    b.set_ylim(0.3, 0.75)
    b.grid(axis="x", visible=False)
    b.set_ylabel("AUC (95% CI)")
    b.set_title("B  Train on one phone, test on the other", loc="left", fontsize=8.5)
    b.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, out, "fig3_site_shortcut")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument("kind", choices=["roc", "calibration", "dca", "ablation", "subgroups", "shortcut"])
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--model", help="for subgroups")
    ap.add_argument("--labels", nargs="*", default=[], help="display names, same order as --models")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = cfg["paths"]["figures"]
    cfg["paths"]["metrics"].mkdir(parents=True, exist_ok=True)
    labels = dict(zip(args.models, args.labels))
    style()
    if args.kind == "shortcut":
        fig_shortcut(cfg, out)
    elif args.kind == "subgroups":
        fig_subgroups(cfg, args.model or args.models[0], out)
    else:
        if not args.models:
            raise SystemExit("--models is required")
        {"roc": fig_roc, "calibration": fig_calibration, "dca": fig_dca, "ablation": fig_ablation}[args.kind](
            cfg, args.models, out, labels)


if __name__ == "__main__":
    main()
