"""Paper figures from the saved out-of-fold predictions (PNG at 300 dpi + PDF).

Subcommands:
  roc          mean ROC over repeats with ±1 SD band (<= 4 models)
  calibration  reliability curves (<= 4 models)
  dca          decision curves: net benefit vs. treat-all and treat-none (<= 4 models)
  ablation     AUC with 95% CI per model, as a dot plot (E3)
  subgroups    forest plot of AUC by subgroup for one model (E4)
  shortcut     paper Figure 3: AUC overall vs within phone (A) and cross-phone transfer (B)
  mitigation   paper Figure 4: pooled vs deconfounded AUC for main models (A) and mitigation methods (B)
  validation   paper Figure 5: metric behaviour vs confounding strength, simulated (A) and resampled real data (B)

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

DECONF_MODELS = [("phone", "Phone only"), ("lowlevel", "Photo statistics"), ("sym_lgbm", "Symptoms (LightGBM)"),
                 ("img_lr_convnext_tiny", "Image, ConvNeXt"), ("img_lr_vit_small_patch14_dinov2", "Image, DINOv2"),
                 ("stack_vit_small_patch14_dinov2", "Stacked fusion, DINOv2"),
                 ("gated_vit_small_patch14_dinov2", "Gated fusion, DINOv2"),
                 ("ft_convnext_tiny", "Fine-tuned ConvNeXt*")]
MITIGATION_METHODS = [("base", "None (reference)"), ("sog", "Colour constancy"), ("adjust", "Phone as covariate"),
                      ("reweight", "Reweighting"), ("center", "Per-phone standardisation"),
                      ("leace", "Concept erasure (LEACE)"), ("leace_rw", "LEACE + reweighting"),
                      ("sog_leace_rw", "Colour + LEACE + reweighting"), ("mlp", "Network, no adversary"),
                      ("adv1", "Adversarial, \u03bb = 1"), ("adv10", "Adversarial, \u03bb = 10")]


def _dumbbell(ax, rows, note=None):
    ys = np.arange(len(rows))[::-1]
    for i, (y_, (lab, tot, dec, lo, hi, extra)) in enumerate(zip(ys, rows)):
        ax.plot([dec, tot], [y_, y_], color=GRID, lw=2.2, zorder=1)
        ax.hlines(y_, lo, hi, color=SERIES[1], lw=1.2, alpha=0.6, zorder=2)
        ax.scatter(tot, y_, color=SERIES[0], s=30, zorder=3, label="Pooled AUC" if i == 0 else None)
        ax.scatter(dec, y_, color=SERIES[1], s=30, zorder=3, label="Deconfounded AUC (95% CI)" if i == 0 else None)
        if extra is not None:
            ax.text(0.765, y_, f"{extra:.3f}", va="center", ha="right", fontsize=7, color=MUTED)
    if note:
        ax.text(0.765, ys[0] + 0.55, note, va="bottom", ha="right", fontsize=6.5, color=MUTED)
    ax.axvline(0.5, color=REF, ls="--", lw=1)
    ax.set_yticks(ys, [r[0] for r in rows], color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.42, 0.77)
    ax.set_ylim(-0.7, len(rows) - 0.05)
    ax.set_xlabel("AUC")


def fig_mitigation(cfg, out, backbone="vit_small_patch14_dinov2"):
    """Panel A: pooled vs deconfounded AUC of the main models. Panel B: the same for each mitigation method."""
    summ = pd.read_csv(cfg["paths"]["metrics"] / "mitigation_summary.csv").set_index("model")
    row = lambda k, lab, probe=False: (lab, summ.loc[k, "auc"], summ.loc[k, "auc_deconf"],  # noqa: E731
                                       summ.loc[k, "auc_deconf_lo"], summ.loc[k, "auc_deconf_hi"],
                                       summ.loc[k, "phone_id_auc"] if probe and k in summ.index else None)
    rows_a = [row(k, lab) for k, lab in DECONF_MODELS if k in summ.index]
    rows_b = [row(f"mit_{m}_{backbone}", lab, probe=True) for m, lab in MITIGATION_METHODS
              if f"mit_{m}_{backbone}" in summ.index]
    rows_b = [(lab, t, d, lo, hi, None if (e is None or np.isnan(e)) else e) for lab, t, d, lo, hi, e in rows_b]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.9), gridspec_kw={"width_ratios": [1, 1.1]})
    _dumbbell(a, rows_a)
    a.set_title("A  Main models", loc="left", fontsize=8.5)
    fig.legend(*a.get_legend_handles_labels(), loc="lower center", fontsize=7, ncol=2, bbox_to_anchor=(0.5, -0.02))
    _dumbbell(b, rows_b, note="phone ID AUC")
    b.set_title("B  Shortcut mitigation (DINOv2 features)", loc="left", fontsize=8.5, pad=14)
    a.set_title("A  Main models", loc="left", fontsize=8.5, pad=14)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out, "fig_mitigation")


def fig_validation(cfg, out):
    """A: simulation, learned marker + site model. B: real data resampled to a given prevalence gap, image model."""
    m = cfg["paths"]["metrics"]
    sim = pd.read_csv(m / "validate_simulation_summary.csv")
    sw = pd.read_csv(m / "validate_sweep_summary.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    series = [("auc", "Pooled AUC", SERIES[0], "o"), ("auc_deconf", "Deconfounded AUC", SERIES[1], "s"),
              ("auc_adjusted", "Adjusted AUC (Janes & Pepe)", SERIES[2], "^")]
    d = sim[sim["score"] == "learned (marker + site)"]
    for col, lab, c, mk in series:
        a.plot(d["delta"], d[col], marker=mk, color=c, ms=4, label=lab)
    a.plot(d["delta"], d["auc_target"], color=INK, ls="--", lw=1.2, label="Target: AUC without confounding")
    s_only = sim[sim["score"] == "site only"]
    a.plot(s_only["delta"], s_only["auc"], color=SERIES[0], ls=":", lw=1.2, label="Site-only score, pooled AUC")
    a.set_title("A  Simulation (true marker AUC 0.60)", loc="left", fontsize=8.5)
    a.set_xlabel("Prevalence gap between sites")
    a.set_ylabel("AUC")
    img, ph = sw[sw["model"] == "image"], sw[sw["model"] == "phone"]
    for col, lab, c, mk in series:
        b.errorbar(img["delta"], img[f"{col}_mean"], yerr=img[f"{col}_std"], marker=mk, color=c, ms=4,
                   elinewidth=1, capsize=0)
    b.axhline(img.loc[img["delta"] == 0, "auc_mean"].iloc[0], color=INK, ls="--", lw=1.2)
    b.plot(ph["delta"], ph["auc_mean"], color=SERIES[0], ls=":", lw=1.2)
    b.axvline(0.29, color=REF, lw=0.8)
    b.text(0.295, 0.45, "full data", fontsize=6.5, color=MUTED, rotation=90, va="bottom")
    b.set_title("B  PGUPharyngitis resampled (DINOv2 image)", loc="left", fontsize=8.5)
    b.set_xlabel("Prevalence gap between phones")
    for ax in (a, b):
        ax.axhline(0.5, color=REF, lw=0.8)
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
        ax.set_ylim(0.42, 0.8)
    fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, out, "fig_validation")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument("kind", choices=["roc", "calibration", "dca", "ablation", "subgroups", "shortcut", "mitigation", "validation"])
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
    elif args.kind == "mitigation":
        fig_mitigation(cfg, out)
    elif args.kind == "validation":
        fig_validation(cfg, out)
    elif args.kind == "subgroups":
        fig_subgroups(cfg, args.model or args.models[0], out)
    else:
        if not args.models:
            raise SystemExit("--models is required")
        {"roc": fig_roc, "calibration": fig_calibration, "dca": fig_dca, "ablation": fig_ablation}[args.kind](
            cfg, args.models, out, labels)


if __name__ == "__main__":
    main()
