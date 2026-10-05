"""Write LaTeX rows for the supplementary tables from results/metrics (no hand transcription).

Usage: python -m src.supp_tables   -> paper/latex/supp_tables/*.tex
"""
from __future__ import annotations

import pandas as pd

from .common import ROOT, add_config_arg, load_config

OUT = ROOT / "paper" / "latex" / "supp_tables"
MIT = [("base", "None (reference)"), ("sog", "Colour constancy"), ("adjust", "Phone as covariate"),
       ("reweight", "Reweighting"), ("center", "Per-phone standardisation"), ("leace", "LEACE"),
       ("leace_rw", "LEACE + reweighting"), ("sog_leace_rw", "Colour + LEACE + reweighting"),
       ("mlp", "Network, no adversary"), ("adv0.3", r"Adversarial, $\lambda = 0.3$"),
       ("adv1", r"Adversarial, $\lambda = 1$"), ("adv3", r"Adversarial, $\lambda = 3$"),
       ("adv10", r"Adversarial, $\lambda = 10$")]
BB = ["vit_small_patch14_dinov2", "convnext_tiny"]


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    cfg = load_config(ap.parse_args().config)
    m = cfg["paths"]["metrics"]
    OUT.mkdir(parents=True, exist_ok=True)

    sym = pd.read_csv(m / "site_compare_symptoms.csv")
    rows = [f"{r.symptom} & {r.samsung_pct:.1f} & {r.xiaomi_pct:.1f} & {r.p:.2g} & {r.p_bonferroni:.2g} \\\\"
            for r in sym.itertuples()]
    (OUT / "symptoms.tex").write_text("\n".join(rows).replace("e-", r"\,e$-$") + "\n")

    sim = pd.read_csv(m / "validate_simulation_summary.csv")
    rows = []
    for score in ["marker only", "site only", "learned (marker + site)"]:
        for r in sim[sim["score"] == score].itertuples():
            rows.append(f"{score.capitalize() if r.delta == 0 else ''} & {r.delta:.1f} & {r.auc_target:.3f} & "
                        f"{r.auc:.3f} ({r.auc_bias:+.3f}) & {r.auc_deconf:.3f} ({r.auc_deconf_bias:+.3f}) & "
                        f"{r.auc_adjusted:.3f} ({r.auc_adjusted_bias:+.3f}) & {r.auc_deconf_sd:.3f} / "
                        f"{r.auc_adjusted_sd:.3f} \\\\")
        rows.append(r"\addlinespace")
    (OUT / "simulation.tex").write_text("\n".join(rows[:-1]).replace("(-0.000)", "(0.000)").replace("(+0.000)", "(0.000)") + "\n")

    sw = pd.read_csv(m / "validate_sweep_summary.csv")
    names = {"phone": "Phone only", "image": "Image (DINOv2)", "image_leace": "Image after LEACE", "symptoms": "Symptoms"}
    rows = []
    for key, lab in names.items():
        for r in sw[sw["model"] == key].itertuples():
            rows.append(f"{lab if r.delta == 0 else ''} & {r.delta:.1f} & {r.auc_mean:.3f} ({r.auc_std:.3f}) & "
                        f"{r.auc_deconf_mean:.3f} ({r.auc_deconf_std:.3f}) & {r.auc_adjusted_mean:.3f} "
                        f"({r.auc_adjusted_std:.3f}) \\\\")
        rows.append(r"\addlinespace")
    (OUT / "sweep.tex").write_text("\n".join(rows[:-1]) + "\n")

    s = pd.read_csv(m / "mitigation_summary.csv").set_index("model")
    rows = []
    for key, lab in MIT:
        cells = []
        for bb in BB:
            r = s.loc[f"mit_{key}_{bb}"]
            cells += [f3(r["auc"]), f3(r["auc_deconf"]), f3(r["auc_adjusted"])]
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    (OUT / "mitigation_adjusted.tex").write_text("\n".join(rows) + "\n")

    c = pd.read_csv(m / "mitigation_cross.csv")
    lab = {"base": "None", "sog": "Colour constancy", "center": "Per-phone standardisation",
           "sog_center": "Colour + standardisation"}
    rows = []
    for bb, bbl in [("vit_small_patch14_dinov2", "DINOv2"), ("convnext_tiny", "ConvNeXt-Tiny")]:
        for key, l in lab.items():
            cells = []
            for tr, te in [("Samsung", "Xiaomi"), ("Xiaomi", "Samsung")]:
                r = c[(c.backbone == bb) & (c.method == key) & (c.train == tr) & (c.test == te)].iloc[0]
                cells.append(f"{r.auc:.3f} [{r.auc_lo:.2f}, {r.auc_hi:.2f}]")
            rows.append(f"{bbl if key == 'base' else ''} & {l} & " + " & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")
    (OUT / "cross.tex").write_text("\n".join(rows[:-1]) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
