"""Turn the raw download into data/processed/{clean.csv, columns.json, images/, audit.md}.

Usage:
    python -m src.prepare_data --inspect     # print the raw columns, then fix configs/default.yaml
    python -m src.prepare_data               # build the processed dataset and the audit report
"""
from __future__ import annotations

import argparse
import itertools
import json
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .common import add_config_arg, load_config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
YES = {"1", "1.0", "yes", "y", "true", "t", "present", "+", "بله", "دارد"}
NO = {"0", "0.0", "no", "n", "false", "f", "absent", "-", "خیر", "ندارد"}
GENDER = {"m": 1, "male": 1, "man": 1, "boy": 1, "مرد": 1,
          "f": 0, "female": 0, "woman": 0, "girl": 0, "زن": 0}


def read_table(path):
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)


def inspect(raw: pd.DataFrame):
    print(f"{len(raw)} rows, {raw.shape[1]} columns\n")
    for c in raw.columns:
        s = raw[c]
        vals = ", ".join(map(str, s.dropna().unique()[:6]))
        print(f"  {str(c)[:32]:<32s} {str(s.dtype):<8s} missing={s.isna().sum():<4d} unique={s.nunique():<4d} e.g. {vals}")


def encode_binary(s: pd.Series, name: str) -> pd.Series:
    low = s.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index)
    out[low.isin(YES)] = 1.0
    out[low.isin(NO)] = 0.0
    bad = s.notna() & out.isna()
    if bad.any():
        raise ValueError(f"Column '{name}' has values that are not yes/no: {sorted(set(low[bad]))[:10]}. "
                         f"Add it to columns.ignore or columns.meta in the config, or recode it.")
    return out


def encode_gender(s: pd.Series) -> pd.Series:
    low = s.astype(str).str.strip().str.lower()
    if low[s.notna()].isin(GENDER).all():
        return low.map(GENDER).astype(float)
    num = pd.to_numeric(s, errors="coerce")
    if set(num.dropna().unique()) <= {0, 1}:
        print("  note: gender is already 0/1; check which code is male in the data paper")
        return num.astype(float)
    raise ValueError(f"Unrecognised gender values: {sorted(set(low))[:10]}")


def parse_votes(raw, diag_cols, bacterial_values):
    bact = {str(v).lower() for v in bacterial_values}
    low = raw[diag_cols].apply(lambda s: s.astype(str).str.strip().str.lower()).where(raw[diag_cols].notna())
    n_votes = low.notna().sum(axis=1)
    n_bact = low.isin(bact).sum(axis=1)
    return n_votes, n_bact, low


def find_images(folder):
    found = {}
    for p in folder.rglob("*"):
        if p.suffix.lower() in IMG_EXT and not p.name.startswith("."):
            found.setdefault(p.stem.strip().lower(), p)
    return found


def process_image(src, dst, short_side):
    with Image.open(src) as im:
        model = im.getexif().get(272)  # EXIF camera model, if present
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        s = short_side / min(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
        im.save(dst, quality=95)
    return (str(model).strip() if model else ""), (w, h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument("--inspect", action="store_true", help="only print the raw columns")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P, C = cfg["paths"], cfg["columns"]

    raw = read_table(P["raw_table"])
    raw.columns = [str(c).strip() for c in raw.columns]
    if args.inspect:
        inspect(raw)
        return

    report = ["# Data audit", ""]
    rx = re.compile(C["diagnosis_regex"], re.I)
    diag_cols = [c for c in raw.columns if rx.search(c)]
    for need in (C["id"], C["age"], C["gender"]):
        if need not in raw.columns:
            raise SystemExit(f"Column '{need}' not found. Run with --inspect and fix the config.")
    if not diag_cols:
        raise SystemExit("No diagnosis columns matched columns.diagnosis_regex. Run --inspect and fix it.")
    meta = [c for c in C.get("meta", []) if c in raw.columns]
    taken = {C["id"], C["age"], C["gender"], *diag_cols, *meta, *C.get("ignore", [])}
    symptoms = [c for c in raw.columns if c not in taken] if C["symptoms"] == "auto" else list(C["symptoms"])

    df = pd.DataFrame({"patient_id": raw[C["id"]].astype(str).str.strip()})
    df["age"] = pd.to_numeric(raw[C["age"]], errors="coerce")
    df["gender"] = encode_gender(raw[C["gender"]])
    for c in symptoms:
        df[c] = encode_binary(raw[c], c)
    for c in meta:
        df[c] = raw[c].astype(str).str.strip().where(raw[c].notna())

    # ---- labels
    n_votes, n_bact, votes = parse_votes(raw, diag_cols, C["bacterial_values"])
    df["n_votes"], df["n_bact"] = n_votes, n_bact
    df["y_soft"] = n_bact / n_votes.replace(0, np.nan)
    df["tie"] = (df["y_soft"] == 0.5).astype(int)
    df["y"] = (df["y_soft"] > 0.5).astype(float)
    tie_rule = C.get("tie_rule", "bacterial")
    if tie_rule == "bacterial":
        df.loc[df["tie"] == 1, "y"] = 1.0
    elif tie_rule == "nonbacterial":
        df.loc[df["tie"] == 1, "y"] = 0.0
    df["agreement"] = (df["y_soft"] - 0.5).abs() * 2

    # ---- images
    out_dir = P["processed"] / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    found = find_images(P["raw_images"])
    phones, sizes, paths = [], [], []
    for pid in df["patient_id"]:
        src = found.get(pid.lower())
        if src is None:
            phones.append(""), sizes.append(""), paths.append("")
            continue
        dst = out_dir / f"{pid}.jpg"
        model, wh = process_image(src, dst, cfg["image"]["short_side"])
        phones.append(model), sizes.append(f"{wh[0]}x{wh[1]}"), paths.append(str(dst.relative_to(P["processed"])))
    df["exif_phone"], df["orig_size"], df["image"] = phones, sizes, paths

    # ---- exclusions
    n0 = len(df)
    no_img = df["image"] == ""
    no_vote = df["n_votes"] == 0
    dup_id = df["patient_id"].duplicated(keep=False)
    drop = no_img | no_vote | dup_id | ((df["tie"] == 1) & (tie_rule == "drop"))
    ids_set = set(df["patient_id"].str.lower())
    orphan_imgs = sorted(k for k in found if k not in ids_set)
    clean = df[~drop].reset_index(drop=True)
    clean["y"] = clean["y"].astype(int)

    # ---- near-duplicate images
    dups = []
    try:
        import imagehash
        hashes = {pid: imagehash.phash(Image.open(P["processed"] / im))
                  for pid, im in zip(clean["patient_id"], clean["image"])}
        for (a, ha), (b, hb) in itertools.combinations(hashes.items(), 2):
            if ha - hb <= 5:
                dups.append((a, b, int(ha - hb)))
    except ImportError:
        report.append("_imagehash not installed: duplicate check skipped._")

    # ---- write outputs
    use_meta = meta + (["exif_phone"] if clean["exif_phone"].ne("").any() else [])
    cols = {"symptoms": symptoms, "binary": symptoms + ["gender"], "meta": use_meta, "diagnosis_columns": diag_cols}
    clean.to_csv(P["processed"] / "clean.csv", index=False)
    (P["processed"] / "columns.json").write_text(json.dumps(cols, indent=2, ensure_ascii=False))

    pos = int(clean["y"].sum())
    report += [
        f"- Rows in table: {n0}; kept: {len(clean)}",
        f"- Dropped: no image {int(no_img.sum())}, no votes {int(no_vote.sum())}, duplicate IDs {int(dup_id.sum())}",
        f"- Images without a table row: {len(orphan_imgs)} {orphan_imgs[:10]}",
        f"- Diagnosis columns: {diag_cols}",
        f"- Bacterial (majority vote): {pos} ({pos / len(clean):.1%}); non-bacterial: {len(clean) - pos}",
        f"- Ties (vote share exactly 0.5): {int(clean['tie'].sum())}, rule = {tie_rule}",
        f"- Unanimous cases: {int((clean['agreement'] == 1).sum())}",
        f"- Meta columns found: {meta or 'none'}; EXIF phone models: {sorted(set(clean['exif_phone']) - {''}) or 'none'}",
        f"- Near-duplicate image pairs (pHash distance <= 5): {len(dups)} {dups[:10]}",
        "", "## Diagnosis values (check that bacterial_values is right)", "",
        "| value | count |", "| --- | --- |",
        *[f"| {v} | {n} |" for v, n in votes.stack().value_counts().items()],
        "", "## Votes per patient", "", "| votes | patients |", "| --- | --- |",
        *[f"| {v} | {n} |" for v, n in clean["n_votes"].value_counts().sort_index().items()],
        "", "## Symptoms: prevalence by class and missing values", "",
        "| symptom | bacterial % | non-bacterial % | missing |", "| --- | --- | --- | --- |",
        *[f"| {c} | {100 * clean.loc[clean.y == 1, c].mean():.1f} | {100 * clean.loc[clean.y == 0, c].mean():.1f} | "
          f"{int(clean[c].isna().sum())} |" for c in cols["binary"]],
        "", f"Age: mean {clean['age'].mean():.1f}, SD {clean['age'].std():.1f}, missing {int(clean['age'].isna().sum())}",
    ]
    for c in use_meta:
        tab = pd.crosstab(clean[c].fillna("missing"), clean["y"])
        report += ["", f"## {c} by class", "", "| value | non-bacterial | bacterial |", "| --- | --- | --- |",
                   *[f"| {i} | {row.get(0, 0)} | {row.get(1, 0)} |" for i, row in tab.iterrows()]]
    (P["processed"] / "audit.md").write_text("\n".join(report) + "\n")
    print("\n".join(report[:12]))
    print(f"\nWrote {P['processed'] / 'clean.csv'} and audit.md")


if __name__ == "__main__":
    main()
