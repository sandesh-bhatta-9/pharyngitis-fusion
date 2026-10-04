"""Generate a fake dataset with the same layout as PGUPharyngitis, to test the pipeline end to end.

The numbers it produces mean nothing. Use it only to check that every script runs.

Usage:
    python -m src.make_synthetic --n 300
    python -m src.prepare_data --config configs/synthetic.yaml   # then run the pipeline with that config
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from .common import ROOT

SYMPTOMS = ["fever", "cough", "tonsillar_exudate", "cervical_lymphadenopathy", "sore_throat", "runny_nose",
            "hoarseness", "headache", "muscle_pain", "fatigue", "sneezing", "conjunctivitis", "nausea",
            "abdominal_pain", "chills", "difficulty_swallowing", "ear_pain", "rash", "bad_breath", "diarrhea"]


def throat_image(rng, red, exudate, w=360, h=300):
    base = np.array([190 + 50 * red, 120 - 40 * red, 120 - 30 * red]) + rng.normal(0, 12, 3)
    im = Image.new("RGB", (w, h), tuple(int(np.clip(v, 0, 255)) for v in base))
    d = ImageDraw.Draw(im)
    d.ellipse([w * 0.3, h * 0.15, w * 0.7, h * 0.95], fill=(70, 20, 30))          # pharynx opening
    for cx in (0.22, 0.78):                                                           # tonsils
        d.ellipse([w * (cx - 0.12), h * 0.35, w * (cx + 0.12), h * 0.8],
                  fill=(int(min(255, 200 + 50 * red)), 90, 100))
        for _ in range(int(exudate * rng.integers(3, 8))):                            # white spots
            x, y = w * (cx + rng.uniform(-0.08, 0.08)), h * rng.uniform(0.4, 0.75)
            d.ellipse([x - 6, y - 5, x + 6, y + 5], fill=(240, 240, 225))
    return im.filter(ImageFilter.GaussianBlur(1.5)).rotate(rng.uniform(-10, 10))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="data_synthetic/raw")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = ROOT / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)

    n = args.n
    bact = rng.random(n) < 0.3
    age = np.where(rng.random(n) < 0.3, rng.integers(3, 15, n), rng.integers(15, 70, n))
    rows = {"patient_ID": [f"P{i:04d}" for i in range(n)], "age": age,
            "gender": rng.choice(["Male", "Female"], n),
            "city": rng.choice(["CityA", "CityB"], n), "phone": rng.choice(["S21 Ultra", "Redmi 8 Pro"], n)}
    for j, s in enumerate(SYMPTOMS):
        lift = {"fever": 0.35, "tonsillar_exudate": 0.4, "cervical_lymphadenopathy": 0.3,
                "cough": -0.35, "runny_nose": -0.3}.get(s, rng.normal(0, 0.05))
        prob = np.clip(0.35 + lift * (bact - 0.3), 0.02, 0.98)
        rows[s] = np.where(rng.random(n) < prob, "yes", "no")
    df = pd.DataFrame(rows)

    # Physicians see symptoms + photo with noise; 4-9 votes each.
    evidence = (1.2 * bact + 0.5 * (df["fever"] == "yes") + 0.6 * (df["tonsillar_exudate"] == "yes")
                - 0.5 * (df["cough"] == "yes") + rng.normal(0, 0.6, n))
    for d in range(1, 10):
        vote = np.where(evidence + rng.normal(0, 0.5, n) > 0.9, "Bacterial", "Non-bacterial")
        has = rng.integers(4, 10, n) >= d
        df[f"Diagnosis_{d}"] = np.where(has, vote, None)
    df.loc[df.sample(frac=0.02, random_state=1).index, "age"] = np.nan

    for pid, b, ex in zip(df["patient_ID"], bact, df["tonsillar_exudate"] == "yes"):
        red = np.clip(0.5 * b + rng.normal(0.3, 0.2), 0, 1)
        throat_image(rng, red, ex).save(out / "images" / f"{pid}.jpg", quality=90)
    df.to_excel(out / "synthetic.xlsx", index=False)
    print(f"Wrote {n} synthetic patients to {out}")


if __name__ == "__main__":
    main()
