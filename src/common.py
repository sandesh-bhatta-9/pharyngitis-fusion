"""Shared helpers: config, seeding, data and fold loading, tabular preprocessing, OOF files."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TAG_SEP = "__"  # model names with this separator are variants (knock-outs, controls), not main models


def add_config_arg(parser):
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config, relative to project root")


def load_config(path: str = "configs/default.yaml") -> dict:
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    cfg = yaml.safe_load(p.read_text())
    cfg["paths"] = {k: (Path(v) if Path(v).is_absolute() else ROOT / v) for k, v in cfg["paths"].items()}
    res = cfg["paths"]["results"]
    cfg["paths"].update(oof=res / "oof", metrics=res / "metrics", models=res / "models")
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def torch_device():
    """NVIDIA GPU if present, else Apple-silicon GPU, else CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------- data

def load_data(cfg) -> tuple[pd.DataFrame, dict]:
    proc = cfg["paths"]["processed"]
    df = pd.read_csv(proc / "clean.csv", dtype={"patient_id": str})
    cols = json.loads((proc / "columns.json").read_text())
    return df, cols


def tab_array(df: pd.DataFrame, cols: dict, drop=()) -> tuple[np.ndarray, int, list[str]]:
    """Tabular inputs as [binary columns..., age]. Returns (X, n_binary, names)."""
    drop = set(drop)
    unknown = drop - set(cols["binary"])
    if unknown:
        raise ValueError(f"--drop-symptoms names not found: {sorted(unknown)}")
    binary = [c for c in cols["binary"] if c not in drop]
    names = binary + ["age"]
    return df[names].to_numpy(dtype=float), len(binary), names


def tab_transformer(n_bin: int, img_cols: list[int] | None = None, img_pca: int | None = None):
    """Impute + scale, fitted on training data only. Optional image block for early fusion."""
    from sklearn.decomposition import PCA
    parts = [
        ("bin", SimpleImputer(strategy="most_frequent", add_indicator=True), list(range(n_bin))),
        ("age", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), [n_bin]),
    ]
    if img_cols:
        img_steps = [("sc", StandardScaler())]
        if img_pca:
            img_steps.append(("pca", PCA(n_components=img_pca, random_state=0)))
        parts.append(("img", Pipeline(img_steps), img_cols))
    return ColumnTransformer(parts, remainder="drop")


# ---------------------------------------------------------------- folds

def load_folds(cfg, df: pd.DataFrame) -> np.ndarray:
    """Array (repeats, n_patients) of outer fold ids aligned to df row order."""
    f = pd.read_csv(cfg["paths"]["processed"] / "folds.csv", dtype={"patient_id": str})
    out = []
    for r in sorted(f["repeat"].unique()):
        m = f[f["repeat"] == r].set_index("patient_id")["fold"]
        out.append(df["patient_id"].map(m).to_numpy())
    arr = np.array(out)
    if np.isnan(arr.astype(float)).any():
        raise RuntimeError("folds.csv does not cover every patient; rerun src.make_folds")
    return arr.astype(int)


def outer_splits(cfg, df, repeats: int | None = None):
    folds = load_folds(cfg, df)
    n_rep = folds.shape[0] if repeats is None else min(repeats, folds.shape[0])
    for r in range(n_rep):
        for k in range(cfg["cv"]["outer_folds"]):
            te = np.where(folds[r] == k)[0]
            tr = np.where(folds[r] != k)[0]
            yield r, k, tr, te


def inner_cv(cfg, r: int, k: int) -> StratifiedKFold:
    return StratifiedKFold(cfg["cv"]["inner_folds"], shuffle=True, random_state=cfg["seed"] + 100 * r + k)


# ---------------------------------------------------------------- thresholds and OOF files

def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising balanced accuracy on (inner, out-of-fold) training predictions."""
    cand = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    scores = np.array([balanced_accuracy_score(y, (p >= t).astype(int)) for t in cand])
    best = np.flatnonzero(scores == scores.max())
    return float(cand[best[len(best) // 2]])


def oof_frame(df, idx, r, k, p, thr, **extra) -> pd.DataFrame:
    out = pd.DataFrame({
        "patient_id": df["patient_id"].to_numpy()[idx],
        "repeat": r, "fold": k,
        "y": df["y"].to_numpy()[idx],
        "y_soft": df["y_soft"].to_numpy()[idx],
        "agreement": df["agreement"].to_numpy()[idx],
        "p": p, "thr": thr,
    })
    for key, val in extra.items():
        out[key] = val
    return out


def save_oof(cfg, name: str, frames: list[pd.DataFrame]) -> pd.DataFrame:
    oof = pd.concat(frames, ignore_index=True)
    d = cfg["paths"]["oof"]
    d.mkdir(parents=True, exist_ok=True)
    oof.to_csv(d / f"{name}.csv", index=False)
    aucs = oof.groupby("repeat").apply(lambda g: roc_auc_score(g["y"], g["p"]), include_groups=False)
    print(f"  {name:<40s} AUC {aucs.mean():.3f} ± {aucs.std(ddof=1) if len(aucs) > 1 else 0:.3f}")
    return oof


def load_oof(cfg, name: str) -> pd.DataFrame:
    return pd.read_csv(cfg["paths"]["oof"] / f"{name}.csv", dtype={"patient_id": str})


def list_models(cfg) -> list[str]:
    return sorted(p.stem for p in cfg["paths"]["oof"].glob("*.csv"))


def backbone_key(name: str) -> str:
    return name.split(".")[0]


def load_features(cfg, key: str, df: pd.DataFrame):
    """Cached embeddings aligned to df order: test (n,D), flip (n,D), aug (K,n,D)."""
    z = np.load(cfg["paths"]["features"] / f"{key}.npz", allow_pickle=False)
    pos = {pid: i for i, pid in enumerate(z["ids"].astype(str))}
    idx = np.array([pos[p] for p in df["patient_id"]])
    return z["test"][idx], z["flip"][idx], z["aug"][:, idx]
