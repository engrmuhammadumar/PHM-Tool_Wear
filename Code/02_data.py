"""
Data utilities for PHM2010 MonoSeq-RUL.
- Reads wear CSVs for training cutters
- Discovers cut CSV files for each cutter
- Builds index lists (train/val/test)
"""
from __future__ import annotations
import os, re, glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List

from sklearn.model_selection import train_test_split

from 01_config import cfg

# -------------------------------
# Wear reader (robust to variants)
# -------------------------------
def read_wear_table(cutter_dir: str) -> Tuple[pd.DataFrame, float]:
    cands = [p for p in glob.glob(os.path.join(cutter_dir, "*.csv")) if "wear" in os.path.basename(p).lower()]
    if not cands:
        raise FileNotFoundError(f"No wear csv in {cutter_dir}")
    wear_file = cands[0]

    # try detect header
    raw0 = pd.read_csv(wear_file, sep=None, engine="python", nrows=5)
    try:
        v = pd.to_numeric(raw0.iloc[0, 0], errors="coerce")
        use_header = bool(pd.isna(v))
    except Exception:
        use_header = True

    raw = (
        pd.read_csv(wear_file, sep=None, engine="python") if use_header
        else pd.read_csv(wear_file, sep=None, engine="python", header=None)
    )
    raw.columns = [str(c).strip().lower() for c in raw.columns]

    def first_present(names):
        for n in names:
            if n in raw.columns:
                return n
        return None

    cut_col = first_present(["cut", "cut_number", "cut no", "cut_no", "c", "index", "id", "0"])
    f1_col  = first_present(["flute_1", "flute1", "f1", "flute 1", "1"])
    f2_col  = first_present(["flute_2", "flute2", "f2", "flute 2", "2"])
    f3_col  = first_present(["flute_3", "flute3", "f3", "flute 3", "3"])

    if cut_col is None or f1_col is None or f2_col is None or f3_col is None:
        tmp = raw.copy().dropna(axis=1, how="all")
        assert tmp.shape[1] >= 4, "Wear file must have >=4 usable columns"
        tmp.columns = [f"col_{i}" for i in range(tmp.shape[1])]
        cut_col, f1_col, f2_col, f3_col = "col_0", "col_1", "col_2", "col_3"
        raw = tmp

    cut_series = raw[cut_col].astype(str).str.extract(r"(\d+)", expand=False)
    cut_series = pd.to_numeric(cut_series, errors="coerce")

    f1 = pd.to_numeric(raw[f1_col], errors="coerce")
    f2 = pd.to_numeric(raw[f2_col], errors="coerce")
    f3 = pd.to_numeric(raw[f3_col], errors="coerce")

    df = pd.DataFrame({
        "cut_number": cut_series,
        "flute_1": f1, "flute_2": f2, "flute_3": f3
    }).dropna()
    df["cut_number"] = df["cut_number"].round().astype(int)

    df["wear_max"] = df[["flute_1", "flute_2", "flute_3"]].max(axis=1)
    EOL = float(df["wear_max"].max())

    # Normalized
    eps = 1e-9
    df["f1_norm"] = df["flute_1"] / (EOL + eps)
    df["f2_norm"] = df["flute_2"] / (EOL + eps)
    df["f3_norm"] = df["flute_3"] / (EOL + eps)
    df["wear_norm"] = df["wear_max"] / (EOL + eps)
    df["rul_norm"]  = 1.0 - df["wear_norm"]

    df["RUL"] = EOL - df["wear_max"]

    return df.sort_values("cut_number").reset_index(drop=True), EOL


# -------------------------------
# Discover cut CSVs
# -------------------------------
def discover_cut_files(cutter_dir: str, cutter_id: int) -> Dict[int, str]:
    all_csvs = glob.glob(os.path.join(cutter_dir, "**", "*.csv"), recursive=True)
    all_csvs = [p for p in all_csvs if "wear" not in os.path.basename(p).lower()]
    cuts = {}
    for p in all_csvs:
        name = os.path.basename(p).lower()
        m = re.search(rf"c[_-]?{cutter_id}[_-]?(\d+)\.csv$", name) or re.search(r"(\d+)\.csv$", name)
        if m:
            cuts[int(m.group(1))] = p
    return dict(sorted(cuts.items()))


# -------------------------------
# Build index for cutters
# -------------------------------
def build_index_for_cutters(cutters: List[str], labeled: bool = True):
    index = []
    eol_map = {}
    for cname in cutters:
        cutter_dir = os.path.join(cfg.base, cname)
        cutter_id = int(re.findall(r"\d+", cname)[0])
        cut_files = discover_cut_files(cutter_dir, cutter_id)

        if labeled:
            wear_df, EOL = read_wear_table(cutter_dir)
            eol_map[cname] = EOL
            present = sorted(set(wear_df["cut_number"].astype(int)).intersection(cut_files.keys()))
            for cutn in present:
                row = wear_df.loc[wear_df["cut_number"] == cutn].iloc[0]
                y_norm = np.array([row["f1_norm"], row["f2_norm"], row["f3_norm"], row["wear_norm"], row["rul_norm"]], dtype=np.float32)
                y_raw  = np.array([row["flute_1"], row["flute_2"], row["flute_3"], row["wear_max"], row["RUL"]], dtype=np.float32)
                index.append({
                    "cutter": cname, "eol": EOL,
                    "cut_number": int(cutn),
                    "path": cut_files[int(cutn)],
                    "y_norm": y_norm, "y_raw": y_raw
                })
        else:
            present = sorted(cut_files.keys())
            for cutn in present:
                index.append({
                    "cutter": cname, "eol": None,
                    "cut_number": int(cutn),
                    "path": cut_files[int(cutn)],
                    "y_norm": None, "y_raw": None
                })
    return index, eol_map


if __name__ == "__main__":
    train_index, train_eols = build_index_for_cutters(cfg.train_cutters, labeled=True)
    test_index, _          = build_index_for_cutters(cfg.test_cutters, labeled=False)
    print("Train samples:", len(train_index))
    print("Test samples :", len(test_index))
    print("Train EOLs   :", {k: round(v, 2) for k, v in train_eols.items()})
