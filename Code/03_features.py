"""
Feature extraction for PHM2010 MonoSeq-RUL.
- Reads a single cut CSV (7 channels)
- Splits into windows (WIN, HOP)
- Computes per-window features (time + freq)
- Aggregates to fixed-length feature sequence (MAX_WINDOWS)
"""
from __future__ import annotations
import os, math, glob
import numpy as np
import pandas as pd

from scipy.signal import welch
from scipy.stats import skew, kurtosis

from 01_config import cfg

# -------------------------------
# Helper: safe load cut CSV
# -------------------------------
def load_cut_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep=None, engine="python")
    df = df.dropna(axis=1, how="all")
    return df.values.astype(np.float32)


# -------------------------------
# Per-window features
# -------------------------------
def extract_features_window(win: np.ndarray, fs: int = 50000) -> np.ndarray:
    feats = []
    for i in range(win.shape[1]):
        x = win[:, i]
        # time domain
        feats += [x.mean(), x.std(), x.min(), x.max(), skew(x), kurtosis(x)]
        # energy
        feats.append(np.sum(x ** 2) / len(x))
        # spectral power (Welch)
        f, Pxx = welch(x, fs=fs, nperseg=min(1024, len(x)))
        feats.append(Pxx.mean())
        feats.append(Pxx.max())
    return np.array(feats, dtype=np.float32)


# -------------------------------
# Aggregate windows -> sequence
# -------------------------------
def extract_feature_sequence(path: str) -> np.ndarray:
    arr = load_cut_csv(path)
    N, C = arr.shape
    feats = []
    for start in range(0, N - cfg.win + 1, cfg.hop):
        win = arr[start:start+cfg.win]
        feats.append(extract_features_window(win))
    feats = np.stack(feats) if feats else np.zeros((1, 10))

    # pad or truncate to MAX_WINDOWS
    if feats.shape[0] >= cfg.max_windows:
        seq = feats[:cfg.max_windows]
    else:
        pad = np.zeros((cfg.max_windows - feats.shape[0], feats.shape[1]), dtype=np.float32)
        seq = np.vstack([feats, pad])
    return seq.astype(np.float32)


if __name__ == "__main__":
    # Quick test
    import random, glob
    all_cuts = glob.glob(os.path.join(cfg.base, cfg.train_cutters[0], "**", "*.csv"), recursive=True)
    cut_file = [p for p in all_cuts if "wear" not in os.path.basename(p).lower()][0]
    seq = extract_feature_sequence(cut_file)
    print("Seq shape:", seq.shape)
