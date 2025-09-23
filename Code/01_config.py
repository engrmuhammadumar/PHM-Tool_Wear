"""
Config for MonoSeq-RUL on PHM2010.
Edit BASE to your dataset root. Everything else can stay as-is to start.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, asdict
from pathlib import Path
import torch

# =============================
# 1) BASIC PATHS & SPLITS
# =============================
# !!! EDIT THIS TO YOUR LOCAL PATH !!!
BASE = r"E:\\Collaboration Work\\With Farooq\\phm dataset\\PHM Challange 2010 Milling"

TRAIN_CUTTERS = ["c1", "c4", "c6"]  # labeled
TEST_CUTTERS  = ["c2", "c3", "c5"]  # unlabeled

# Where to drop intermediate CSVs, models, plots
ROOT_OUT = Path("artifacts_monoseq").resolve()
FEAT_DIR = ROOT_OUT / "features"
MODEL_DIR = ROOT_OUT / "models"
PLOT_DIR  = ROOT_OUT / "plots"
CSV_DIR   = ROOT_OUT / "csv"

# =============================
# 2) SIGNAL / WINDOW SETTINGS
# =============================
FS = 50_000                 # Hz (per-channel sample rate)
WIN = 4096                  # samples per window (≈82 ms)
HOP = 2048                  # hop between windows (50% overlap)
MAX_WINDOWS = 96            # max windows to summarize per cut (cap for speed)

# =============================
# 3) TRAINING SETTINGS
# =============================
SEED = 42
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 35
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
VAL_RATIO = 0.15            # last 15% cuts per labeled cutter for validation

# Backbone
HIDDEN = 256
DROPOUT = 0.2
NUM_LAYERS = 2              # we use 2 to enable recurrent dropout

# Loss mixing
LAMBDA_SMOOTH_DELTA = 0.1   # smoothness on wear increments
LAMBDA_PHASE = 0.2          # auxiliary phase classification weight

# Conformal
CONF_ALPHA = 0.1            # 90% intervals by default

# Test-time adaptation
TTA_STEPS = 20
TTA_LR = 1e-4
TTA_SMOOTH = 0.1

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    base: str = BASE
    train_cutters: list[str] = None
    test_cutters: list[str] = None
    fs: int = FS
    win: int = WIN
    hop: int = HOP
    max_windows: int = MAX_WINDOWS

    seed: int = SEED
    batch_size: int = BATCH_SIZE
    lr: float = LR
    epochs: int = EPOCHS
    weight_decay: float = WEIGHT_DECAY
    grad_clip: float = GRAD_CLIP
    val_ratio: float = VAL_RATIO

    hidden: int = HIDDEN
    dropout: float = DROPOUT
    num_layers: int = NUM_LAYERS

    lambda_smooth_delta: float = LAMBDA_SMOOTH_DELTA
    lambda_phase: float = LAMBDA_PHASE

    conf_alpha: float = CONF_ALPHA

    tta_steps: int = TTA_STEPS
    tta_lr: float = TTA_LR
    tta_smooth: float = TTA_SMOOTH

    root_out: Path = ROOT_OUT
    feat_dir: Path = FEAT_DIR
    model_dir: Path = MODEL_DIR
    plot_dir: Path = PLOT_DIR
    csv_dir: Path = CSV_DIR

    device: torch.device = DEVICE

    def __post_init__(self):
        if self.train_cutters is None:
            self.train_cutters = TRAIN_CUTTERS
        if self.test_cutters is None:
            self.test_cutters = TEST_CUTTERS


def make_dirs(cfg: Config):
    for p in [cfg.root_out, cfg.feat_dir, cfg.model_dir, cfg.plot_dir, cfg.csv_dir]:
        p.mkdir(parents=True, exist_ok=True)


def show(cfg: Config):
    print("\n=== MonoSeq-RUL Config ===")
    for k, v in asdict(cfg).items():
        print(f"{k}: {v}")
    print("Device:", cfg.device)
    print("==========================\n")


# Convenience singleton (import this in other modules)
cfg = Config()
make_dirs(cfg)
if __name__ == "__main__":
    show(cfg)
