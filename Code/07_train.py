# 06_train.py
# Training loop for MonoSeqRUL

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config import *
from data import build_index_for_cutters
from dataset import PHMWindowDataset, collate_fn
from model import MonoSeqModel
from losses import nll_gaussian, monotonic_smoothness_loss, phase_classification_loss


def train_val_split(index, val_ratio=VAL_RATIO):
    val_n = int(len(index) * val_ratio)
    train_n = len(index) - val_n
    return random_split(index, [train_n, val_n],
                        generator=torch.Generator().manual_seed(SEED))


def train_model(train_index, val_index, scaler):
    # Build datasets
    train_ds = PHMWindowDataset(train_index, scaler=scaler)
    val_ds   = PHMWindowDataset(val_index, scaler=scaler)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    # Model + optimizer
    model = MonoSeqModel().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    best_val = 1e18
    for epoch in range(1, EPOCHS+1):
        # ---- Training ----
        model.train()
        tr_loss = 0.0
        for X, L, Xp, Lp, y_norm, y_raw, eol, cutn, cutter in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            X, L, y_norm, y_raw = X.to(DEVICE), L.to(DEVICE), y_norm.to(DEVICE), y_raw.to(DEVICE)

            opt.zero_grad()
            wear_pred, wear_var, phase_logits, increments = model(X, L)

            mask = torch.isfinite(y_norm[:, :, 3]).float()  # wear_norm exists?
            wear_loss = nll_gaussian(wear_pred, wear_var, y_norm[:, :, 3], mask)
            mono_loss = monotonic_smoothness_loss(increments, mask)
            phase_loss = phase_classification_loss(phase_logits, y_norm[:, :, -1].long(), mask)

            loss = wear_loss + 0.1*mono_loss + 0.1*phase_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss += loss.item() * X.size(0)

        tr_loss /= len(train_ds)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, L, Xp, Lp, y_norm, y_raw, eol, cutn, cutter in val_loader:
                X, L, y_norm = X.to(DEVICE), L.to(DEVICE), y_norm.to(DEVICE)
                wear_pred, wear_var, phase_logits, increments = model(X, L)
                mask = torch.isfinite(y_norm[:, :, 3]).float()
                wear_loss = nll_gaussian(wear_pred, wear_var, y_norm[:, :, 3], mask)
                mono_loss = monotonic_smoothness_loss(increments, mask)
                phase_loss = phase_classification_loss(phase_logits, y_norm[:, :, -1].long(), mask)
                loss = wear_loss + 0.1*mono_loss + 0.1*phase_loss
                val_loss += loss.item() * X.size(0)

        val_loss /= len(val_ds)
        sched.step(val_loss)

        print(f"Epoch {epoch:02d} | train {tr_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_monoseq.pt")
            print("  ↳ Saved best model.")

    return model
