# 05_losses.py
# Loss functions for MonoSeqRUL

import torch
import torch.nn as nn


def nll_gaussian(y_pred, y_var, y_true, mask=None):
    """
    Heteroscedastic regression loss (negative log likelihood).
    y_pred: [B,T] mean predictions
    y_var:  [B,T] log variance predictions
    y_true: [B,T] ground truth targets
    mask:   [B,T] binary mask (1=valid, 0=ignore)
    """
    if mask is None:
        mask = torch.ones_like(y_true)

    var = torch.exp(y_var)
    loss = 0.5 * (torch.log(var) + (y_true - y_pred) ** 2 / var)
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-9)


def monotonic_smoothness_loss(increments, mask=None, lambda_smooth=0.1):
    """
    Penalize negative increments (to enforce monotonicity)
    and encourage smooth step-to-step progression.
    increments: [B,T] non-negative wear increments
    """
    if mask is None:
        mask = torch.ones_like(increments)

    # Negative increments penalty
    neg_penalty = torch.relu(-increments) * mask

    # Smoothness penalty: difference between consecutive increments
    diff = increments[:, 1:] - increments[:, :-1]
    smooth_penalty = diff.abs() * mask[:, 1:]

    loss = neg_penalty.sum() + lambda_smooth * smooth_penalty.sum()
    return loss / (mask.sum() + 1e-9)


def phase_classification_loss(phase_logits, phase_targets, mask=None):
    """
    Cross-entropy loss for degradation phase classification.
    phase_logits: [B,T,C] (logits for C phases)
    phase_targets: [B,T] integer labels
    mask: [B,T] binary mask
    """
    ce = nn.CrossEntropyLoss(reduction="none")
    B, T, C = phase_logits.shape
    phase_logits = phase_logits.reshape(B*T, C)
    phase_targets = phase_targets.reshape(B*T)
    losses = ce(phase_logits, phase_targets)
    if mask is not None:
        mask = mask.reshape(B*T)
        losses = losses * mask
    return losses.mean()
