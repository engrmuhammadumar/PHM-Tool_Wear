"""
Model for PHM2010 MonoSeq-RUL.
- Encoder: BiGRU/LSTM over feature sequence
- Heads: (1) increment regression (monotone cum-sum)
          (2) phase classification (early/mid/late)
          (3) uncertainty via variance outputs
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from 01_config import cfg


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int = cfg.hidden):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, hidden)

    def forward(self, x):
        h, _ = self.rnn(x)
        h = F.relu(self.fc(h))
        return h  # (B,T,H)


class MonoSeqModel(nn.Module):
    def __init__(self, input_dim: int, hidden: int = cfg.hidden, n_phases: int = 3):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden)
        # increment regression head
        self.inc_mu = nn.Linear(hidden, 1)
        self.inc_logvar = nn.Linear(hidden, 1)
        # phase classifier
        self.phase = nn.Linear(hidden, n_phases)

    def forward(self, x):
        h = self.encoder(x)  # (B,T,H)
        mu = self.inc_mu(h).squeeze(-1)         # (B,T)
        logvar = self.inc_logvar(h).squeeze(-1) # (B,T)
        phase_logits = self.phase(h).mean(dim=1)  # average over time

        # cumulative sum for monotonic wear
        wear_mu = torch.cumsum(F.softplus(mu), dim=1)
        wear_var = torch.cumsum(F.softplus(logvar), dim=1)

        return wear_mu, wear_var, phase_logits


if __name__ == "__main__":
    B, T, Fdim = 2, cfg.max_windows, 20
    x = torch.randn(B, T, Fdim)
    net = MonoSeqModel(input_dim=Fdim)
    wear_mu, wear_var, phase_logits = net(x)
    print(wear_mu.shape, wear_var.shape, phase_logits.shape)
