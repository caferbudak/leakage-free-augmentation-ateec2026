"""
Hybrid augmentation (CVAE + SMOTE): a data augmentation method that combines
class-conditional variational autoencoder (CVAE) sampling with SMOTE
interpolation.

ATEEC 2026 — Assoc. Prof. Cafer Budak, Dicle University.

Method:
  D_aug = D_SMOTE ∪ D_CVAE
    D_SMOTE : SMOTE oversamples the minority class up to the majority class
              size (class balance).
    D_CVAE  : the class-conditional VAE draws one synthetic sample from the
              prior (z ~ N(0, I)) for every training sample (diversity
              beyond the convex hull of the data).
Augmentation is applied ONLY to the training fold (leakage-free protocol).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE

SEED = 123


# ------------------------------ Preprocessing ---------------------------------
def minmax_fit(X: np.ndarray):
    """Estimates per-feature minimum and range from the training fold
    (a zero range is set to 1)."""
    mn = np.nanmin(X, 0)
    mx = np.nanmax(X, 0)
    rg = np.where(mx - mn == 0, 1.0, mx - mn)
    return mn, rg


def minmax_apply(X: np.ndarray, mn: np.ndarray, rg: np.ndarray) -> np.ndarray:
    """Fills missing values with the column median and scales to [0, 1]."""
    Xf = np.where(np.isnan(X), np.nanmedian(X, 0), X)
    return np.clip((Xf - mn) / rg, 0.0, 1.0)


def _k_neighbors(y: np.ndarray) -> int:
    """Safe number of SMOTE neighbours given the minority class size."""
    return max(1, min(5, int(np.bincount(y).min()) - 1))


# ------------------------- Conditional VAE (CVAE) -----------------------------
class CVAE(nn.Module):
    """Class-conditional variational autoencoder with one hidden layer."""

    def __init__(self, d: int, lat: int, h: int, n_classes: int):
        super().__init__()
        self.lat = lat
        self.n_classes = n_classes
        self.enc = nn.Sequential(nn.Linear(d + n_classes, h), nn.Tanh())
        self.mu = nn.Linear(h, lat)
        self.logvar = nn.Linear(h, lat)
        self.dec = nn.Sequential(nn.Linear(lat + n_classes, h), nn.Tanh(),
                                 nn.Linear(h, d), nn.Sigmoid())

    def forward(self, x, c):
        h = self.enc(torch.cat([x, c], 1))
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)   # reparameterization
        xhat = self.dec(torch.cat([z, c], 1))
        return xhat, mu, logvar


def train_cvae(X: np.ndarray, y: np.ndarray, n_classes: int,
               lat: int = 6, h: int = 32, epochs: int = 250,
               beta: float = 0.1, seed: int = SEED) -> CVAE:
    """
    Trains the CVAE on the [0, 1]-scaled training fold.
    Loss: L = ||x - x_hat||^2 + beta * KL( q(z|x,y) || N(0,I) )
    """
    torch.manual_seed(seed)
    d = X.shape[1]
    model = CVAE(d, min(lat, d), h, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    Xt = torch.tensor(X, dtype=torch.float32)
    C = torch.tensor(np.eye(n_classes)[y], dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        xhat, mu, logvar = model(Xt, C)
        recon = ((xhat - Xt) ** 2).sum(1).mean()
        kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)).mean()
        (recon + beta * kl).backward()
        opt.step()
    return model


def cvae_sample(model: CVAE, y: np.ndarray, n_classes: int, seed: int = 7) -> np.ndarray:
    """Draws z ~ N(0, I) from the prior and decodes it conditioned on the given labels."""
    torch.manual_seed(seed)
    with torch.no_grad():
        C = torch.tensor(np.eye(n_classes)[y], dtype=torch.float32)
        z = torch.randn(len(y), model.lat)
        return model.dec(torch.cat([z, C], 1)).numpy()


# --------------------- Hybrid augmentation (CVAE + SMOTE) ---------------------
def hybrid_augment(X: np.ndarray, y: np.ndarray, n_classes: int, seed: int = SEED):
    """
    Returns the augmented training set: (X_aug, y_aug, n_smote, n_cvae).
    X is expected to be scaled to [0, 1]. SMOTE balances the classes; the CVAE
    adds one synthetic sample per training sample (with the original label).
    """
    X_sm, y_sm = SMOTE(k_neighbors=_k_neighbors(y), random_state=seed).fit_resample(X, y)
    model = train_cvae(X, y, n_classes, seed=seed)
    X_cv = cvae_sample(model, y, n_classes, seed=7)
    X_aug = np.vstack([X_sm, X_cv])
    y_aug = np.concatenate([y_sm, y])
    return X_aug, y_aug, len(X_sm), len(X_cv)
