"""
Produces the data-driven figures:
  figures/figure3_delta_ba.png        : leakage-free ΔBA per dataset (bar chart).
  figures/figure_pelvic_pca.png       : Pelvic — original vs synthetic (2-D PCA).
  figures/figure_pelvic_features.png  : Pelvic — feature distributions (original vs synthetic).

Note: Figure 1 (method flow) and Figure 2 (CVAE architecture) in the paper are
schematic drawings by the author; they are provided in the figures/ folder.

Run: python src/make_figures.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

from hybrid_augmentation import train_cvae, cvae_sample, minmax_fit, minmax_apply, _k_neighbors

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data", "datasets_verified")
DATASETS = ["pelvic", "hepatitis", "parkinsons", "heart_disease", "pima_diabetes", "glass", "sonar"]
CORE = ["LogisticRegression", "SVC (RBF)", "k-NN", "RandomForest", "ExtraTrees", "XGBoost"]
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})


def delta_ba_bar():
    R = pd.read_csv(os.path.join(RESULTS, "results_unified_5x2.csv"))
    c6 = R[R.model.isin(CORE)]
    b = c6[c6.proto == "base"].groupby("ds").bal.mean()
    v = c6[c6.proto == "lf"].groupby("ds").bal.mean()
    delta = ((v - b) * 100).reindex(DATASETS)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#2e9e6b" if x >= 0 else "#c0392b" for x in delta]
    ax.bar(range(len(delta)), delta.values, color=colors)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(range(len(delta)))
    ax.set_xticklabels(delta.index, rotation=25, ha="right")
    ax.set_ylabel("Δ balanced accuracy (percentage points)")
    ax.set_title("Leakage-free gain of hybrid augmentation (mean of 6 core classifiers)")
    for i, x in enumerate(delta.values):
        ax.text(i, x + (0.1 if x >= 0 else -0.25), f"{x:+.1f}", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "figure3_delta_ba.png"), bbox_inches="tight")
    plt.close()


def synthetic_quality(ds="pelvic"):
    d = pd.read_csv(os.path.join(DATA, f"{ds}.csv"))
    feats = list(d.columns[:-1])
    X = d[feats].values.astype(float); y = d.iloc[:, -1].values.astype(int)
    nc = len(np.unique(y))
    mn, rg = minmax_fit(X); Xs = minmax_apply(X, mn, rg)
    Xsm, ysm = SMOTE(k_neighbors=_k_neighbors(y), random_state=123).fit_resample(Xs, y)
    Xsm_new = Xsm[np.arange(len(Xsm)) >= len(Xs)]
    v = train_cvae(Xs, y, nc, seed=123); Xcv = cvae_sample(v, y, nc, seed=7)
    # PCA
    pca = PCA(2).fit(Xs)
    Po, Psm, Pcv = pca.transform(Xs), pca.transform(Xsm_new), pca.transform(Xcv)
    ev = pca.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(figsize=(7, 5.6))
    ax.scatter(Po[:, 0], Po[:, 1], s=34, c="#9aa0a6", edgecolor="#5f6368", lw=0.4, alpha=0.85, label=f"Original (N={len(Xs)})")
    ax.scatter(Psm[:, 0], Psm[:, 1], s=28, marker="^", c="#1a73e8", alpha=0.7, label="SMOTE")
    ax.scatter(Pcv[:, 0], Pcv[:, 1], s=30, marker="x", c="#e8710a", alpha=0.85, label="CVAE")
    ax.set_xlabel(f"PCA 1 (%{ev[0]:.1f})"); ax.set_ylabel(f"PCA 2 (%{ev[1]:.1f})")
    ax.set_title(f"{ds} — Original vs synthetic (PCA)"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f"figure_{ds}_pca.png"), bbox_inches="tight"); plt.close()
    # feature distributions
    inv = lambda Z: Z * rg + mn
    Xsyn = np.vstack([inv(Xsm_new), inv(Xcv)]); Xo = inv(Xs)
    n = len(feats); cols = 3; rowsn = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rowsn, cols, figsize=(4 * cols, 2.3 * rowsn))
    for i, ax in enumerate(np.array(axes).ravel()):
        if i >= n:
            ax.axis("off"); continue
        ax.hist(Xo[:, i], bins=20, density=True, alpha=0.4, color="#5f6368", label="Original")
        ax.hist(Xsyn[:, i], bins=20, density=True, histtype="step", lw=1.8, color="#e8710a", label="Synthetic")
        ax.set_title(feats[i].replace("_", " "), fontsize=9)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"{ds} — Feature distributions: original vs synthetic (hybrid augmentation)", y=1.0)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f"figure_{ds}_features.png"), bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    delta_ba_bar()
    synthetic_quality("pelvic")
    print("Figures written to the figures/ folder.")
