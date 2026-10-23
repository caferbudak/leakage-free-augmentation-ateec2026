# Leakage-Free Evaluation of Variational Autoencoder and SMOTE-Based Oversampling for Small and Imbalanced Tabular Data

This repository contains the complete code, data and results of the **ATEEC 2026**
paper "Leakage-Free Evaluation of Variational Autoencoder and SMOTE-Based
Oversampling for Small and Imbalanced Tabular Data" (*Assoc. Prof. Cafer Budak*,
Dicle University), in a fully **reproducible** form. The paper itself is written
in Turkish (`paper/`).

## Method

**Hybrid augmentation** (CVAE + SMOTE) enlarges the training set with synthetic
samples from two complementary sources:

- **SMOTE** (Synthetic Minority Over-sampling Technique) — balances the classes
  by linear interpolation between minority-class samples (inside the convex hull).
- **CVAE** (class-conditional variational autoencoder) — adds diversity beyond
  the convex hull by sampling from the prior (`z ~ N(0, I)`) of the learned
  class-conditional latent distribution.

Augmented training set: `D_aug = D_SMOTE ∪ D_CVAE`. Loss:
`L = ||x − x̂||² + β · D_KL(q(z|x,y) || N(0, I))`.

**Evaluation principle (leakage-free):** the data are first split into
stratified training and test folds; preprocessing (missForest imputation, min–max
scaling to [0, 1]) and augmentation are estimated/applied **on the training fold
only**, and the test fold is kept out of every step. In addition, a **naive**
protocol (augmenting the full data before splitting) shows how data leakage
inflates performance.

## Datasets (7 UCI, binary)

| Dataset | N | d | Class distribution | Imbalance ratio | Missing cells |
|---|---:|---:|---:|---:|---:|
| Pelvic | 310 | 6 | 210 / 100 | 2.10 | 0 |
| Hepatitis | 155 | 19 | 123 / 32 | 3.84 | 167 |
| Parkinson's | 195 | 22 | 48 / 147 | 3.06 | 0 |
| Heart Disease | 303 | 13 | 164 / 139 | 1.18 | 6 |
| Pima Diabetes | 768 | 8 | 500 / 268 | 1.87 | 0 |
| Glass | 214 | 9 | 51 / 163 | 3.20 | 0 |
| Sonar | 208 | 60 | 97 / 111 | 1.14 | 0 |

Glass is binarised as window vs. non-window glass. The verified datasets are in
`data/datasets_verified/` (last column: `target`); `_validation_summary.csv`
documents the source and coding of each dataset.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

If PyTorch fails to install, see <https://pytorch.org/get-started/locally/> for
the build matching your platform.

## Reproduction

```bash
# Produces Tables II, III and IV, the Wilcoxon tests and the D_aug composition;
# writes the raw per-fold results to results/results_unified_5x2.csv.
python src/run_experiments.py

# Produces the data-driven figures (ΔBA bar chart, Pelvic synthetic-data quality).
python src/make_figures.py
```

Protocol: **5-fold × 2-repeat stratified cross-validation**, fixed seed
(`SEED=123`). Missing values (Hepatitis, Heart Disease) are imputed with
missForest (Stekhoven & Bühlmann, *Bioinformatics* 28(1):112–118, 2012;
random forests with 100 trees, regression for continuous and classification for
binary/low-level variables), fitted on the training fold and applied to the test fold. CVAE: latent dimension 6, hidden layer 32, 250 epochs, `β=0.1`.
Main evaluation with 6 core classifiers; 15 classifiers for robustness.

## Expected results (leakage-free, 6 core classifiers)

| Metric | Baseline | Hybrid augmentation | Δ (percentage points) |
|---|---:|---:|---:|
| Balanced accuracy | 79.62 | 82.11 | **+2.49** |
| Macro-F1 | 80.35 | 81.47 | +1.11 |
| Plain accuracy | 84.21 | 84.03 | −0.18 |

The gain is positive on all 7 datasets and in 36 of the 42 dataset–classifier
combinations. **Dataset-level paired Wilcoxon** test (main test): `p = 0.016`.
The naive protocol artificially inflates plain accuracy by ~6.7 percentage
points; this gain disappears under the leakage-free protocol.

> Note: small differences may arise from BLAS/threading and library versions;
> the overall pattern and the statistical conclusions do not change.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── data/datasets_verified/      # 7 UCI datasets (+ validation summary)
├── src/
│   ├── hybrid_augmentation.py   # method: CVAE + SMOTE
│   ├── missforest.py            # missForest imputation (fit on training fold)
│   ├── run_experiments.py       # reproduces all tables (5x2)
│   └── make_figures.py          # data-driven figures
├── results/results_unified_5x2.csv
├── figures/                     # Figure 1 (flow), Figure 2 (CVAE architecture)
└── paper/Budak_ATEEC_2026_v8.docx
```

Figure 1 (method flow) and Figure 2 (CVAE architecture) are schematic drawings;
`src/make_figures.py` produces the result and synthetic-data quality figures.

## Citation

> C. Budak, "Leakage-Free Evaluation of Variational Autoencoder and SMOTE-Based
> Oversampling for Small and Imbalanced Tabular Data," ATEEC 2026.

## License

MIT — see [LICENSE](LICENSE).
