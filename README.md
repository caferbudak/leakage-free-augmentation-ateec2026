# Leakage-Free Evaluation of Variational Autoencoder and SMOTE-Based Oversampling for Small and Imbalanced Tabular Data

This repository contains the complete code, data and results of the **ATEEC 2026**
paper "Leakage-Free Evaluation of Variational Autoencoder and SMOTE-Based
Oversampling for Small and Imbalanced Tabular Data" (*Assoc. Prof. Cafer Budak*,
Dicle University), in a fully **reproducible** form. The paper itself is written
in Turkish (`paper/`). All results in the paper were produced with release
**`v1.0-ateec2026`** of this repository.

## Method

**Hybrid augmentation** (CVAE + SMOTE) enlarges the training set with synthetic
samples from two complementary sources:

- **SMOTE** (Synthetic Minority Over-sampling Technique) — balances the classes
  by linear interpolation between minority-class samples (inside the convex hull).
- **CVAE** (class-conditional variational autoencoder) — adds diversity beyond
  the convex hull by sampling from the prior (`z ~ N(0, I)`) of the learned
  class-conditional latent distribution.

`D_SMOTE` contains the original training samples and the SMOTE minority samples;
`D_CVAE` contains one CVAE sample per training sample. The augmented training set
is their row-wise concatenation with duplicates kept: `D_aug = [D_SMOTE ; D_CVAE]`.
Loss: `L = ||x − x̂||² + β · D_KL(q(z|x,y) || N(0, I))`.

**Evaluation principle (leakage-free):** the data are first split into
stratified training and test folds; preprocessing (missForest imputation, min–max
scaling to [0, 1]) and augmentation are estimated/applied **on the training fold
only**, and the test fold is kept out of every step. In addition, a **naive**
protocol (imputation, scaling and augmentation of the full data before splitting)
shows how data leakage inflates performance.

## Method selection and external validation

The hybrid augmentation and the CVAE settings were selected in exploratory
screening experiments (about 25 candidate augmentation methods) on the seven
**development** datasets; results on these datasets are therefore exploratory.
To test the effect of this selection, the method and all settings were frozen and
evaluated with the same protocol on four **external validation** datasets that
were never used in method selection. They were chosen, before any result was seen,
by the rule: public UCI datasets not used in screening, N ≤ 1000, binary or
binarised, numeric features.

## Datasets

| Dataset | Role | N | d | Class distribution | Imbalance ratio | Missing cells |
|---|---|---:|---:|---:|---:|---:|
| Pelvic | development | 310 | 6 | 210 / 100 | 2.10 | 0 |
| Hepatitis | development | 155 | 19 | 123 / 32 | 3.84 | 167 |
| Parkinson's | development | 195 | 22 | 48 / 147 | 3.06 | 0 |
| Heart Disease | development | 303 | 13 | 164 / 139 | 1.18 | 6 |
| Pima Diabetes | development | 768 | 8 | 500 / 268 | 1.87 | 0 |
| Glass | development | 214 | 9 | 51 / 163 | 3.20 | 0 |
| Sonar | development | 208 | 60 | 97 / 111 | 1.14 | 0 |
| Ionosphere | external | 351 | 34 | 225 / 126 | 1.79 | 0 |
| Haberman | external | 306 | 3 | 225 / 81 | 2.78 | 0 |
| Ecoli (imU vs rest) | external | 336 | 7 | 301 / 35 | 8.60 | 0 |
| WDBC | external | 569 | 30 | 357 / 212 | 1.68 | 0 |

Glass is binarised as window vs. non-window glass; Ecoli as imU vs. all other
classes (imbalanced-learn benchmark definition). Development datasets are in
`data/datasets_verified/`, external validation datasets in
`data/external_validation/` (last column: `target`); each folder has a
`_validation_summary.csv` documenting source and coding.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

If PyTorch fails to install, see <https://pytorch.org/get-started/locally/> for
the build matching your platform.

## Reproduction

```bash
# Development datasets: Tables II and III, Wilcoxon tests, D_aug composition;
# raw per-fold results -> results/results_unified_5x2.csv
python src/run_experiments.py

# External validation (frozen method): raw per-fold results
# -> results/results_external_5x2.csv
python src/run_experiments.py --external

# Table IV (per-dataset balanced accuracy and macro-F1, Δ ± SD over folds) and
# Supplementary Table S1 (all dataset × classifier results)
# -> results/supplementary_table_S1.csv
python src/make_supplementary.py

# Data-driven figures (ΔBA bar chart, Pelvic synthetic-data quality)
python src/make_figures.py
```

Protocol: **5-fold × 2-repeat stratified cross-validation**, fixed seed
(`SEED=123`). Missing values (Hepatitis, Heart Disease) are imputed with
missForest (Stekhoven & Bühlmann, *Bioinformatics* 28(1):112–118, 2012;
random forests with 100 trees, regression for continuous and classification for
binary/low-level variables), fitted on the training fold and applied to the test
fold. CVAE: latent dimension 6, hidden layer 32, 250 epochs, `β=0.1`.
Main evaluation with 6 core classifiers; 15 classifiers for robustness.
Aggregation: the 10 folds of each dataset–classifier pair are averaged; dataset
values are the mean over the 6 core classifiers; the Wilcoxon test is run on the
dataset-level values.

## Expected results (leakage-free, 6 core classifiers, balanced accuracy %)

| Datasets | Baseline | Hybrid augmentation | Δ (percentage points) |
|---|---:|---:|---:|
| Development (7) | 79.62 | 82.11 | **+2.49** |
| External validation (4) | 77.70 | 83.31 | **+5.61** |

- Development: gain positive on all 7 datasets and in 36 of 42 dataset–classifier
  combinations; dataset-level paired Wilcoxon `p = 0.016`. Macro-F1 +1.11.
- External validation: gain positive on all 4 datasets and in 21 of 24
  combinations (`p = 0.125`, the smallest attainable value for n = 4); all 11
  datasets pooled: 11/11 positive, `p = 0.001`. Macro-F1 +3.79.
- The naive protocol artificially inflates plain accuracy by ~6.7 (development)
  and ~5.4 (external) percentage points; this gain disappears under the
  leakage-free protocol.

> Note: small differences may arise from BLAS/threading and library versions;
> the overall pattern and the statistical conclusions do not change.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   ├── datasets_verified/       # 7 development datasets (+ validation summary)
│   └── external_validation/     # 4 external validation datasets (+ summary)
├── src/
│   ├── hybrid_augmentation.py   # method: CVAE + SMOTE
│   ├── missforest.py            # missForest imputation (fit on training fold)
│   ├── run_experiments.py       # all experiments (5x2); --external for validation
│   ├── make_supplementary.py    # Table IV and Supplementary Table S1
│   └── make_figures.py          # data-driven figures
├── results/
│   ├── results_unified_5x2.csv  # development, raw per-fold results
│   ├── results_external_5x2.csv # external validation, raw per-fold results
│   └── supplementary_table_S1.csv
├── figures/                     # Figure 1 (flow), Figure 2 (CVAE architecture)
└── paper/Budak_ATEEC_2026_v10.docx
```

Figure 1 (method flow) and Figure 2 (CVAE architecture) are schematic drawings;
`src/make_figures.py` produces the result and synthetic-data quality figures.

## Citation

> C. Budak, "Leakage-Free Evaluation of Variational Autoencoder and SMOTE-Based
> Oversampling for Small and Imbalanced Tabular Data," ATEEC 2026.
> Code and data: https://github.com/caferbudak/leakage-free-augmentation-ateec2026 (release v1.0-ateec2026)

## License

MIT — see [LICENSE](LICENSE).
