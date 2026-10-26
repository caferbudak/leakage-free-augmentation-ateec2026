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
- **CVAE** (class-conditional variational autoencoder) — is intended to add
  diversity beyond the convex hull by sampling from the prior (`z ~ N(0, I)`)
  and decoding conditioned on the class label (the fraction of samples that
  actually fall outside the hull is not quantified in the paper).

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

Class distribution is given as class 0 / class 1 of the `target` column.
Development datasets are in `data/datasets_verified/`, external validation
datasets in `data/external_validation/` (last column: `target`); each folder has a
`_validation_summary.csv` documenting source and coding. These CSV files are the
exact inputs of all experiments.

### Sources and target coding

| Dataset | Source (UCI ID) | DOI | `target` = 1 | `target` = 0 |
|---|---|---|---|---|
| Pelvic | Vertebral Column (212), 2-class file | [10.24432/C5K89B](https://doi.org/10.24432/C5K89B) | normal (NO) | abnormal (AB) |
| Hepatitis | Hepatitis (46) | [10.24432/C5Q59J](https://doi.org/10.24432/C5Q59J) | die | live |
| Parkinson's | Parkinsons (174) | [10.24432/C59C74](https://doi.org/10.24432/C59C74) | Parkinson's (`status` = 1) | healthy |
| Heart Disease | Heart Disease (45), Cleveland | [10.24432/C52P4X](https://doi.org/10.24432/C52P4X) | disease present (`num` 1–4) † | absent (`num` = 0) |
| Pima Diabetes | Pima Indians Diabetes (NIDDK; classic UCI distribution, also OpenML data set 37) | — ‡ | diabetes | no diabetes |
| Glass | Glass Identification (42) | [10.24432/C5WW2P](https://doi.org/10.24432/C5WW2P) | window glass (types 1–4; no type-4 records) † | non-window glass (types 5–7) |
| Sonar | Connectionist Bench (Sonar, Mines vs. Rocks) (151) | [10.24432/C5T01Q](https://doi.org/10.24432/C5T01Q) | mine (M) | rock (R) |
| Ionosphere | Ionosphere (52) | [10.24432/C5W01B](https://doi.org/10.24432/C5W01B) | bad (b) | good (g) |
| Haberman | Haberman's Survival (43) | [10.24432/C5XK51](https://doi.org/10.24432/C5XK51) | died within 5 years (class 2) | survived ≥ 5 years (class 1) |
| Ecoli | Ecoli (39) | [10.24432/C5388M](https://doi.org/10.24432/C5388M) | imU † | all other classes |
| WDBC | Breast Cancer Wisconsin (Diagnostic) (17), via scikit-learn's bundled copy | [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B) | malignant | benign |

† Binary target derived by the author from the original multi-class label
(Heart Disease: presence vs. absence, the usual Cleveland definition; Glass:
window vs. non-window grouping of the dataset documentation; Ecoli: imU vs. rest,
imbalanced-learn benchmark definition). ‡ No UCI DOI could be verified for Pima.
Other preparation steps: identifier columns removed (Parkinson's `name`);
Hepatitis yes/no and sex codes recoded to 1/0; missing values are kept in the CSV
files and imputed inside each training fold.

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

Protocol details:

- **Splits:** 5-fold stratified cross-validation repeated twice,
  `RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=123)`. The
  baseline and the leakage-free hybrid augmentation use the same 10 splits for
  every classifier; the naive protocol splits the augmented pool with the same
  generator.
- **Seeds:** 123 for the splits, classifiers, missForest, SMOTE and CVAE
  training; 7 for CVAE sampling. One CVAE is trained and sampled per fold.
- **Imputation:** missForest (Stekhoven & Bühlmann, *Bioinformatics*
  28(1):112–118, 2012), adapted to the train/test split: random forests with 100
  trees (regression for continuous variables, classification for variables with
  at most 5 distinct integer values in the training fold), at most 10 iterations,
  fitted on the training fold; test-fold missing values are initialised with the
  training-fold mean/mode and then filled by replaying the stored forests.
- **Scaling:** min–max to [0, 1] with training-fold statistics; test values
  outside the training range are clipped.
- **SMOTE:** minority class oversampled to the majority size,
  `k_neighbors = min(5, n_minority − 1)`.
- **CVAE:** one hidden layer (32 tanh units) in encoder and decoder, latent
  dimension `min(6, d)` (3 for Haberman, where d = 3), sigmoid output, Adam
  (learning rate 5·10⁻³), `β = 0.1`, 250 full-batch epochs.
- **Classifiers:** 6 core classifiers (main result) and 15 classifiers
  (robustness), configured in `core_classifiers()` and
  `extended_classifiers()` of `src/run_experiments.py`: RandomForest,
  ExtraTrees, XGBoost and LightGBM with 300 trees; otherwise library defaults
  (except convergence limits and `reg_param=0.1` for QDA).
- **Aggregation:** the 10 folds of each dataset–classifier pair are averaged;
  dataset values are the mean over the 6 core classifiers; the Wilcoxon test is
  run on the dataset-level values.

Raw result files (`results/*_5x2.csv`) contain one row per dataset ×
classifier × protocol × fold with the columns `ds, model, proto, repeat, fold,
acc, bal, f1` (`proto`: `base` = no augmentation, `lf` = leakage-free hybrid
augmentation, `naive` = augmentation before splitting; `bal` = balanced
accuracy, `f1` = macro-F1).

## Expected results (leakage-free, 6 core classifiers, balanced accuracy %)

| Datasets | Baseline | Hybrid augmentation | Δ (percentage points) |
|---|---:|---:|---:|
| Development (7) | 79.62 | 82.11 | **+2.49** |
| External validation (4) | 77.70 | 83.31 | **+5.61** |

- Development: gain positive on all 7 datasets and in 36 of 42 dataset–classifier
  combinations; dataset-level paired Wilcoxon `p = 0.016`. Macro-F1 +1.11.
- External validation: gain positive on all 4 datasets and in 21 of 24
  combinations (`p = 0.125`, the smallest attainable value for n = 4); all 11
  datasets pooled: 11/11 positive, `p = 0.001` (smallest attainable value for
  n = 11). The pooled test includes the seven development datasets used in
  method selection and is therefore not independent validation evidence.
  Macro-F1 +3.79.
- The naive protocol artificially inflates plain accuracy by ~6.7 (development)
  and ~5.4 (external) percentage points; this gain disappears under the
  leakage-free protocol.

**Tested environment:** Python 3.13.17 with the exact package versions in
`requirements.txt`, CPU only. A clean re-run of all scripts in this environment
(26 Oct 2026) reproduced the published per-fold results and
`supplementary_table_S1.csv` exactly (identical metric values in every row;
total run time about 9 minutes on CPU). With other platforms or library versions, small numerical
differences may arise from BLAS/threading.

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
