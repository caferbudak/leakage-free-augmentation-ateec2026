"""
ATEEC 2026 — reproduction of the hybrid augmentation (CVAE + SMOTE) experiments.

All tables are produced by ONE consistent protocol:
  * 5-fold x 2-repeat stratified cross-validation (fixed seed).
  * Preprocessing (missForest imputation, min-max scaling) and augmentation
    are fitted on the training fold ONLY; the test fold is transformed with
    the training-fold estimates.
  * 6 core classifiers (main result) + an extended set of 15 classifiers (robustness).
  * Naive protocol: augment the full data -> then CV (demonstrates data leakage).

Run:
    python src/run_experiments.py              # 7 development datasets
    python src/run_experiments.py --external   # 4 external validation datasets
                                               # (frozen method, never used in method selection)
Output:
    results/results_unified_5x2.csv   (development, raw per-fold results)
    results/results_external_5x2.csv  (external validation, raw per-fold results)
    console: Tables II/III/IV, Wilcoxon tests, D_aug composition.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier, BaggingClassifier)
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from hybrid_augmentation import hybrid_augment, minmax_fit, minmax_apply, SEED
from missforest import MissForest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "datasets_verified")
RESULTS = os.path.join(ROOT, "results")
OUT_NAME = "results_unified_5x2.csv"

DATASETS = ["pelvic", "hepatitis", "parkinsons", "heart_disease",
            "pima_diabetes", "glass", "sonar"]

# External validation: datasets selected before seeing any result, never used in
# method selection (public UCI, N <= 1000, binary or binarised, numeric features).
EXTERNAL_DATA = os.path.join(ROOT, "data", "external_validation")
EXTERNAL_DATASETS = ["ionosphere", "haberman", "ecoli", "wdbc"]
EXTERNAL_OUT_NAME = "results_external_5x2.csv"
N_SPLITS, N_REPEATS = 5, 2


def core_classifiers():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=SEED),
        "SVC (RBF)": SVC(random_state=SEED),
        "k-NN": KNeighborsClassifier(),
        "RandomForest": RandomForestClassifier(300, random_state=SEED),
        "ExtraTrees": ExtraTreesClassifier(300, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, eval_metric="logloss",
                                 random_state=SEED, verbosity=0),
    }


def extended_classifiers():
    clf = core_classifiers()
    clf.update({
        "Ridge": RidgeClassifier(random_state=SEED),
        "LinearSVC": LinearSVC(random_state=SEED, max_iter=5000),
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=SEED),
        "AdaBoost": AdaBoostClassifier(random_state=SEED),
        "Bagging": BaggingClassifier(random_state=SEED),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),
        "LightGBM": LGBMClassifier(n_estimators=300, random_state=SEED, verbose=-1),
    })
    return clf


def _safe_predict(proto, Xa, ya, Xte):
    try:
        c = clone(proto); c.fit(Xa, ya); return c.predict(Xte)
    except Exception:
        return None


def run():
    rows, comp = [], []
    core = core_classifiers()
    allc = extended_classifiers()
    for ds in DATASETS:
        d = pd.read_csv(os.path.join(DATA, f"{ds}.csv"))
        X = d.iloc[:, :-1].values.astype(float)
        y = d.iloc[:, -1].values.astype(int)
        nc = len(np.unique(y))

        # --- NAIVE (plain accuracy): augment full data -> 5x2 CV (leakage) ---
        Ximp = MissForest(random_state=SEED).fit_transform(X)      # imputed on the FULL data
        mn, rg = minmax_fit(Ximp)
        Xall = minmax_apply(Ximp, mn, rg)
        Xh, yh, _, _ = hybrid_augment(Xall, y, nc)
        for tr, te in RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                              random_state=SEED).split(Xh, yh):
            for nm, proto in core.items():
                p = _safe_predict(proto, Xh[tr], yh[tr], Xh[te])
                if p is not None:
                    rows.append(dict(ds=ds, model=nm, proto="naive",
                                     acc=accuracy_score(yh[te], p), bal=np.nan, f1=np.nan))

        # --- BASELINE + LEAK-FREE (5x2) ---
        for tr, te in RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                              random_state=SEED).split(X, y):
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            mf = MissForest(random_state=SEED).fit(Xtr)                # training fold only
            Xtr_i, Xte_i = mf.train_imputed_, mf.transform(Xte)
            m, r = minmax_fit(Xtr_i)
            Xs, Xts = minmax_apply(Xtr_i, m, r), minmax_apply(Xte_i, m, r)
            Xa, ya, n_sm, n_cv = hybrid_augment(Xs, ytr, nc)
            if not any(c["ds"] == ds for c in comp):   # first fold of each dataset
                comp.append(dict(ds=ds, n_tr=len(Xs), smote=n_sm, cvae=n_cv,
                                 total=len(Xa), cls_tr=np.bincount(ytr).tolist(),
                                 cls_aug=np.bincount(ya).tolist()))
            for nm, proto in allc.items():
                pb = _safe_predict(proto, Xs, ytr, Xts)
                pv = _safe_predict(proto, Xa, ya, Xts)
                if pb is not None:
                    rows.append(dict(ds=ds, model=nm, proto="base",
                                     acc=accuracy_score(yte, pb),
                                     bal=balanced_accuracy_score(yte, pb),
                                     f1=f1_score(yte, pb, average="macro")))
                if pv is not None:
                    rows.append(dict(ds=ds, model=nm, proto="lf",
                                     acc=accuracy_score(yte, pv),
                                     bal=balanced_accuracy_score(yte, pv),
                                     f1=f1_score(yte, pv, average="macro")))
        print(f"[done] {ds}", flush=True)

    R = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    R.to_csv(os.path.join(RESULTS, OUT_NAME), index=False)
    report(R, comp, list(core.keys()))


def report(R, comp, CORE):
    def cm(sub, val):
        return sub.groupby(["ds", "model"])[val].mean()
    c6 = R[R.model.isin(CORE)]
    base = cm(c6[c6.proto == "base"], "acc")
    naive = cm(c6[c6.proto == "naive"], "acc")
    lf = cm(c6[c6.proto == "lf"], "acc")

    print("\n=== TABLE II (per classifier, plain accuracy %, 6 core) ===")
    for m in CORE:
        b, n, l = base.xs(m, level=1).mean(), naive.xs(m, level=1).mean(), lf.xs(m, level=1).mean()
        print(f"  {m:14s} Orig {b*100:.2f}  Naive {n*100:.2f} ({(n-b)*100:+.2f})  LeakFree {l*100:.2f} ({(l-b)*100:+.2f})")
    B, N, L = base.groupby('model').mean().mean(), naive.groupby('model').mean().mean(), lf.groupby('model').mean().mean()
    print(f"  MEAN           Orig {B*100:.2f}  Naive {N*100:.2f} ({(N-B)*100:+.2f})  LeakFree {L*100:.2f} ({(L-B)*100:+.2f})")

    print("\n=== TABLE III (per dataset, plain accuracy %, 6 core) ===")
    for ds in DATASETS:
        b, n, l = base.xs(ds, level=0).mean(), naive.xs(ds, level=0).mean(), lf.xs(ds, level=0).mean()
        print(f"  {ds:14s} Orig {b*100:.2f}  Naive {n*100:.2f} ({(n-b)*100:+.2f})  LeakFree {l*100:.2f} ({(l-b)*100:+.2f})")

    print("\n=== TABLE IV (6 core, leakage-free: Baseline -> Hybrid augmentation) ===")
    for val, nm in [("bal", "Balanced accuracy"), ("f1", "Macro-F1"), ("acc", "Plain accuracy")]:
        b = cm(c6[c6.proto == "base"], val); v = cm(c6[c6.proto == "lf"], val)
        print(f"  {nm:16s} {b.mean()*100:.2f} -> {v.mean()*100:.2f}  (Δ {(v.mean()-b.mean())*100:+.2f})")
    bb = cm(c6[c6.proto == "base"], "bal"); vv = cm(c6[c6.proto == "lf"], "bal")
    dd = (vv - bb).dropna()
    wc = stats.wilcoxon(vv, bb)
    gb = c6[c6.proto == "base"].groupby("ds").bal.mean()
    gv = c6[c6.proto == "lf"].groupby("ds").bal.mean()
    wd = stats.wilcoxon(gv, gb)
    print(f"  Pos/neg cells: {(dd>0).sum()}/{(dd<0).sum()}")
    print(f"  Wilcoxon cells({len(dd)}) p={wc.pvalue:.2e} (supplementary) ; "
          f"datasets({len(gb)}) p={wd.pvalue:.3f} (MAIN)")

    b15 = cm(R[R.proto == "base"], "bal"); v15 = cm(R[R.proto == "lf"], "bal")
    print(f"\n=== ROBUSTNESS (15 classifiers) ΔBA = {(v15.mean()-b15.mean())*100:+.2f} pp "
          f"(pos/neg {((v15-b15)>0).sum()}/{((v15-b15)<0).sum()}) ===")

    print("\n=== D_aug COMPOSITION (example fold) ===")
    print(pd.DataFrame(comp).to_string(index=False))


if __name__ == "__main__":
    import sys
    if "--external" in sys.argv:
        DATA, DATASETS, OUT_NAME = EXTERNAL_DATA, EXTERNAL_DATASETS, EXTERNAL_OUT_NAME
    run()
