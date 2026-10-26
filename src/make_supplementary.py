"""
Builds the per-dataset summary used in Table IV of the paper and the full
dataset x classifier breakdown (Supplementary Table S1) from the raw results.

Inputs : results/results_unified_5x2.csv   (development datasets)
         results/results_external_5x2.csv  (external validation datasets)
Outputs: results/supplementary_table_S1.csv
         console: Table IV rows (6 core classifiers; Δ ± SD over the 10 folds)

Aggregation: for every dataset-classifier pair the 10 folds (5-fold x 2 repeats)
are averaged; dataset values are the mean over the 6 core classifiers; the SD of
Δ is computed over folds after averaging the 6 core classifiers within each fold.

Run: python src/make_supplementary.py
"""
from __future__ import annotations
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
CORE = ["LogisticRegression", "SVC (RBF)", "k-NN", "RandomForest", "ExtraTrees", "XGBoost"]
SETS = [("development", "results_unified_5x2.csv"), ("external", "results_external_5x2.csv")]


def load():
    frames = []
    for name, fn in SETS:
        R = pd.read_csv(os.path.join(RESULTS, fn))
        R = R[R.proto.isin(["base", "lf"])].copy()
        R["set"] = name
        frames.append(R)
    return pd.concat(frames, ignore_index=True)


def supplementary(R):
    g = R.groupby(["set", "ds", "model", "proto"])[["bal", "f1"]].mean().unstack("proto") * 100
    S = pd.DataFrame({
        "ba_baseline": g[("bal", "base")], "ba_hybrid": g[("bal", "lf")],
        "ba_delta": g[("bal", "lf")] - g[("bal", "base")],
        "f1_baseline": g[("f1", "base")], "f1_hybrid": g[("f1", "lf")],
        "f1_delta": g[("f1", "lf")] - g[("f1", "base")],
    }).round(2).reset_index()
    S["core"] = S.model.isin(CORE)
    S.to_csv(os.path.join(RESULTS, "supplementary_table_S1.csv"), index=False)
    return S


def table_iv(R):
    C = R[R.model.isin(CORE)]
    per_fold = C.groupby(["set", "ds", "repeat", "fold", "proto"])[["bal", "f1"]].mean().unstack("proto") * 100
    d = pd.DataFrame({"dba": per_fold[("bal", "lf")] - per_fold[("bal", "base")],
                      "df1": per_fold[("f1", "lf")] - per_fold[("f1", "base")]})
    cell = C.groupby(["set", "ds", "model", "proto"])[["bal", "f1"]].mean().unstack("proto") * 100
    ds_mean = cell.groupby(["set", "ds"]).mean()
    rows = []
    for (s, ds), r in ds_mean.iterrows():
        rows.append(dict(set=s, ds=ds,
                         ba_base=r[("bal", "base")], ba_hyb=r[("bal", "lf")],
                         dba=r[("bal", "lf")] - r[("bal", "base")], dba_sd=d.loc[(s, ds)].dba.std(),
                         f1_base=r[("f1", "base")], f1_hyb=r[("f1", "lf")],
                         df1=r[("f1", "lf")] - r[("f1", "base")], df1_sd=d.loc[(s, ds)].df1.std()))
    T = pd.DataFrame(rows)
    return T


if __name__ == "__main__":
    R = load()
    S = supplementary(R)
    T = table_iv(R)
    pd.set_option("display.width", 200)
    for s in ["development", "external"]:
        X = T[T.set == s].drop(columns="set")
        print(f"\n=== TABLE IV — {s} (6 core, %, Δ ± SD over folds) ===")
        print(X.round(2).to_string(index=False))
        m = X.drop(columns=["ds", "dba_sd", "df1_sd"]).mean()
        print("  MEAN", m.round(2).to_dict())
    print(f"\nSupplementary Table S1 written: {len(S)} rows")
