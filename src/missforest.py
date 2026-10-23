"""
missForest imputation (Stekhoven & Bühlmann, Bioinformatics 28(1):112-118, 2012)
with an inductive extension: the imputer is fitted on the training fold and the
stored per-variable forests are replayed on the test fold, so the test fold
never contributes to any estimate.

Algorithm (as in the original paper):
  * Initial guess: mean (continuous) / mode (categorical) of the observed
    training values.
  * Variables with missing values are visited in order of increasing number
    of missing entries; for each one a random forest (regression for
    continuous, classification for categorical variables) is fitted on the
    rows where it is observed, using all other variables as predictors, and
    its missing entries are predicted.
  * Passes are repeated until the change between successive imputations
    increases for the first time for both variable types (if present):
      Delta_N = sum_{j in N} (X_new - X_old)^2 / sum_{j in N} X_new^2
      Delta_F = #changed categorical entries / #missing categorical entries
    and the imputation of the previous pass is returned (max_iter passes).
  * Defaults: 100 trees, mtry = floor(sqrt(p)).

Categorical variables: integer-valued variables with at most `max_levels`
distinct observed values in the training fold (binary and low-level codes).
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class MissForest:
    def __init__(self, n_estimators: int = 100, max_iter: int = 10,
                 max_levels: int = 5, random_state: int = 123):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.max_levels = max_levels
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        self.p_ = X.shape[1]
        miss = np.isnan(X)
        self.cat_ = np.zeros(self.p_, dtype=bool)
        self.init_ = np.zeros(self.p_)
        for j in range(self.p_):
            obs = X[~miss[:, j], j]
            levels = np.unique(obs)
            self.cat_[j] = (len(levels) <= self.max_levels and np.all(levels == np.round(levels)))
            if self.cat_[j]:
                vals, cnt = np.unique(obs, return_counts=True)
                self.init_[j] = vals[np.argmax(cnt)]
            else:
                self.init_[j] = obs.mean()
        # visiting order: increasing number of missing values (only variables with NaN)
        nmiss = miss.sum(0)
        self.order_ = [j for j in np.argsort(nmiss, kind="stable") if nmiss[j] > 0]
        self.passes_ = []                      # list of passes; each pass = {j: model}
        if not self.order_:
            self.train_imputed_ = X.copy()
            return self
        Xi = np.where(miss, self.init_, X)
        mtry = max(1, int(np.floor(np.sqrt(self.p_ - 1)))) if self.p_ > 1 else 1
        catN, conN = self.cat_[self.order_], ~self.cat_[self.order_]
        prev_dN, prev_dF = np.inf, np.inf
        X_prev, passes_prev = Xi.copy(), []
        for it in range(self.max_iter):
            X_old = Xi.copy()
            models = {}
            for j in self.order_:
                rows_obs, rows_mis = ~miss[:, j], miss[:, j]
                others = [k for k in range(self.p_) if k != j]
                Est = RandomForestClassifier if self.cat_[j] else RandomForestRegressor
                m = Est(n_estimators=self.n_estimators, max_features=mtry,
                        random_state=self.random_state + it, n_jobs=-1)
                m.fit(Xi[rows_obs][:, others], X[rows_obs, j])
                Xi[rows_mis, j] = m.predict(Xi[rows_mis][:, others])
                models[j] = m
            # stopping criterion
            cols = np.array(self.order_)
            con, cat = cols[conN], cols[catN]
            dN = (np.sum((Xi[:, con] - X_old[:, con]) ** 2) / max(np.sum(Xi[:, con] ** 2), 1e-12)
                  if len(con) else 0.0)
            dF = (np.sum(Xi[:, cat][miss[:, cat]] != X_old[:, cat][miss[:, cat]]) / max(miss[:, cat].sum(), 1)
                  if len(cat) else 0.0)
            increased = ((dN > prev_dN) or not len(con)) and ((dF > prev_dF) or not len(cat))
            if it > 0 and increased:
                Xi, self.passes_ = X_prev, passes_prev   # return the previous pass
                break
            X_prev, passes_prev = Xi.copy(), self.passes_ + [models]
            self.passes_ = passes_prev
            prev_dN, prev_dF = dN, dF
        self.train_imputed_ = Xi
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Imputes new data by replaying the stored passes (training-fold models)."""
        X = np.asarray(X, dtype=float)
        miss = np.isnan(X)
        if not miss.any():
            return X.copy()
        Xi = np.where(miss, self.init_, X)
        for models in self.passes_:
            for j in self.order_:
                rows_mis = miss[:, j]
                if rows_mis.any():
                    others = [k for k in range(self.p_) if k != j]
                    Xi[rows_mis, j] = models[j].predict(Xi[rows_mis][:, others])
        # a variable that had no missing values in training but has some here
        # keeps its training mean/mode (initial guess)
        return Xi

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).train_imputed_
