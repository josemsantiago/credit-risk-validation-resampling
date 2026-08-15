"""Feature construction, encoding, imputation, scaling, and selection.

Everything a model sees is built here, and everything is fitted on training rows
only. The three pieces that decide whether the pipeline is honest are the
imputation constants, the target encoding, and the scaler: each is estimated
inside the training partition and then applied outward, so no test loan ever
contributes to the numbers used to transform it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer      # noqa: F401
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

import config as C

EMP_LENGTH = {
    "< 1 year": 0.0, "1 year": 1.0, "2 years": 2.0, "3 years": 3.0,
    "4 years": 4.0, "5 years": 5.0, "6 years": 6.0, "7 years": 7.0,
    "8 years": 8.0, "9 years": 9.0, "10+ years": 10.0,
}

# A blank mths_since_last_delinq means no delinquency was ever recorded, which
# is a fact about the borrower rather than a gap in the file. Filling it with a
# median would place these borrowers among those who did default recently, so it
# is filled at a ceiling beyond the observed range and flagged by an indicator.
NEVER_CEILING = 200.0


def build(frame):
    """Assemble the origination-time design frame: the raw numeric columns, the
    derived ratios, the informative-missingness indicators, and the categorical
    columns still in their raw form for the encoders downstream."""
    d = pd.DataFrame(index=frame.index)

    for col in C.NUMERIC_RAW:
        d[col] = pd.to_numeric(frame[col], errors="coerce")

    # ---- credit age, the one feature that needs both dates -------------------
    age_days = (frame["issue_date"] - frame["earliest_cr_date"]).dt.days
    d["credit_age_years"] = age_days / 365.25

    # ---- income-relative burden ---------------------------------------------
    inc = frame["annual_inc"].replace(0, np.nan)
    d["log_annual_inc"] = np.log1p(frame["annual_inc"].clip(lower=0))
    d["loan_to_income"] = frame["loan_amnt"] / inc
    d["installment_to_income"] = frame["installment"] * 12.0 / inc
    # LendingClub's reported dti excludes the loan being applied for. Adding this
    # loan's own annualized payment gives the burden the borrower would carry.
    d["total_debt_burden"] = frame["dti"] + 100.0 * d["installment_to_income"]
    d["revol_bal_to_income"] = frame["revol_bal"] / inc
    d["log_revol_bal"] = np.log1p(frame["revol_bal"].clip(lower=0))

    # ---- account structure ---------------------------------------------------
    total_acc = frame["total_acc"].replace(0, np.nan)
    d["open_acc_ratio"] = frame["open_acc"] / total_acc

    # ---- payment history, with its missingness made explicit ----------------
    d["never_delinquent"] = frame["mths_since_last_delinq"].isna().astype(float)
    d["months_since_delinq"] = frame["mths_since_last_delinq"].fillna(NEVER_CEILING)
    d["no_public_record"] = frame["mths_since_last_record"].isna().astype(float)
    d["months_since_record"] = frame["mths_since_last_record"].fillna(NEVER_CEILING)

    # ---- employment ----------------------------------------------------------
    d["emp_length_missing"] = frame["emp_length"].isna().astype(float)
    d["emp_length_years"] = frame["emp_length"].map(EMP_LENGTH)

    # ---- categoricals, raw; encoded by the fitted Encoder --------------------
    home = frame["home_ownership"].where(
        ~frame["home_ownership"].isin(C.HOME_OWNERSHIP_OTHER), "OTHER")
    d["home_ownership"] = home
    for col in ["verification_status", "grade"] + C.TARGET_ENCODE:
        d[col] = frame[col].astype(str)

    return d


def _interaction_name(a, b):
    return f"{a}_x_{b}"


class Encoder:
    """Fits every constant a design matrix needs, on training rows only.

    Imputation, one-hot levels, target encodings, interaction terms, and the
    scaler are all learned in `fit` and replayed in `transform`. The imputation
    and scaling strategies are arguments rather than choices, because the study
    compares them.
    """

    def __init__(self, imputation="median", scaling="standard", seed=C.SEED,
                 target_encode=True, oof_target_encoding=True,
                 interactions=True, missing_treatment="indicator",
                 extra_numeric=()):
        # Extra columns to carry through untouched. Used only by the leakage
        # experiment, which feeds in the post-origination fields on purpose.
        self.extra_numeric = list(extra_numeric)
        self.imputation = imputation
        self.scaling = scaling
        self.seed = seed
        self.target_encode = target_encode
        self.oof_target_encoding = oof_target_encoding
        self.interactions = interactions
        # How the two informative gaps are handled. "indicator" keeps the ceiling
        # fill and the flag that says the value was never recorded; "median"
        # throws the flag away and lets the imputer fill from the observed
        # distribution, which places a borrower who was never delinquent among
        # those who were delinquent a median time ago; "drop" removes both
        # columns and the flag. The three are compared in the report.
        self.missing_treatment = missing_treatment

    # ------------------------------------------------------------------ fit
    def fit(self, design, y):
        y = np.asarray(y).astype(int)
        self.onehot_levels_ = {
            col: sorted(design[col].dropna().unique()) for col in C.ONEHOT}

        # ---- target encoding, fitted out of fold ---------------------------
        # An encoding fitted on all the training rows and then applied to those
        # same rows lets each loan's own outcome into its own feature. Fitting it
        # on out-of-fold rows removes that, and the study measures what the
        # difference is worth by running it both ways.
        self.te_maps_, self.te_prior_ = {}, float(y.mean())
        if self.target_encode:
            for col in C.TARGET_ENCODE:
                self.te_maps_[col] = self._te_map(design[col], y)

        numeric = self._numeric_frame(design, y, fitting=True)

        # ---- imputation ------------------------------------------------------
        self.imputer_ = self._make_imputer()
        self.imputer_.fit(numeric.to_numpy(dtype=float))

        filled = pd.DataFrame(self.imputer_.transform(numeric.to_numpy(dtype=float)),
                              columns=numeric.columns, index=numeric.index)
        full = self._add_interactions(filled)
        self.columns_ = list(full.columns)

        # ---- scaling ---------------------------------------------------------
        self.scaler_ = {"standard": StandardScaler, "minmax": MinMaxScaler,
                        "robust": RobustScaler}[self.scaling]()
        self.scaler_.fit(full.to_numpy(dtype=float))
        return self

    # ------------------------------------------------------------ transform
    def transform(self, design):
        numeric = self._numeric_frame(design, None, fitting=False)
        numeric = numeric[self._numeric_columns_]
        filled = pd.DataFrame(self.imputer_.transform(numeric.to_numpy(dtype=float)),
                              columns=numeric.columns, index=numeric.index)
        full = self._add_interactions(filled)[self.columns_]
        return self.scaler_.transform(full.to_numpy(dtype=float))

    def fit_transform(self, design, y):
        return self.fit(design, y).transform(design)

    # -------------------------------------------------------------- internals
    def _make_imputer(self):
        if self.imputation == "median":
            return SimpleImputer(strategy="median")
        if self.imputation == "mean":
            return SimpleImputer(strategy="mean")
        if self.imputation == "iterative":
            # Chained equations: each column with gaps is regressed on the others
            # and filled from that fit, cycling until the fills stop moving. This
            # is the single-completion form of multiple imputation, which is what
            # a deterministic pipeline can carry.
            return IterativeImputer(max_iter=10, random_state=self.seed,
                                    sample_posterior=False, initial_strategy="median")
        if self.imputation == "knn":
            return KNNImputer(n_neighbors=5)
        raise ValueError(f"unknown imputation {self.imputation}")

    def _te_map(self, series, y):
        """Smoothed out-of-fold target encoding for one column.

        The stored map is the full-training-partition encoding, which is what
        test rows get; the training rows themselves are encoded from the folds
        that exclude them, so a row never sees its own outcome. Smoothing pulls
        a level with few loans toward the global default rate, in proportion to
        how few.
        """
        prior = float(y.mean())
        agg = pd.DataFrame({"level": series.to_numpy(), "y": y}).groupby("level")["y"]
        stats = agg.agg(["sum", "count"])
        smoothed = (stats["sum"] + C.TE_SMOOTHING * prior) / (stats["count"] + C.TE_SMOOTHING)
        return smoothed.to_dict()

    def _te_oof(self, series, y):
        """Encode the training rows themselves, each from folds that exclude it."""
        values = np.full(len(series), np.nan)
        cv = StratifiedKFold(n_splits=C.TE_FOLDS, shuffle=True, random_state=self.seed)
        arr = series.to_numpy()
        for inner_train, inner_val in cv.split(arr.reshape(-1, 1), y):
            m = self._te_map(series.iloc[inner_train], y[inner_train])
            prior = float(y[inner_train].mean())
            values[inner_val] = [m.get(v, prior) for v in arr[inner_val]]
        return values

    def _informative_missing_columns(self):
        """Which of the informative-missingness columns survive, and how."""
        ceilings = ["months_since_delinq", "months_since_record"]
        flags = ["never_delinquent", "no_public_record"]
        if self.missing_treatment == "drop":
            return [], []
        if self.missing_treatment == "median":
            return ceilings, []                    # ceiling restored to NaN below
        return ceilings, flags

    def _numeric_frame(self, design, y, fitting):
        """Every column as a float, with the categoricals encoded."""
        ceilings, flags = self._informative_missing_columns()
        keep = [c for c in C.NUMERIC_RAW + C.NUMERIC_DERIVED
                if c not in ("months_since_delinq", "months_since_record")]
        keep += ceilings
        keep += [c for c in C.MISSING_INDICATORS
                 if c in flags or c == "emp_length_missing"]
        keep += [c for c in self.extra_numeric if c in design.columns]
        base = design[keep].astype(float).copy()
        if self.missing_treatment == "median":
            for col in ceilings:                   # let the imputer fill them instead
                base.loc[base[col] >= NEVER_CEILING, col] = np.nan
        parts = [base]

        onehot = {}
        for col in C.ONEHOT:
            for level in self.onehot_levels_[col]:
                onehot[f"{col}={level}"] = (design[col] == level).astype(float)
        parts.append(pd.DataFrame(onehot, index=design.index))

        if self.target_encode:
            te = {}
            for col in C.TARGET_ENCODE:
                if fitting and self.oof_target_encoding:
                    te[f"{col}_te"] = self._te_oof(design[col], y)
                else:
                    te[f"{col}_te"] = design[col].map(self.te_maps_[col]).fillna(
                        self.te_prior_).to_numpy()
            parts.append(pd.DataFrame(te, index=design.index))

        out = pd.concat(parts, axis=1)
        if fitting:
            self._numeric_columns_ = list(out.columns)
        return out

    def _add_interactions(self, filled):
        if not self.interactions:
            return filled
        out = filled.copy()
        for a, b in C.INTERACTIONS:
            out[_interaction_name(a, b)] = filled[a].to_numpy() * filled[b].to_numpy()
        return out


def build_leaky(frame):
    """The same design frame with the post-origination columns added back.

    Built only so the study can measure what including them is worth. Every
    column here is a function of what the borrower did after the money was lent,
    so a model fitted on this frame is reporting the outcome, not predicting it.
    """
    d = build(frame)
    for col in C.POST_ORIGINATION:
        if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col]):
            d[f"leak_{col}"] = pd.to_numeric(frame[col], errors="coerce")
    return d


LEAKY_COLUMNS = [f"leak_{c}" for c in C.POST_ORIGINATION]


def feature_names(encoder):
    return list(encoder.columns_)


def pretty(name):
    """Table-ready label for a design-matrix column."""
    if "_x_" in name:
        a, b = name.split("_x_")
        return f"{pretty(a)} x {pretty(b)}"
    if name.endswith("_te"):
        return C.LABELS.get(name, name[:-3].replace("_", " ").capitalize()
                            + " (target-encoded)")
    if "=" in name:
        col, level = name.split("=", 1)
        return f"{col.replace('_', ' ').capitalize()}: {level.title()}"
    return C.LABELS.get(name, name.replace("_", " ").capitalize())


# ===================================================== feature selection
def select_mutual_information(X, y, names, k, seed):
    """Mutual information between each feature and the outcome, estimated by the
    k-nearest-neighbour method for a continuous predictor against a discrete
    target. It scores each feature alone, so it sees non-linear dependence but
    not redundancy between two features that carry the same thing."""
    scores = mutual_info_classif(X, y, random_state=seed)
    order = np.argsort(scores)[::-1]
    return dict(method="Mutual information",
                scores={names[i]: float(scores[i]) for i in range(len(names))},
                selected=[names[i] for i in order[:k]])


def select_lasso(X, y, names, k, seed):
    """L1-penalized logistic regression, with the penalty tightened until exactly
    k coefficients survive. Selection is joint rather than marginal, so a feature
    duplicated elsewhere is dropped rather than double-counted."""
    lo, hi = 1e-4, 10.0
    best = None
    for _ in range(40):                      # bisection on the inverse penalty
        mid = np.sqrt(lo * hi)
        m = LogisticRegression(penalty="l1", C=mid, solver="liblinear",
                               max_iter=C.LOGIT_MAX_ITER, random_state=seed).fit(X, y)
        n = int((np.abs(m.coef_[0]) > 1e-8).sum())
        best = m
        if n == k:
            break
        if n < k:
            lo = mid
        else:
            hi = mid
    coef = np.abs(best.coef_[0])
    order = np.argsort(coef)[::-1]
    return dict(method="LASSO",
                scores={names[i]: float(coef[i]) for i in range(len(names))},
                selected=[names[i] for i in order[:k]])


def select_rfe(X, y, names, k, seed, step=0.1):
    """Recursive feature elimination around an L2 logistic model: fit, drop the
    weakest tenth, refit, repeat. Unlike the marginal scores it re-ranks what
    remains after every removal, so it accounts for what a dropped feature was
    standing in for."""
    est = LogisticRegression(C=C.LOGIT_C, max_iter=C.LOGIT_MAX_ITER,
                             random_state=seed)
    sel = RFE(est, n_features_to_select=k, step=step).fit(X, y)
    rank = sel.ranking_
    return dict(method="RFE",
                scores={names[i]: float(-rank[i]) for i in range(len(names))},
                selected=[names[i] for i in np.argsort(rank)[:k]])


def selection_agreement(selections):
    """Which features every method keeps, and which only one does."""
    sets = {s["method"]: set(s["selected"]) for s in selections}
    everything = sorted(set().union(*sets.values()))
    rows = []
    for name in everything:
        picked = [m for m, s in sets.items() if name in s]
        rows.append(dict(feature=pretty(name), variable=name,
                         n_methods=len(picked), methods=", ".join(sorted(picked))))
    return pd.DataFrame(rows).sort_values(
        ["n_methods", "feature"], ascending=[False, True]).reset_index(drop=True)
