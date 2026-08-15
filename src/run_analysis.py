"""Orchestrator: run the whole credit-risk validation study end to end.

Verifies and scopes the LendingClub file, measures what the three leakage routes
are worth, compares imputation and scaling choices, selects features three ways,
runs seven cross-validation protocols against each other, applies ten treatments
of the class imbalance inside every training fold, puts bootstrap confidence
intervals and an out-of-bag estimate around the result, and takes the errors
apart by borrower, by risk decile, and by what each one costs. Writes every
figure, CSV table, and a single results.json.

    cd src && python3 run_analysis.py        # full run
    QUICK=1 python3 run_analysis.py          # reduced budgets; exercises every path
"""
import json
import os
import time
import warnings

# Pin the linear-algebra backend to one thread before NumPy loads, which is the
# only point at which it can be set. Two reasons, and the second was a surprise.
# Determinism: a threaded reduction leaves its summation order to the scheduler,
# which moves the last bits of every probability. Speed: almost every fit in this
# study is small, and on a 299 x 48 matrix the thread handshake costs far more
# than the arithmetic. A hundred such fits take 8.25 seconds across eight threads
# and 0.14 seconds on one. The large fits do not lose by it, and the resamplers
# get their parallelism from joblib instead, where it is order-independent.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import bootstrap as B
import config as C
import data as D
import errors as E
import features as F
import models as M
import plots as PL
import resample as R
import validate as V

# Scope suppression: the pinned stack's Future/Deprecation churn, and the
# convergence notices liblinear emits while the LASSO bisection walks the
# penalty down to the requested feature count.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
warnings.filterwarnings("ignore", module="sklearn")
np.random.seed(C.SEED)

GRID_MODELS = ["Logistic regression", "Gradient boosting"]
HEADLINE_MODEL = "Logistic regression"
IMPUTATION_SUBSAMPLE = 15_000                # k-nearest-neighbour imputation is O(n^2)
SELECT_K = 15                                # features each selector is asked for
GRID_FOLDS = 5                               # folds the resampling grid runs on
LEAKAGE_METHODS = ["Random over", "SMOTE", "SMOTE-ENN"]

QUICK = bool(os.environ.get("QUICK"))
if QUICK:                                    # smoke budgets; every code path still runs
    C.BOOTSTRAP_N = 60
    C.BOOTSTRAP_MODEL_N = 4
    C.BOOTSTRAP_JACKKNIFE_N = 120
    C.LOOCV_N = 250
    C.CV_REPEATS = 2
    IMPUTATION_SUBSAMPLE = 2_500
    GRID_FOLDS = 2
    QUICK_OOB_N = 8_000
    # The boundary cleaners are quadratic in the fold size, so the smoke run
    # gives them a fold they can finish. Every method still runs.
    QUICK_GRID_N = 6_000
    QUICK_LEAKAGE_N = 6_000


def _round(o, nd=4):
    """Recursively round floats so results.json is clean and diff-friendly."""
    if isinstance(o, (float, np.floating)):
        return round(float(o), nd)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    if isinstance(o, np.ndarray):
        return _round(o.tolist(), nd)
    if isinstance(o, pd.DataFrame):
        return _round(o.to_dict(orient="records"), nd)
    return o


def _write(df, name):
    df.to_csv(C.OUT_DIR / f"table_{name}.csv", index=False)
    return df


def _stage(label, t0):
    print(f"  [{time.time() - t0:7.1f}s] {label}", flush=True)


def main():
    t0 = time.time()
    results = {"seed": C.SEED, "quick": QUICK}
    rng = np.random.RandomState(C.SEED)

    # ================================================================== data
    raw = D.load_raw()
    _write(D.status_breakdown(raw), "loan_status")
    funnel = _write(D.cohort_funnel(raw), "cohort_funnel")
    cohort = D.get_cohort(raw)
    vintages = _write(D.describe_cohort(cohort), "vintages")
    costs = D.price_errors(cohort)
    leak_audit = _write(D.leakage_audit(cohort), "leakage_audit")
    _stage(f"cohort: {len(cohort):,} loans, {cohort[C.TARGET].mean()*100:.2f}% default", t0)

    results["cohort"] = dict(
        raw_loans=int(len(raw)), loans=int(len(cohort)),
        default_rate=float(cohort[C.TARGET].mean()),
        imbalance_ratio=float((1 - cohort[C.TARGET].mean()) / cohort[C.TARGET].mean()),
        defaults=int(cohort[C.TARGET].sum()),
        majority_baseline_accuracy=float(1 - cohort[C.TARGET].mean()),
        first_issue=str(cohort["issue_date"].min().date()),
        last_issue=str(cohort["issue_date"].max().date()),
        issue_months=int(cohort["issue_date"].dt.to_period("M").nunique()),
        funnel=funnel.to_dict(orient="records"))
    results["costs"] = costs
    results["leakage_audit"] = leak_audit.to_dict(orient="records")
    results["excluded_columns"] = dict(
        post_origination=len(C.POST_ORIGINATION),
        not_yet_collected=len(C.NOT_YET_COLLECTED),
        non_predictive=len(C.NON_PREDICTIVE),
        total=len(C.POST_ORIGINATION) + len(C.NOT_YET_COLLECTED)
        + len(C.NON_PREDICTIVE),
        source_columns=C.RAW_COLS)

    design = F.build(cohort)
    y_all = cohort[C.TARGET].to_numpy().astype(int)
    train_idx, test_idx = _split_indices(cohort, C.SEED)
    d_tr, d_te = design.iloc[train_idx], design.iloc[test_idx]
    y_tr, y_te = y_all[train_idx], y_all[test_idx]
    f_tr, f_te = cohort.iloc[train_idx], cohort.iloc[test_idx]
    results["split"] = dict(n_train=int(len(y_tr)), n_test=int(len(y_te)),
                            train_default_rate=float(y_tr.mean()),
                            test_default_rate=float(y_te.mean()))

    # ================================================================== leakage
    results["leakage"] = _leakage_experiments(cohort, design, y_all, d_tr, y_tr,
                                              d_te, y_te, rng, t0)

    # ============================================== preprocessing and features
    results["imputation"] = _imputation_experiment(design, y_all, rng, t0)
    results["scaling"] = _scaling_experiment(d_tr, y_tr, d_te, y_te, t0)
    selection, results["selection"] = _selection_experiment(d_tr, y_tr, d_te, y_te, t0)

    # ================================================================== the model
    enc = F.Encoder(seed=C.SEED).fit(d_tr, y_tr)
    names = F.feature_names(enc)
    Xtr, Xte = enc.transform(d_tr), enc.transform(d_te)
    baseline = {}
    for name in M.MODELS:
        p, model = M.fit_predict(name, Xtr, y_tr, Xte, C.SEED)
        baseline[name] = p
    baseline["Prior (predict the base rate)"] = np.full(len(y_te), float(y_tr.mean()))
    base_table = _write(pd.DataFrame(
        [dict(model=n, **M.evaluate(y_te, p)) for n, p in baseline.items()]
    ).sort_values("roc_auc", ascending=False).reset_index(drop=True), "model_comparison")
    results["model_comparison"] = base_table.to_dict(orient="records")
    results["n_features"] = len(names)
    _stage(f"baseline models fitted ({len(names)} features)", t0)

    p_head = baseline[HEADLINE_MODEL]

    # ================================================================== cross-validation
    results["cross_validation"], cv_records, sweep = _cv_experiments(
        d_tr, y_tr, f_tr, t0)
    _write(sweep, "stratification_sweep")

    # ================================================================== resampling
    (results["resampling"], grid_table, resample_shapes, neigh,
     holdout_resample, y_grid, p_grid_oof) = _resampling_experiments(
        d_tr, y_tr, d_te, y_te, t0)

    # ================================================================== bootstrap
    results["bootstrap"] = _bootstrap_experiments(
        d_tr, y_tr, y_te, p_head, names, t0)

    # ================================================================== errors
    results["errors"] = _error_experiments(f_te, y_te, p_head, baseline, costs,
                                           y_grid, p_grid_oof, t0)

    # ================================================================== figures
    _stage("drawing figures", t0)
    PL.fig_cohort(raw, cohort, funnel, vintages, C.FIG_DIR / "fig01_cohort.png")
    PL.fig_leakage(leak_audit, results["leakage"], C.FIG_DIR / "fig02_leakage.png")
    PL.fig_features(design, cohort, selection, results["imputation"],
                    results["selection"], C.FIG_DIR / "fig03_features.png")
    PL.fig_crossval(cv_records, sweep, results["cross_validation"],
                    C.FIG_DIR / "fig04_crossvalidation.png")
    PL.fig_resampling(resample_shapes, grid_table, neigh,
                      C.FIG_DIR / "fig05_resampling.png")
    PL.fig_resampling_effect(holdout_resample, results["resampling"], y_te,
                             C.FIG_DIR / "fig06_resampling_effect.png")
    PL.fig_bootstrap(results["bootstrap"], C.FIG_DIR / "fig07_bootstrap.png")
    PL.fig_errors(results["errors"], y_te, p_head, C.FIG_DIR / "fig08_errors.png")
    PL.fig_operating_point(results["errors"], results["costs"],
                           C.FIG_DIR / "fig09_operating_point.png")

    # ================================================================== persist
    results["runtime_sec"] = round(time.time() - t0, 2)
    with open(C.OUT_DIR / "results.json", "w") as f:
        json.dump(_round(results), f, indent=2)
    _summary(results, base_table)


# ===================================================== helpers
def _split_indices(cohort, seed):
    """A stratified 70/30 hold-out, with both partitions returned in issue-date
    order.

    The cohort is sorted by issue date, so sorting the indices back after the
    shuffle costs nothing and buys the forward-chaining protocol its meaning:
    TimeSeriesSplit splits on row position, and on a shuffled partition that is
    not a split in time at all.
    """
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(cohort))
    tr, te = train_test_split(idx, test_size=C.TEST_SIZE, random_state=seed,
                              stratify=cohort[C.TARGET].to_numpy())
    return np.sort(tr), np.sort(te)


def _fit_score(d_tr, y_tr, d_te, y_te, model=HEADLINE_MODEL, seed=C.SEED, **enc_kw):
    enc = F.Encoder(seed=seed, **enc_kw).fit(d_tr, y_tr)
    p, _ = M.fit_predict(model, enc.transform(d_tr), y_tr, enc.transform(d_te), seed)
    return p, M.evaluate(y_te, p)


# ===================================================== leakage
def _leakage_experiments(cohort, design, y_all, d_tr, y_tr, d_te, y_te, rng, t0):
    """Three ways the outcome can reach the training set, each priced in AUC."""
    out = {}

    # ---- 1. post-origination columns ---------------------------------------
    leaky = F.build_leaky(cohort)
    tr_idx, te_idx = _split_indices(cohort, C.SEED)
    enc_l = F.Encoder(seed=C.SEED, extra_numeric=F.LEAKY_COLUMNS).fit(
        leaky.iloc[tr_idx], y_tr)
    p_leak, _ = M.fit_predict(HEADLINE_MODEL, enc_l.transform(leaky.iloc[tr_idx]),
                              y_tr, enc_l.transform(leaky.iloc[te_idx]), C.SEED)
    _, honest = _fit_score(d_tr, y_tr, d_te, y_te)

    # `recoveries` on its own, with no other feature at all.
    rec = cohort["recoveries"].to_numpy(dtype=float)
    only_rec = M.ranking_metrics(y_te, (rec[te_idx] > 0).astype(float))

    rows = [
        dict(feature_set="Origination-time features only", roc_auc=honest["roc_auc"],
             pr_auc=honest["pr_auc"], note="what a lender could actually have known"),
        dict(feature_set="Plus post-origination columns",
             roc_auc=M.ranking_metrics(y_te, p_leak)["roc_auc"],
             pr_auc=M.ranking_metrics(y_te, p_leak)["pr_auc"],
             note="16 columns recorded after the money was lent"),
        dict(feature_set="`recoveries` > 0 alone", roc_auc=only_rec["roc_auc"],
             pr_auc=only_rec["pr_auc"],
             note="one post-origination column, used as the entire model"),
    ]
    _write(pd.DataFrame(rows), "leakage_features")
    out["features"] = rows
    out["feature_inflation"] = float(
        M.ranking_metrics(y_te, p_leak)["roc_auc"] - honest["roc_auc"])
    _stage("leakage: post-origination features", t0)

    # ---- 2. target encoding fitted in sample --------------------------------
    _, oof = _fit_score(d_tr, y_tr, d_te, y_te, oof_target_encoding=True)
    _, naive = _fit_score(d_tr, y_tr, d_te, y_te, oof_target_encoding=False)
    enc_oof = F.Encoder(seed=C.SEED, oof_target_encoding=True).fit(d_tr, y_tr)
    enc_naive = F.Encoder(seed=C.SEED, oof_target_encoding=False).fit(d_tr, y_tr)
    p_in_oof, _ = M.fit_predict(HEADLINE_MODEL, enc_oof.transform(d_tr), y_tr,
                                enc_oof.transform(d_tr), C.SEED)
    p_in_naive, _ = M.fit_predict(HEADLINE_MODEL, enc_naive.transform(d_tr), y_tr,
                                  enc_naive.transform(d_tr), C.SEED)
    te_rows = [
        dict(encoding="Out of fold", apparent_roc_auc=M.ranking_metrics(y_tr, p_in_oof)["roc_auc"],
             holdout_roc_auc=oof["roc_auc"]),
        dict(encoding="Fitted in sample", apparent_roc_auc=M.ranking_metrics(y_tr, p_in_naive)["roc_auc"],
             holdout_roc_auc=naive["roc_auc"]),
    ]
    for r in te_rows:
        r["optimism"] = r["apparent_roc_auc"] - r["holdout_roc_auc"]
    _write(pd.DataFrame(te_rows), "leakage_target_encoding")
    out["target_encoding"] = te_rows
    _stage("leakage: target encoding", t0)

    # ---- 3. resampling before the split -------------------------------------
    # Both arms see the same loans and the same split proportion. The only
    # difference is the order: one resamples the whole cohort and then splits it,
    # the other splits first and resamples the training side alone.
    if QUICK:
        keep = np.sort(np.random.RandomState(C.SEED).choice(
            len(y_all), QUICK_LEAKAGE_N, replace=False))
        design, y_all = design.iloc[keep].reset_index(drop=True), y_all[keep]
        tr_q, te_q = _split_indices(design.assign(**{C.TARGET: y_all}), C.SEED)
        d_tr, y_tr = design.iloc[tr_q], y_all[tr_q]
        d_te, y_te = design.iloc[te_q], y_all[te_q]

    enc = F.Encoder(seed=C.SEED).fit(d_tr, y_tr)
    X_all = enc.transform(design)
    Xtr_f, Xte_f = enc.transform(d_tr), enc.transform(d_te)
    rows = []
    for method in LEAKAGE_METHODS:
        Xa, ya, Xb, yb = R.resample_before_split(
            X_all, y_all, method, C.SEED, C.TEST_SIZE, np.random.RandomState(C.SEED))
        p_wrong, _ = M.fit_predict(HEADLINE_MODEL, Xa, ya, Xb, C.SEED)
        wrong = M.ranking_metrics(yb, p_wrong)["roc_auc"]

        Xr, yr, w, _ = R.apply(method, Xtr_f, y_tr, C.SEED)
        p_right, _ = M.fit_predict(HEADLINE_MODEL, Xr, yr, Xte_f, C.SEED,
                                   class_weight=w)
        right = M.ranking_metrics(y_te, p_right)["roc_auc"]
        rows.append(dict(method=method, resampled_before_split=wrong,
                         resampled_inside_the_fold=right, inflation=wrong - right))
    _write(pd.DataFrame(rows), "leakage_resampling_order")
    out["resampling_order"] = rows
    _stage("leakage: resampling order", t0)
    return out


# ===================================================== preprocessing
def _imputation_experiment(design, y_all, rng, t0):
    """Two questions that both get called imputation and are not the same.

    The first is what to do about the handful of columns with a fraction of a
    percent missing, where the answer barely matters. The second is what to do
    about the two columns whose blanks mean something, where it matters a great
    deal.
    """
    n = min(IMPUTATION_SUBSAMPLE, len(y_all))
    idx = rng.choice(len(y_all), n, replace=False)
    d, y = design.iloc[idx].reset_index(drop=True), y_all[idx]
    cut = int(0.7 * n)
    d_tr, d_te, y_tr, y_te = d.iloc[:cut], d.iloc[cut:], y[:cut], y[cut:]

    rows = []
    for strategy in ["median", "mean", "iterative", "knn"]:
        s = time.time()
        _, m = _fit_score(d_tr, y_tr, d_te, y_te, imputation=strategy)
        rows.append(dict(question="Sparse accidental gaps", choice=strategy,
                         roc_auc=m["roc_auc"], pr_auc=m["pr_auc"], brier=m["brier"],
                         seconds=round(time.time() - s, 1)))
    for treatment, label in [("indicator", "ceiling fill + flag"),
                             ("median", "median fill, no flag"),
                             ("drop", "columns dropped")]:
        s = time.time()
        _, m = _fit_score(d_tr, y_tr, d_te, y_te, missing_treatment=treatment)
        rows.append(dict(question="Informative blanks", choice=label,
                         roc_auc=m["roc_auc"], pr_auc=m["pr_auc"], brier=m["brier"],
                         seconds=round(time.time() - s, 1)))
    table = _write(pd.DataFrame(rows), "imputation")
    _stage("imputation comparison", t0)

    # Missingness as the source file carries it, not as the design frame does.
    # `build` has already filled the two informative columns at their ceiling, so
    # reading their rate off the design frame would report zero and hide the
    # larger of the two problems.
    missing = {c: float(design[c].isna().mean()) for c in design.columns
               if design[c].dtype.kind == "f" and design[c].isna().any()}
    missing["months_since_delinq"] = float(design["never_delinquent"].mean())
    missing["months_since_record"] = float(design["no_public_record"].mean())
    informative = ["months_since_delinq", "months_since_record"]
    return dict(subsample=int(n), rows=table.to_dict(orient="records"),
                missing_rates=missing, informative=informative)


def _scaling_experiment(d_tr, y_tr, d_te, y_te, t0):
    """Scaling is cosmetic for the trees and decisive for anything that measures
    a distance, which is every synthetic oversampler in the grid."""
    rows = []
    for scaling in ["standard", "minmax", "robust"]:
        for model in ["Logistic regression", "Gradient boosting"]:
            _, m = _fit_score(d_tr, y_tr, d_te, y_te, model=model, scaling=scaling)
            rows.append(dict(scaling=scaling, model=model, roc_auc=m["roc_auc"],
                             brier=m["brier"]))
    # What the scaler does to a distance-based resampler, which is the reason it
    # is not a free choice here.
    for scaling in ["standard", "minmax", "robust"]:
        enc = F.Encoder(seed=C.SEED, scaling=scaling).fit(d_tr, y_tr)
        Xr, yr, _, _ = R.apply("SMOTE", enc.transform(d_tr), y_tr, C.SEED)
        p, _ = M.fit_predict(HEADLINE_MODEL, Xr, yr, enc.transform(d_te), C.SEED)
        m = M.evaluate(y_te, p)
        rows.append(dict(scaling=scaling, model="Logistic regression + SMOTE",
                         roc_auc=m["roc_auc"], brier=m["brier"]))
    table = _write(pd.DataFrame(rows), "scaling")
    _stage("scaling comparison", t0)
    return table.to_dict(orient="records")


def _selection_experiment(d_tr, y_tr, d_te, y_te, t0):
    enc = F.Encoder(seed=C.SEED).fit(d_tr, y_tr)
    X, names = enc.transform(d_tr), F.feature_names(enc)
    Xte = enc.transform(d_te)

    selections = [
        F.select_mutual_information(X, y_tr, names, SELECT_K, C.SEED),
        F.select_lasso(X, y_tr, names, SELECT_K, C.SEED),
        F.select_rfe(X, y_tr, names, SELECT_K, C.SEED),
    ]
    agreement = _write(F.selection_agreement(selections), "selection_agreement")

    rows = []
    for s in selections:
        cols = [names.index(f) for f in s["selected"]]
        p, _ = M.fit_predict(HEADLINE_MODEL, X[:, cols], y_tr, Xte[:, cols], C.SEED)
        m = M.evaluate(y_te, p)
        rows.append(dict(method=s["method"], k=SELECT_K, roc_auc=m["roc_auc"],
                         pr_auc=m["pr_auc"], brier=m["brier"],
                         top_features=", ".join(F.pretty(f) for f in s["selected"][:5])))
    p_full, _ = M.fit_predict(HEADLINE_MODEL, X, y_tr, Xte, C.SEED)
    m_full = M.evaluate(y_te, p_full)
    rows.append(dict(method="All features", k=len(names), roc_auc=m_full["roc_auc"],
                     pr_auc=m_full["pr_auc"], brier=m_full["brier"], top_features="n/a"))
    table = _write(pd.DataFrame(rows), "selection_performance")

    # ---- how much of the model is LendingClub's own underwriting ------------
    # The interest rate, the grade, and the sub-grade are not borrower
    # attributes: they are the price LendingClub set after running its own risk
    # model. A lender building a scorecard from scratch would not have them, so
    # it is worth knowing how much of the discrimination they carry alone.
    lc_own = [n for n in names if n.startswith("grade=") or n == "int_rate"
              or n == "sub_grade_te" or n.startswith("int_rate_x_")]
    blocks = [("LendingClub's own price and grade", lc_own),
              ("Borrower attributes only", [n for n in names if n not in lc_own]),
              ("Both", names)]
    block_rows = []
    for label, cols in blocks:
        j = [names.index(n) for n in cols]
        p, _ = M.fit_predict(HEADLINE_MODEL, X[:, j], y_tr, Xte[:, j], C.SEED)
        m = M.evaluate(y_te, p)
        block_rows.append(dict(feature_block=label, n_features=len(cols),
                               roc_auc=m["roc_auc"], pr_auc=m["pr_auc"],
                               brier=m["brier"]))
    _write(pd.DataFrame(block_rows), "feature_blocks")

    # How performance moves as the budget of features is tightened.
    curve = []
    ranked = sorted(names, key=lambda f: -selections[1]["scores"][f])
    for k in [3, 5, 10, 15, 20, 30, len(names)]:
        cols = [names.index(f) for f in ranked[:k]]
        p, _ = M.fit_predict(HEADLINE_MODEL, X[:, cols], y_tr, Xte[:, cols], C.SEED)
        curve.append(dict(k=k, roc_auc=M.ranking_metrics(y_te, p)["roc_auc"]))
    _write(pd.DataFrame(curve), "selection_curve")

    _stage("feature selection", t0)
    scores = {s["method"]: s["scores"] for s in selections}
    return (dict(selections=selections, scores=scores, ranked=ranked),
            dict(k=SELECT_K, performance=table.to_dict(orient="records"),
                 agreement=agreement.to_dict(orient="records"),
                 curve=curve, n_features=len(names),
                 unanimous=int((agreement["n_methods"] == 3).sum()),
                 single_method=int((agreement["n_methods"] == 1).sum())))


# ===================================================== cross-validation
def _cv_experiments(d_tr, y_tr, f_tr, t0):
    """Every protocol on the full training partition, then every protocol again
    on a subsample small enough for leave-one-out, so the comparison including
    LOOCV is like for like."""
    records = []
    for name, splitter, note in V.protocols(C.SEED, include_loocv=False):
        r = V.run_protocol(name, splitter, d_tr, y_tr, C.SEED, HEADLINE_MODEL)
        r["note"] = note
        records.append(r)
        _stage(f"cv: {name}", t0)
    full = _write(V.summarize(records, y_tr), "cross_validation")

    folds = pd.concat([r["folds"] for r in records if len(r["folds"])],
                      ignore_index=True)
    _write(folds, "cv_folds")

    # ---- the small-sample comparison, where leave-one-out is affordable -----
    rng = np.random.RandomState(C.SEED)
    idx = _stratified_subsample(y_tr, C.LOOCV_N, rng)
    d_s = d_tr.iloc[idx].reset_index(drop=True)
    y_s = y_tr[idx]
    small = []
    for name, splitter, note in V.protocols(C.SEED, include_loocv=True):
        if name.startswith("Time-series"):
            continue                            # the subsample is not in date order
        r = V.run_protocol(name, splitter, d_s, y_s, C.SEED, HEADLINE_MODEL)
        r["note"] = note
        small.append(r)
        _stage(f"cv (n={len(y_s)}): {name}", t0)
    small_table = _write(V.summarize(small, y_s), "cross_validation_small")
    loo = [r for r in small if r["name"] == "LOOCV"][0]
    loo_note = V.loocv_note(y_s, loo["oof"])

    # ---- what a forward-chaining split sees that a random one cannot --------
    ts = [r for r in records if r["name"].startswith("Time-series")][0]
    drift = _write(V.temporal_drift(f_tr, ts["oof"], y_tr), "temporal_drift")

    sweep = V.stratification_sweep(
        d_tr, y_tr, C.SEED, [200, 500, 1_000, 2_500, 5_000, 20_000],
        HEADLINE_MODEL)
    _stage("stratification sweep", t0)

    return (dict(full=full.to_dict(orient="records"),
                 small=small_table.to_dict(orient="records"),
                 small_n=int(len(y_s)),
                 loocv=loo_note,
                 temporal_drift=drift.to_dict(orient="records"),
                 notes={r["name"]: r["note"] for r in records}),
            records, sweep)


def _stratified_subsample(y, n, rng):
    """A subsample that keeps the cohort's default rate, so the small-sample
    comparison is not also a change of prevalence."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = int(round(n * y.mean()))
    return np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False),
                                   rng.choice(neg, n - n_pos, replace=False)]))


# ===================================================== resampling
def _resampling_experiments(d_tr, y_tr, d_te, y_te, t0):
    """Ten treatments of the imbalance, applied inside every training fold.

    The resampled fold is built once and every model is fitted on that same
    fold, so the comparison across models is exact and the expensive part is
    paid for once.
    """
    if QUICK:
        keep = np.sort(np.random.RandomState(C.SEED + 1).choice(
            len(y_tr), QUICK_GRID_N, replace=False))
        d_tr, y_tr = d_tr.iloc[keep].reset_index(drop=True), y_tr[keep]
    cv = StratifiedKFold(n_splits=GRID_FOLDS, shuffle=True, random_state=C.SEED)
    splits = list(cv.split(d_tr, y_tr))

    oof = {(m, r): np.full(len(y_tr), np.nan) for m in GRID_MODELS for r in C.RESAMPLERS}
    shapes, fold_rows = [], []
    for k, (tr, te) in enumerate(splits):
        enc = F.Encoder(seed=C.SEED).fit(d_tr.iloc[tr], y_tr[tr])
        Xtr, Xte = enc.transform(d_tr.iloc[tr]), enc.transform(d_tr.iloc[te])
        for method in C.RESAMPLERS:
            Xr, yr, weight, secs = R.apply(method, Xtr, y_tr[tr], C.SEED)
            if k == 0:
                shapes.append(R.describe(method, y_tr[tr], yr, secs))
            for model in GRID_MODELS:
                p, _ = M.fit_predict(model, Xr, yr, Xte, C.SEED, class_weight=weight)
                oof[(model, method)][te] = p
                fold_rows.append(dict(model=model, method=method, fold=k + 1,
                                      **M.evaluate(y_tr[te], p)))
        _stage(f"resampling grid: fold {k + 1}/{GRID_FOLDS}", t0)

    shape_table = _write(R.summarize_grid(shapes), "resampling_shapes")
    _write(pd.DataFrame(fold_rows), "resampling_folds")

    rows = []
    for (model, method), p in oof.items():
        m = M.evaluate(y_tr, p)
        f = pd.DataFrame(fold_rows).query("model == @model and method == @method")
        rows.append(dict(model=model, method=method, **m,
                         roc_auc_fold_sd=float(f["roc_auc"].std(ddof=1)),
                         f1_fold_sd=float(f["f1"].std(ddof=1))))
    grid = _write(R.summarize_grid(rows, by_model=True), "resampling_grid")

    # ---- the control the whole comparison turns on --------------------------
    # Every resampler moves the decision threshold. Moving the threshold on the
    # untouched model does the same thing for nothing, so that is what each
    # method has to beat.
    base = oof[(HEADLINE_MODEL, "None")]
    tuned_t = M.best_threshold(y_tr, base, "f1")
    control = []
    for method in C.RESAMPLERS:
        p = oof[(HEADLINE_MODEL, method)]
        control.append(dict(
            method=method,
            roc_auc=M.ranking_metrics(y_tr, p)["roc_auc"],
            pr_auc=M.ranking_metrics(y_tr, p)["pr_auc"],
            ece=M.ranking_metrics(y_tr, p)["ece"],
            mean_predicted=float(np.mean(p)),
            f1_at_half=M.threshold_metrics(y_tr, p, 0.5)["f1"],
            recall_at_half=M.threshold_metrics(y_tr, p, 0.5)["recall"],
            f1_at_best=M.threshold_metrics(
                y_tr, p, M.best_threshold(y_tr, p, "f1"))["f1"]))
    control_table = _write(pd.DataFrame(control), "resampling_threshold_control")

    # ---- what SMOTE's synthetic rows actually are ---------------------------
    enc = F.Encoder(seed=C.SEED).fit(d_tr, y_tr)
    Xfull = enc.transform(d_tr)
    neigh = [R.neighbourhood_summary(Xfull, y_tr, m, C.SEED,
                                     rng=np.random.RandomState(C.SEED))
             for m in ["Random over", "SMOTE", "ADASYN", "SMOTE-Tomek", "SMOTE-ENN"]]
    neigh = pd.DataFrame([n for n in neigh if n])
    _write(neigh, "synthetic_neighbourhood")

    # ---- and the same grid scored once on the untouched hold-out ------------
    # Each method's cut-off is chosen on its own out-of-fold predictions inside
    # the training partition and then applied unchanged to the test loans.
    # Choosing it on the test set, which is what a "best F1" column usually
    # means, would let every method tune against the data it is being scored on.
    holdout = []
    Xte_h = enc.transform(d_te)
    for method in C.RESAMPLERS:
        Xr, yr, weight, _ = R.apply(method, Xfull, y_tr, C.SEED)
        p, _ = M.fit_predict(HEADLINE_MODEL, Xr, yr, Xte_h, C.SEED,
                             class_weight=weight)
        t_train = M.best_threshold(y_tr, oof[(HEADLINE_MODEL, method)], "f1")
        holdout.append(dict(method=method, predictions=p,
                            **M.evaluate(y_te, p),
                            chosen_threshold=float(t_train),
                            f1_at_chosen=M.threshold_metrics(y_te, p, t_train)["f1"],
                            f1_at_best=M.threshold_metrics(
                                y_te, p, M.best_threshold(y_te, p, "f1"))["f1"]))
    holdout_table = _write(
        R.summarize_grid([{k: v for k, v in h.items() if k != "predictions"}
                          for h in holdout]), "resampling_holdout")
    _stage("resampling grid complete", t0)

    return (dict(grid=grid.to_dict(orient="records"),
                 shapes=shape_table.to_dict(orient="records"),
                 threshold_control=control_table.to_dict(orient="records"),
                 tuned_threshold=float(tuned_t),
                 holdout=holdout_table.to_dict(orient="records"),
                 neighbourhood=neigh.to_dict(orient="records") if len(neigh) else []),
            grid, shape_table, neigh, holdout, y_tr, base)


# ===================================================== bootstrap
def _bootstrap_experiments(d_tr, y_tr, y_te, p_te, names, t0):
    cis = _write(B.all_metric_cis(y_te, p_te, C.SEED), "bootstrap_intervals")
    if QUICK:
        keep = np.sort(np.random.RandomState(C.SEED + 2).choice(
            len(y_tr), QUICK_OOB_N, replace=False))
        d_tr, y_tr = d_tr.iloc[keep].reset_index(drop=True), y_tr[keep]
    _stage("bootstrap intervals", t0)

    oob = {}
    for model in GRID_MODELS:
        r = B.out_of_bag(d_tr, y_tr, model, C.SEED)
        oob[model] = {k: v for k, v in r.items()
                      if k not in ("coefficients", "bagged_predictions", "bagged_index")}
        if model == HEADLINE_MODEL:
            stability = _write(B.coefficient_stability(r["coefficients"], names),
                               "coefficient_stability")
            sel = B.selection_stability(r["coefficients"], names, k=SELECT_K)
        _stage(f"bootstrap out-of-bag: {model}", t0)
    forest = B.forest_oob(d_tr, y_tr, C.SEED)
    imp = _write(pd.DataFrame(dict(
        feature=[F.pretty(n) for n in forest["feature_names"]],
        variable=forest["feature_names"],
        importance=forest["importances"])).sort_values(
            "importance", ascending=False).reset_index(drop=True), "forest_importance")
    _write(pd.DataFrame([{**v, "model": k} for k, v in oob.items()]
                        + [dict(model="Random forest (bagging OOB)",
                                oob_error=forest["oob_error"],
                                oob_roc_auc=forest["oob_roc_auc"],
                                oob_brier=forest["oob_brier"],
                                replicates=forest["trees"])]), "out_of_bag")
    _stage("random forest out-of-bag", t0)

    return dict(intervals=cis.to_dict(orient="records"),
                out_of_bag=oob,
                forest_oob={k: v for k, v in forest.items()
                            if k not in ("importances", "feature_names",
                                         "oob_predictions")},
                importance=imp.head(15).to_dict(orient="records"),
                stability=stability.to_dict(orient="records"),
                selection_stability={k: v for k, v in sel.items()
                                     if k != "selection_frequency"},
                selection_frequency=sel["selection_frequency"])


# ===================================================== errors
def _error_experiments(f_te, y_te, p_te, baseline, costs, y_tr, p_tr, t0):
    cost_fn = costs["mean_loss_per_default"]
    cost_fp = costs["mean_interest_per_repaid"]

    thresholds = _write(E.threshold_table(y_te, p_te, cost_fn, cost_fp), "thresholds")

    # The same six rules, chosen on the training partition's out-of-fold
    # predictions and applied to the test loans. The gap between this table and
    # the one above is the optimism a threshold picked on the evaluation data
    # carries, and it is a different quantity for every rule.
    transfer = _write(E.threshold_transfer(y_tr, p_tr, y_te, p_te, cost_fn, cost_fp),
                      "threshold_transfer")
    deciles = _write(E.decile_errors(y_te, p_te, f_te), "risk_deciles")
    profile = _write(E.error_profile(f_te, y_te, p_te, 0.5), "error_profile")
    curve = M.cost_curve(y_te, p_te, cost_fn, cost_fp)
    _write(curve, "cost_curve")
    approval = _write(E.approval_curve(y_te, p_te, f_te, costs["lgd_mean"],
                                       costs["margin_mean"]), "approval_curve")

    cost_t = M.best_threshold(y_te, p_te, "cost", cost_fn, cost_fp)
    slices = pd.concat([
        E.slice_errors(f_te, y_te, p_te, cost_t, "grade", "Grade"),
        E.slice_errors(f_te, y_te, p_te, cost_t, "purpose", "Purpose"),
        E.slice_errors(f_te, y_te, p_te, cost_t, "term_months", "Term (months)"),
        E.slice_errors(f_te, y_te, p_te, cost_t, "home_ownership", "Home ownership"),
        E.slice_errors(f_te.assign(
            income_band=pd.qcut(f_te["annual_inc"], 5,
                                labels=["lowest fifth", "second", "third",
                                        "fourth", "highest fifth"])),
            y_te, p_te, cost_t, "income_band", "Income"),
    ], ignore_index=True)
    _write(slices, "error_slices")

    reliab = pd.concat([M.reliability(y_te, p).assign(model=n)
                        for n, p in baseline.items() if n != "Prior (predict the base rate)"],
                       ignore_index=True)
    _write(reliab, "calibration")

    priced = {name: E.cost_of_errors(y_te, p_te, f_te, t, costs["lgd_mean"],
                                     costs["margin_mean"])
              for name, t in [("Default (0.5)", 0.5),
                              ("Minimum expected cost", cost_t),
                              ("Maximum F1", M.best_threshold(y_te, p_te, "f1"))]}
    _stage("error analysis", t0)

    return dict(thresholds=thresholds.to_dict(orient="records"),
                threshold_transfer=transfer.to_dict(orient="records"),
                deciles=deciles.to_dict(orient="records"),
                profile=profile.to_dict(orient="records"),
                slices=slices.to_dict(orient="records"),
                cost_curve=curve.to_dict(orient="records"),
                approval_curve=approval.to_dict(orient="records"),
                priced=priced,
                cost_optimal_threshold=float(cost_t),
                confusion_at_half=E.confusion_at(y_te, p_te, 0.5).tolist(),
                confusion_at_cost=E.confusion_at(y_te, p_te, cost_t).tolist(),
                calibration=reliab.to_dict(orient="records"))


# ===================================================== summary
def _summary(r, base_table):
    print("=" * 78)
    print(f"Credit-risk validation study — seed {r['seed']} — {r['runtime_sec']}s")
    print("=" * 78)
    c = r["cohort"]
    print(f"Cohort: {c['loans']:,} matured loans of {c['raw_loans']:,} records, "
          f"{c['default_rate']*100:.2f}% default (1:{c['imbalance_ratio']:.1f}), "
          f"{c['first_issue']} to {c['last_issue']}")
    print(f"Errors priced from the cohort: a missed default costs "
          f"${r['costs']['mean_loss_per_default']:,.0f}, a false alarm "
          f"${r['costs']['mean_interest_per_repaid']:,.0f} "
          f"({r['costs']['cost_ratio']:.2f}x)")
    print("\nHold-out performance:")
    print(base_table[["model", "roc_auc", "pr_auc", "f1", "recall", "precision",
                      "brier", "ece"]].to_string(index=False,
                                                 float_format=lambda v: f"{v:.4f}"))
    print(f"\nLeakage, in ROC-AUC: post-origination columns "
          f"+{r['leakage']['feature_inflation']:.4f}; in-sample target encoding "
          f"+{r['leakage']['target_encoding'][1]['holdout_roc_auc'] - r['leakage']['target_encoding'][0]['holdout_roc_auc']:+.4f}; "
          f"resampling before the split up to "
          f"+{max(x['inflation'] for x in r['leakage']['resampling_order']):.4f}")
    print("\nCross-validation protocols:")
    print(pd.DataFrame(r["cross_validation"]["full"]).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nResampling (stratified 5-fold, out of fold, logistic regression):")
    g = pd.DataFrame(r["resampling"]["grid"]).query("model == 'Logistic regression'")
    print(g[["method", "roc_auc", "pr_auc", "f1", "recall", "precision", "ece"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nThresholds:")
    print(pd.DataFrame(r["errors"]["thresholds"])[
        ["rule", "threshold", "recall", "precision", "f1", "cost_per_loan"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote figures/ and outputs/ (tables + results.json).")


if __name__ == "__main__":
    main()
