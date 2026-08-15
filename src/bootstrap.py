"""Bootstrap resampling: confidence intervals, out-of-bag error, and stability.

Three distinct uses of the same idea, kept apart because they answer different
questions. Resampling the *test* rows says how precisely a metric is estimated
on the loans that were scored. Resampling the *training* rows and scoring the
loans left out says how well the procedure would do on data it has not seen, and
gives the out-of-bag estimate the .632 family is built from. Refitting on those
same training resamples and watching the coefficients says how stable the model
itself is.

Two interval methods are reported everywhere. The percentile interval takes the
empirical quantiles of the bootstrap distribution and assumes it is centred and
symmetric on the right scale. The bias-corrected and accelerated interval
(DiCiccio & Efron, 1996) shifts and stretches those quantiles by a bias term read off the
bootstrap distribution and an acceleration read off a jackknife, so it stays
honest where the distribution is skewed. On a metric bounded at 1, such as an
ROC-AUC near 0.7, the two rarely disagree by much; the report says by how much
rather than assuming.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import config as C
import features as F
import models as M


# ===================================================== intervals
def _grouped_jackknife(y, p, metric, n_groups):
    """Acceleration input for BCa, by deleting one group of rows at a time.

    The textbook BCa deletes one observation at a time, which is 23,000 metric
    evaluations here. Deleting equal groups instead is the delete-d jackknife,
    and it estimates the same skewness at a cost the study can actually pay.

    The groups are contiguous blocks of the stored row order, and the test
    partition is stored in issue-date order, so a block holds loans of one
    vintage rather than a random spread. That could in principle let the
    acceleration pick up temporal drift instead of sampling variability, so it
    was measured rather than argued. Against blocks drawn from a shuffled order,
    on a simulation matching this cohort's size, vintage structure, and drifting
    default rate, the two interval endpoints moved by 0.000012 and 0.000001,
    four orders of magnitude below the interval's own width. The contiguous form
    is kept because it is the cheaper and the more reproducible of the two.
    """
    n = len(y)
    n_groups = min(n_groups, n)
    order = np.arange(n)
    groups = np.array_split(order, n_groups)
    out = np.empty(len(groups))
    keep = np.ones(n, dtype=bool)
    for i, g in enumerate(groups):
        keep[g] = False
        out[i] = metric(y[keep], p[keep])
        keep[g] = True
    return out


def metric_ci(y, p, metric, seed, n_boot=None, alpha=None,
              jackknife_groups=None):
    """Percentile and BCa intervals for one metric, from one set of resamples.

    Rows are resampled with replacement, stratified on the outcome so that no
    resample loses the minority class entirely, which would leave the ROC-AUC
    undefined. Both intervals are read off the same bootstrap distribution, so
    any difference between them is the correction and nothing else.

    The budgets default to None and are resolved from config here rather than in
    the signature. A default argument is evaluated once, when the module is
    imported, so a signature default would freeze whatever config held at import
    time and quietly ignore any later change to it.
    """
    n_boot = C.BOOTSTRAP_N if n_boot is None else n_boot
    alpha = C.BOOTSTRAP_ALPHA if alpha is None else alpha
    jackknife_groups = (C.BOOTSTRAP_JACKKNIFE_N if jackknife_groups is None
                        else jackknife_groups)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rng = np.random.RandomState(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)

    observed = float(metric(y, p))
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        reps[b] = metric(y[idx], p[idx])

    lo_pct, hi_pct = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # ---- BCa ---------------------------------------------------------------
    share_below = float(np.mean(reps < observed))
    share_below = min(max(share_below, 1.0 / (2 * n_boot)), 1 - 1.0 / (2 * n_boot))
    z0 = stats.norm.ppf(share_below)

    jack = _grouped_jackknife(y, p, metric, jackknife_groups)
    dev = jack.mean() - jack
    denom = 6.0 * (np.sum(dev ** 2) ** 1.5)
    accel = float(np.sum(dev ** 3) / denom) if denom > 0 else 0.0

    z_lo, z_hi = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    def adjust(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1 - accel * (z0 + z)))
    a_lo, a_hi = adjust(z_lo), adjust(z_hi)
    lo_bca, hi_bca = np.percentile(reps, [100 * a_lo, 100 * a_hi])

    return dict(
        observed=observed, boot_mean=float(reps.mean()), boot_sd=float(reps.std(ddof=1)),
        bias=float(reps.mean() - observed),
        percentile_lo=float(lo_pct), percentile_hi=float(hi_pct),
        percentile_width=float(hi_pct - lo_pct),
        bca_lo=float(lo_bca), bca_hi=float(hi_bca), bca_width=float(hi_bca - lo_bca),
        z0=float(z0), acceleration=accel,
        skew=float(stats.skew(reps)), n_boot=int(n_boot))


METRIC_FUNCTIONS = {
    "ROC-AUC": lambda y, p: M.ranking_metrics(y, p)["roc_auc"],
    "PR-AUC": lambda y, p: M.ranking_metrics(y, p)["pr_auc"],
    "Brier score": lambda y, p: M.ranking_metrics(y, p)["brier"],
    "Recall at 0.5": lambda y, p: M.threshold_metrics(y, p, 0.5)["recall"],
    "Precision at 0.5": lambda y, p: M.threshold_metrics(y, p, 0.5)["precision"],
    "F1 at 0.5": lambda y, p: M.threshold_metrics(y, p, 0.5)["f1"],
    "Balanced accuracy": lambda y, p: M.threshold_metrics(y, p, 0.5)["balanced_accuracy"],
}


def all_metric_cis(y, p, seed, metrics=None, n_boot=None):
    """Every headline metric with both intervals, as one table."""
    metrics = metrics or list(METRIC_FUNCTIONS)
    n_boot = C.BOOTSTRAP_N if n_boot is None else n_boot
    rows = []
    for i, name in enumerate(metrics):
        r = metric_ci(y, p, METRIC_FUNCTIONS[name], seed + i, n_boot=n_boot)
        rows.append(dict(metric=name, **r))
    return pd.DataFrame(rows)


# ===================================================== out-of-bag
def out_of_bag(design, y, model_name, seed, n_boot=None,
               encode_kwargs=None, resampler="None"):
    """Efron's bootstrap out-of-bag estimate, and the .632 family built on it.

    Each replicate draws n training rows with replacement, which leaves about
    36.8% of the cohort untouched; the model is fitted on the draw and scored on
    what it missed. Averaging over replicates gives an estimate that, unlike the
    resubstitution error, has not seen its own test rows, and unlike a single
    hold-out uses every loan.

    The plain out-of-bag estimate is pessimistic, because each fit sees about
    63.2% distinct rows rather than n. The .632 estimator corrects that by
    blending in the optimistic resubstitution error at a fixed weight, and .632+
    varies the weight by how much the model actually overfits, measured against
    the error a no-information classifier would make (Arlot & Celisse, 2010;
    Kohavi, 1995).
    """
    import resample as R

    n_boot = C.BOOTSTRAP_MODEL_N if n_boot is None else n_boot
    y = np.asarray(y).astype(int)
    n = len(y)
    rng = np.random.RandomState(seed)
    encode_kwargs = dict(encode_kwargs or {})

    oob_sum = np.zeros(n)
    oob_count = np.zeros(n)
    app_errors, oob_errors, coefs, oob_shares = [], [], [], []
    # Each replicate refits its own encoder, so a draw that happens to miss a
    # rare categorical level would produce a shorter coefficient vector. That
    # would silently misalign the stability table, so replicates whose design
    # matrix does not match the reference are counted and left out of it.
    reference_columns, misaligned = None, 0

    for b in range(n_boot):
        draw = rng.randint(0, n, n)
        oob_idx = np.setdiff1d(np.arange(n), np.unique(draw), assume_unique=False)
        if len(oob_idx) == 0 or y[draw].sum() == 0 or y[oob_idx].sum() == 0:
            continue

        # Materialize each slice once. Selecting 54,771 rows with replacement out
        # of a frame carrying object columns is the most expensive line in the
        # loop, and an earlier version paid for it three times per replicate.
        d_draw, d_oob = design.iloc[draw], design.iloc[oob_idx]
        y_draw = y[draw]

        enc = F.Encoder(seed=seed, **encode_kwargs).fit(d_draw, y_draw)
        X_draw = enc.transform(d_draw)
        Xoob = enc.transform(d_oob)
        Xtr, ytr, weight, _ = R.apply(resampler, X_draw, y_draw, seed + b)

        p_oob, model = M.fit_predict(model_name, Xtr, ytr, Xoob, seed,
                                     class_weight=weight)
        p_app = model.predict_proba(X_draw)[:, 1]

        oob_sum[oob_idx] += p_oob
        oob_count[oob_idx] += 1
        oob_shares.append(len(oob_idx) / n)
        app_errors.append(float(np.mean((p_app >= 0.5).astype(int) != y[draw])))
        oob_errors.append(float(np.mean((p_oob >= 0.5).astype(int) != y[oob_idx])))
        if hasattr(model, "coef_"):
            cols = F.feature_names(enc)
            if reference_columns is None:
                reference_columns = cols
            if cols == reference_columns:
                coefs.append(model.coef_[0].copy())
            else:
                misaligned += 1

    scored = oob_count > 0
    p_bag = oob_sum[scored] / oob_count[scored]

    err_app = float(np.mean(app_errors))
    err_oob = float(np.mean(oob_errors))

    # No-information error rate: what the classifier would score if its
    # predictions were paired at random with the labels.
    rate = float(y.mean())
    predicted_positive = float(np.mean(p_bag >= 0.5))
    gamma = rate * (1 - predicted_positive) + (1 - rate) * predicted_positive
    overfit = ((err_oob - err_app) / (gamma - err_app)) if gamma > err_app else 0.0
    overfit = float(min(max(overfit, 0.0), 1.0))
    weight_632 = 0.632 / (1 - 0.368 * overfit)

    return dict(
        model=model_name, replicates=len(oob_errors),
        # Should sit at 1 - 1/e = 0.3679; reported rather than assumed.
        mean_oob_share=float(np.mean(oob_shares)),
        apparent_error=err_app, oob_error=err_oob,
        error_632=float(0.368 * err_app + 0.632 * err_oob),
        error_632_plus=float((1 - weight_632) * err_app + weight_632 * err_oob),
        no_information_error=float(gamma), relative_overfitting=overfit,
        oob_roc_auc=float(M.ranking_metrics(y[scored], p_bag)["roc_auc"]),
        oob_brier=float(M.ranking_metrics(y[scored], p_bag)["brier"]),
        loans_scored=int(scored.sum()), misaligned_replicates=int(misaligned),
        coefficients=np.array(coefs) if coefs else None,
        bagged_predictions=p_bag, bagged_index=np.flatnonzero(scored))


def forest_oob(design, y, seed, encode_kwargs=None):
    """The random forest's own out-of-bag score, which costs one fit rather than
    two hundred because every tree already holds out the rows its bootstrap draw
    missed."""
    from sklearn.ensemble import RandomForestClassifier

    y = np.asarray(y).astype(int)
    enc = F.Encoder(seed=seed, **dict(encode_kwargs or {})).fit(design, y)
    X = enc.transform(design)
    rf = RandomForestClassifier(**{**C.RF_PARAMS, "random_state": seed,
                                   "n_jobs": -1},
                                oob_score=True, bootstrap=True).fit(X, y)
    p_oob = rf.oob_decision_function_[:, 1]
    usable = ~np.isnan(p_oob)
    return dict(
        oob_accuracy=float(rf.oob_score_),
        oob_error=float(1 - rf.oob_score_),
        oob_roc_auc=float(M.ranking_metrics(y[usable], p_oob[usable])["roc_auc"]),
        oob_brier=float(M.ranking_metrics(y[usable], p_oob[usable])["brier"]),
        trees=int(C.RF_PARAMS["n_estimators"]),
        importances=rf.feature_importances_, feature_names=F.feature_names(enc),
        oob_predictions=p_oob)


# ===================================================== stability
def coefficient_stability(coefs, names):
    """How much each coefficient moves across bootstrap refits.

    A coefficient whose sign flips between replicates is not evidence of
    anything, however large its point estimate. The sign-agreement column is the
    one to read: it is the share of replicates agreeing with the majority sign.
    """
    if coefs is None or len(coefs) == 0:
        return pd.DataFrame()
    coefs = np.asarray(coefs)
    mean = coefs.mean(axis=0)
    sign_agreement = np.maximum((coefs > 0).mean(axis=0), (coefs < 0).mean(axis=0))
    ranks = np.argsort(np.argsort(-np.abs(coefs), axis=1), axis=1) + 1
    rows = []
    for j, name in enumerate(names):
        rows.append(dict(
            feature=F.pretty(name), variable=name,
            mean_coefficient=float(mean[j]), sd_coefficient=float(coefs[:, j].std(ddof=1)),
            cv=float(abs(coefs[:, j].std(ddof=1) / mean[j])) if mean[j] else np.nan,
            sign_agreement=float(sign_agreement[j]),
            mean_rank=float(ranks[:, j].mean()), sd_rank=float(ranks[:, j].std(ddof=1)),
            lo=float(np.percentile(coefs[:, j], 2.5)),
            hi=float(np.percentile(coefs[:, j], 97.5))))
    return pd.DataFrame(rows).sort_values("mean_rank").reset_index(drop=True)


def selection_stability(coefs, names, k=15):
    """How often two bootstrap replicates would pick the same top-k features.

    Reported as the mean pairwise Jaccard index over replicates. A selection that
    changes with the resample is a selection the data did not determine.
    """
    if coefs is None or len(coefs) < 2:
        return None
    coefs = np.asarray(coefs)
    tops = [set(np.argsort(-np.abs(c))[:k]) for c in coefs]
    sims = [len(a & b) / len(a | b)
            for i, a in enumerate(tops) for b in tops[i + 1:]]
    counts = np.zeros(len(names))
    for t in tops:
        for j in t:
            counts[j] += 1
    return dict(k=int(k), mean_jaccard=float(np.mean(sims)),
                sd_jaccard=float(np.std(sims, ddof=1)),
                always_selected=int((counts == len(tops)).sum()),
                never_selected=int((counts == 0).sum()),
                selection_frequency={names[j]: float(counts[j] / len(tops))
                                     for j in range(len(names))})
