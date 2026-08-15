"""Cross-validation protocols, and what separates them.

Seven schemes are run against the same model, the same features, and the same
rows: five- and ten-fold, each plain and stratified; repeated stratified
five-fold; leave-one-out; and a forward-chaining split over the issue date. Each
returns the same record, so the comparison is of the protocol and nothing else.

Two of them cannot answer the same question as the rest and the code says so
where it happens. Leave-one-out holds out a single loan, so a fold contains one
class and has no ROC curve; only the pooled predictions can be scored, and the
fold-to-fold spread that every other protocol reports does not exist. Forward
chaining never shuffles, so its folds are different populations rather than
replicates, and its spread measures drift instead of noise.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (KFold, LeaveOneOut, RepeatedStratifiedKFold,
                                     StratifiedKFold, TimeSeriesSplit)

import config as C
import features as F
import models as M

# Protocols that hold out one row at a time, so no per-fold metric exists.
POOLED_ONLY = {"LOOCV"}
# Protocols whose folds are ordered in time rather than exchangeable.
TEMPORAL = {"Time-series split (5)"}


def protocols(seed, include_loocv=True):
    """Every scheme the study compares, as (name, splitter, note).

    The order is the order the report reads them in: the two unstratified
    schemes, the two stratified ones, the repeated version of the headline
    scheme, then the two that answer a different question.
    """
    out = [(f"K-fold ({k})", KFold(n_splits=k, shuffle=True, random_state=seed),
            "folds drawn without regard to the outcome") for k in C.CV_FOLDS]
    out += [(f"Stratified k-fold ({k})",
             StratifiedKFold(n_splits=k, shuffle=True, random_state=seed),
             "each fold carries the cohort's default rate") for k in C.CV_FOLDS]
    out.append(("Repeated stratified (5 x 5)",
                RepeatedStratifiedKFold(n_splits=5, n_repeats=C.CV_REPEATS,
                                        random_state=seed),
                "the same scheme re-drawn five times, to separate fold noise from split noise"))
    if include_loocv:
        out.append(("LOOCV", LeaveOneOut(),
                    "one loan held out at a time; no per-fold metric is defined"))
    out.append((f"Time-series split ({C.TS_SPLITS})",
                TimeSeriesSplit(n_splits=C.TS_SPLITS),
                "forward chaining over issue date; folds are vintages, not replicates"))
    return out


def run_protocol(name, splitter, design, y, seed, model_name, encode_kwargs=None,
                 resampler="None"):
    """Run one protocol end to end and return its per-fold record.

    The encoder is refitted inside every training fold. That is the expensive
    way to do it and the only correct one: an encoder fitted once on all the
    rows would carry the held-out loans' outcomes into the target encoding and
    their values into the imputation and scaling constants.
    """
    import resample as R

    encode_kwargs = dict(encode_kwargs or {})
    y = np.asarray(y).astype(int)
    t0 = time.time()

    oof = np.full(len(y), np.nan)
    fold_rows = []
    for k, (tr, te) in enumerate(splitter.split(design, y)):
        enc = F.Encoder(seed=seed, **encode_kwargs).fit(design.iloc[tr], y[tr])
        Xtr, Xte = enc.transform(design.iloc[tr]), enc.transform(design.iloc[te])
        ytr = y[tr]
        Xtr, ytr, weight, _ = R.apply(resampler, Xtr, ytr, seed)
        p, _ = M.fit_predict(model_name, Xtr, ytr, Xte, seed, class_weight=weight)
        oof[te] = p

        if name in POOLED_ONLY:
            continue
        fold_rows.append(dict(
            protocol=name, fold=k + 1, n_train=int(len(tr)), n_test=int(len(te)),
            minority_train=int(ytr.sum()), minority_test=int(y[te].sum()),
            test_default_rate=float(y[te].mean()),
            **{key: val for key, val in M.evaluate(y[te], p).items()
               if key in ("roc_auc", "pr_auc", "brier", "ece", "f1", "recall",
                          "precision", "balanced_accuracy")}))

    seconds = time.time() - t0
    folds = pd.DataFrame(fold_rows)
    # Forward chaining never holds out its first training block, so a fifth of
    # the rows are never scored. The pooled metric is computed over the rows the
    # protocol actually predicted, and how many that was is reported beside it.
    scored = ~np.isnan(oof)
    pooled = M.evaluate(y[scored], oof[scored]) if scored.sum() and y[scored].sum() else None
    return dict(name=name, folds=folds, pooled=pooled, oof=oof, seconds=seconds,
                n_scored=int(scored.sum()),
                n_fits=int(splitter.get_n_splits(design, y)))


def summarize(records, y):
    """One row per protocol: what it estimated, how much it varied, and what it
    cost. The spread column is blank for leave-one-out because the quantity does
    not exist, not because it was not computed."""
    rows = []
    for r in records:
        folds = r["folds"]
        has_folds = len(folds) > 0
        rows.append(dict(
            protocol=r["name"], fits=r["n_fits"], loans_scored=r["n_scored"],
            mean_fold_roc_auc=float(folds["roc_auc"].mean()) if has_folds else np.nan,
            sd_fold_roc_auc=float(folds["roc_auc"].std(ddof=1)) if has_folds and len(folds) > 1 else np.nan,
            min_fold_roc_auc=float(folds["roc_auc"].min()) if has_folds else np.nan,
            max_fold_roc_auc=float(folds["roc_auc"].max()) if has_folds else np.nan,
            pooled_roc_auc=float(r["pooled"]["roc_auc"]) if r["pooled"] else np.nan,
            pooled_pr_auc=float(r["pooled"]["pr_auc"]) if r["pooled"] else np.nan,
            pooled_brier=float(r["pooled"]["brier"]) if r["pooled"] else np.nan,
            min_minority_in_fold=int(folds["minority_test"].min()) if has_folds else 1,
            seconds=round(r["seconds"], 1)))
    return pd.DataFrame(rows)


def stratification_sweep(design, y, seed, sizes, model_name, k=10, draws=8):
    """When stratification stops being cosmetic.

    At 55,000 loans and a 13.4% default rate every random fold carries roughly
    the right number of defaults, so stratifying changes almost nothing. The
    question is where that stops being true. Each size is subsampled repeatedly
    and split both ways, and what is reported is how much the *fold composition*
    varies, which is the thing stratification controls.
    """
    y = np.asarray(y).astype(int)
    rng = np.random.RandomState(seed)
    rows = []
    for n in sizes:
        for draw in range(draws):
            idx = rng.choice(len(y), n, replace=False)
            ysub = y[idx]
            if ysub.sum() < k:                     # too few defaults to fold at all
                continue
            for label, splitter in [
                    ("K-fold", KFold(n_splits=k, shuffle=True, random_state=seed + draw)),
                    ("Stratified k-fold",
                     StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + draw))]:
                rates = [ysub[te].mean() for _, te in splitter.split(np.zeros(n), ysub)]
                empty = sum(1 for _, te in splitter.split(np.zeros(n), ysub)
                            if ysub[te].sum() == 0)
                rows.append(dict(n=n, draw=draw, scheme=label,
                                 fold_rate_sd=float(np.std(rates, ddof=1)),
                                 fold_rate_range=float(np.max(rates) - np.min(rates)),
                                 folds_with_no_default=int(empty)))
    df = pd.DataFrame(rows)
    return df.groupby(["n", "scheme"]).agg(
        fold_rate_sd=("fold_rate_sd", "mean"),
        fold_rate_range=("fold_rate_range", "mean"),
        folds_with_no_default=("folds_with_no_default", "mean")).reset_index()


def loocv_note(y, oof):
    """What leave-one-out can and cannot report.

    Its per-fold accuracy is a Bernoulli draw: each fold is either right or
    wrong, so the fold-to-fold standard deviation is a property of the accuracy
    itself and carries no information about the estimator's variance. Only the
    pooled predictions can be ranked, so only pooled metrics exist.
    """
    y = np.asarray(y).astype(int)
    correct = ((oof >= 0.5).astype(int) == y).astype(float)
    return dict(
        n_fits=int(len(y)),
        pooled_roc_auc=float(roc_auc_score(y, oof)),
        pooled_accuracy=float(correct.mean()),
        fold_accuracy_sd=float(correct.std(ddof=1)),
        bernoulli_sd=float(np.sqrt(correct.mean() * (1 - correct.mean()))),
        folds_with_one_class=int(len(y)))


def temporal_drift(cohort, oof, y):
    """Performance by vintage, for the forward-chaining result. A protocol that
    shuffles cannot show this, because it mixes every vintage into every fold."""
    df = pd.DataFrame(dict(year=cohort["issue_date"].dt.year.to_numpy(),
                           y=np.asarray(y).astype(int), p=np.asarray(oof)))
    df = df[df["p"].notna()]           # the first training block is never held out
    rows = []
    for year, d in df.groupby("year"):
        if d["y"].nunique() < 2:
            continue
        rows.append(dict(issue_year=int(year), loans=int(len(d)),
                         default_rate=float(d["y"].mean()),
                         roc_auc=float(roc_auc_score(d["y"], d["p"])),
                         mean_predicted=float(d["p"].mean())))
    return pd.DataFrame(rows)
