"""The model zoo and the metrics every experiment is scored on.

Three classifiers span the range a lender would consider: penalized logistic
regression, which is still the credit-scoring reference; a gradient-boosted tree
ensemble, the strongest thing on tabular data; and a random forest, which brings
a bagging out-of-bag estimate along for free. Metrics are split into two groups
on purpose. Ranking metrics score the ordering and never look at a threshold;
threshold metrics score the decisions a fixed cut-off produces. The distinction
carries most of this study's results, because resampling moves the second group
without touching the first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, log_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)

import config as C

MODELS = ["Logistic regression", "Gradient boosting", "Random forest"]


def make_model(name, seed, class_weight=None):
    """A fresh, seeded estimator. `class_weight` is the cost-sensitive
    alternative to resampling: it reweights the loss instead of changing the
    training set, so it is the control the resamplers are measured against."""
    if name == "Logistic regression":
        return LogisticRegression(C=C.LOGIT_C, max_iter=C.LOGIT_MAX_ITER,
                                  class_weight=class_weight, random_state=seed)
    if name == "Gradient boosting":
        # HistGradientBoosting has no class_weight argument in this version, so
        # the weights are passed as sample weights at fit time instead.
        return HistGradientBoostingClassifier(**{**C.GB_PARAMS, "random_state": seed})
    if name == "Random forest":
        return RandomForestClassifier(**{**C.RF_PARAMS, "random_state": seed},
                                      class_weight=class_weight)
    raise ValueError(f"unknown model {name}")


def balanced_sample_weight(y):
    """The weights `class_weight='balanced'` would apply, for the estimator that
    does not take the argument."""
    y = np.asarray(y)
    n, counts = len(y), np.bincount(y, minlength=2)
    return np.where(y == 1, n / (2.0 * counts[1]), n / (2.0 * counts[0]))


def fit_predict(name, Xtr, ytr, Xte, seed, class_weight=None):
    """Fit on the training rows and return predicted default probabilities."""
    model = make_model(name, seed, class_weight=class_weight)
    if name == "Gradient boosting" and class_weight == "balanced":
        model.fit(Xtr, ytr, sample_weight=balanced_sample_weight(ytr))
    else:
        model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1], model


# ===================================================== metrics
def expected_calibration_error(y, p, bins=C.CALIBRATION_BINS):
    """Mean absolute gap between predicted risk and observed default rate, over
    equal-count bins. Equal-count rather than equal-width because at a 13.4%
    base rate the top equal-width bins would hold almost nothing."""
    order = np.argsort(p)
    edges = np.array_split(order, bins)
    total = 0.0
    for idx in edges:
        if len(idx) == 0:
            continue
        total += len(idx) * abs(p[idx].mean() - y[idx].mean())
    return float(total / len(y))


def reliability(y, p, bins=C.CALIBRATION_BINS):
    """The reliability curve behind the calibration error."""
    order = np.argsort(p)
    rows = []
    for k, idx in enumerate(np.array_split(order, bins)):
        if len(idx) == 0:
            continue
        rows.append(dict(bin=k + 1, n=len(idx), predicted=float(p[idx].mean()),
                         observed=float(y[idx].mean())))
    return pd.DataFrame(rows)


def threshold_metrics(y, p, threshold):
    """Everything a fixed cut-off decides."""
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    return dict(
        threshold=float(threshold), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        accuracy=float((tp + tn) / len(y)), precision=float(precision),
        recall=float(recall), specificity=float(specificity), f1=float(f1),
        balanced_accuracy=float((recall + specificity) / 2),
        g_mean=float(np.sqrt(recall * specificity)), mcc=float(mcc))


def ranking_metrics(y, p):
    """Everything that depends only on the ordering, plus the two proper scoring
    rules. None of these can be moved by a decision threshold."""
    return dict(
        roc_auc=float(roc_auc_score(y, p)),
        pr_auc=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))),
        ece=expected_calibration_error(np.asarray(y), np.asarray(p)),
        mean_predicted=float(np.mean(p)))


def evaluate(y, p, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    return {**ranking_metrics(y, p), **threshold_metrics(y, p, threshold)}


# ===================================================== thresholds
def best_threshold(y, p, criterion, cost_fn=None, cost_fp=None, grid=512):
    """The cut-off that optimizes one criterion, searched on a quantile grid of
    the predicted probabilities so the grid is dense where the predictions are."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    candidates = np.unique(np.quantile(p, np.linspace(0.001, 0.999, grid)))
    best, best_score = 0.5, -np.inf
    for t in candidates:
        m = threshold_metrics(y, p, t)
        if criterion == "f1":
            score = m["f1"]
        elif criterion == "youden":
            score = m["recall"] + m["specificity"] - 1
        elif criterion == "g_mean":
            score = m["g_mean"]
        elif criterion == "cost":
            score = -(m["fn"] * cost_fn + m["fp"] * cost_fp)
        else:
            raise ValueError(criterion)
        if score > best_score:
            best, best_score = float(t), score
    return best


def expected_cost(y, p, threshold, cost_fn, cost_fp):
    """Total dollars lost at a threshold, and the same figure per loan."""
    m = threshold_metrics(np.asarray(y).astype(int), np.asarray(p, float), threshold)
    total = m["fn"] * cost_fn + m["fp"] * cost_fp
    return dict(threshold=float(threshold), total_cost=float(total),
                cost_per_loan=float(total / len(y)),
                approved=int(m["tn"] + m["fn"]), declined=int(m["tp"] + m["fp"]),
                **{k: m[k] for k in ("tp", "fp", "tn", "fn", "precision",
                                     "recall", "f1")})


def cost_curve(y, p, cost_fn, cost_fp, grid=200):
    """Cost per loan across the whole threshold range, for the figure and for
    reading off how flat the optimum is."""
    thresholds = np.unique(np.quantile(np.asarray(p, float),
                                       np.linspace(0.0, 0.999, grid)))
    return pd.DataFrame([expected_cost(y, p, t, cost_fn, cost_fp)
                         for t in thresholds])


# ===================================================== curves
def roc_points(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return fpr, tpr, float(roc_auc_score(y, p))


def pr_points(y, p):
    precision, recall, _ = precision_recall_curve(y, p)
    return recall, precision, float(average_precision_score(y, p))
