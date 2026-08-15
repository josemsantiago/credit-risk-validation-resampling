"""Error analysis: which mistakes the model makes, where it makes them, and what
each one costs.

A confusion matrix at 0.5 is where most reports stop. It is the least
informative view available on an imbalanced problem, because at a 13.4% base
rate a model that declines almost nothing scores well on accuracy and catches
almost no defaults. The three views here go further: errors sliced by the
borrower characteristics a credit officer would ask about, errors priced by what
they actually cost, and the threshold that minimizes that cost rather than the
threshold nobody chose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import models as M


def confusion_at(y, p, threshold):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for a in (0, 1):
        for b in (0, 1):
            cm[a, b] = int(np.sum((y == a) & (pred == b)))
    return cm


def slice_errors(frame, y, p, threshold, by, label=None, min_n=100):
    """False-negative and false-positive rates within each level of one column.

    The false-negative rate is the share of real defaults the model approved,
    which is the error the lender pays for. Levels smaller than `min_n` are
    dropped, because a rate over thirty loans is noise.
    """
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    d = pd.DataFrame({"g": frame[by].to_numpy(), "y": y, "pred": pred,
                      "p": np.asarray(p)})
    rows = []
    for level, g in d.groupby("g", sort=True):
        if len(g) < min_n:
            continue
        pos, neg = g[g["y"] == 1], g[g["y"] == 0]
        rows.append(dict(
            variable=label or by, level=str(level), loans=int(len(g)),
            default_rate=float(g["y"].mean()),
            mean_predicted=float(g["p"].mean()),
            false_negative_rate=float((pos["pred"] == 0).mean()) if len(pos) else np.nan,
            false_positive_rate=float((neg["pred"] == 1).mean()) if len(neg) else np.nan,
            recall=float((pos["pred"] == 1).mean()) if len(pos) else np.nan,
            precision=float(g.loc[g["pred"] == 1, "y"].mean())
            if (g["pred"] == 1).any() else np.nan,
            calibration_gap=float(g["p"].mean() - g["y"].mean())))
    return pd.DataFrame(rows)


def decile_errors(y, p, frame=None, amount_col="loan_amnt"):
    """The model's own risk deciles, which is the slice a lender actually uses.

    Reading down this table says whether the score orders borrowers: the
    observed default rate should climb monotonically, and the predicted column
    beside it should track it rather than merely rank with it.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    rows = []
    for k, idx in enumerate(np.array_split(order, 10)):
        row = dict(decile=k + 1, loans=int(len(idx)),
                   mean_predicted=float(p[idx].mean()),
                   observed_default_rate=float(y[idx].mean()),
                   defaults=int(y[idx].sum()))
        if frame is not None:
            row["mean_loan_amount"] = float(frame[amount_col].to_numpy()[idx].mean())
            row["principal_at_risk"] = float(
                frame[amount_col].to_numpy()[idx][y[idx] == 1].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_table(y, p, cost_fn, cost_fp):
    """The four thresholds a report might choose, and what each one decides.

    They are not close together, and the choice between them is a business
    decision rather than a statistical one. Reporting only the 0.5 row, as an
    accuracy table does, hides that a choice was made at all.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rows = []
    named = [
        ("Default (0.5)", 0.5, "the cut-off nobody chose"),
        ("Prevalence (base rate)", float(y.mean()), "decline above the cohort's own default rate"),
        ("Maximum F1", M.best_threshold(y, p, "f1"), "balances precision and recall"),
        ("Youden J", M.best_threshold(y, p, "youden"), "maximizes sensitivity plus specificity"),
        ("Maximum G-mean", M.best_threshold(y, p, "g_mean"), "geometric mean of the two rates"),
        ("Minimum expected cost", M.best_threshold(y, p, "cost", cost_fn, cost_fp),
         "minimizes dollars lost, given what each error costs"),
    ]
    for name, t, note in named:
        c = M.expected_cost(y, p, t, cost_fn, cost_fp)
        m = M.threshold_metrics(y, p, t)
        rows.append(dict(rule=name, threshold=t, note=note,
                         approved=c["approved"], declined=c["declined"],
                         **{k: m[k] for k in ("tp", "fp", "tn", "fn", "precision",
                                              "recall", "specificity", "f1",
                                              "balanced_accuracy", "accuracy")},
                         cost_per_loan=c["cost_per_loan"],
                         total_cost=c["total_cost"]))
    return pd.DataFrame(rows)


def threshold_transfer(y_train, p_train, y_test, p_test, cost_fn, cost_fp):
    """Each rule's cut-off, chosen out of fold on the training loans and then
    applied unchanged to the test loans.

    A threshold is a fitted parameter. Choosing it on the same loans it is scored
    on is the same mistake as choosing a hyperparameter there, and it inflates
    exactly the metrics it was chosen to maximize. The optimism column is the
    difference between the score the rule reaches when tuned on the test set and
    the score it reaches when it has to transfer.
    """
    y_train, y_test = np.asarray(y_train).astype(int), np.asarray(y_test).astype(int)
    p_train, p_test = np.asarray(p_train, float), np.asarray(p_test, float)
    rows = []
    rules = [("Default (0.5)", lambda: 0.5),
             ("Prevalence (base rate)", lambda: float(y_train.mean())),
             ("Maximum F1", lambda: M.best_threshold(y_train, p_train, "f1")),
             ("Youden J", lambda: M.best_threshold(y_train, p_train, "youden")),
             ("Maximum G-mean", lambda: M.best_threshold(y_train, p_train, "g_mean")),
             ("Minimum expected cost",
              lambda: M.best_threshold(y_train, p_train, "cost", cost_fn, cost_fp))]
    for name, pick in rules:
        t = pick()
        transferred = M.threshold_metrics(y_test, p_test, t)
        tuned_t = (0.5 if name.startswith("Default") else
                   float(y_test.mean()) if name.startswith("Prevalence") else
                   M.best_threshold(y_test, p_test,
                                    {"Maximum F1": "f1", "Youden J": "youden",
                                     "Maximum G-mean": "g_mean",
                                     "Minimum expected cost": "cost"}[name],
                                    cost_fn, cost_fp))
        tuned = M.threshold_metrics(y_test, p_test, tuned_t)
        cost_here = M.expected_cost(y_test, p_test, t, cost_fn, cost_fp)
        rows.append(dict(
            rule=name, threshold_from_training=float(t),
            threshold_from_test=float(tuned_t),
            f1_transferred=transferred["f1"], f1_tuned_on_test=tuned["f1"],
            optimism=tuned["f1"] - transferred["f1"],
            recall=transferred["recall"], precision=transferred["precision"],
            cost_per_loan=cost_here["cost_per_loan"]))
    return pd.DataFrame(rows)


def cost_of_errors(y, p, frame, threshold, lgd, margin):
    """Errors priced in dollars of this cohort's own principal.

    A missed default loses `lgd` of the amount lent; a false alarm forgoes
    `margin` of it. Using each loan's own principal rather than the cohort mean
    matters, because the model's errors are not evenly distributed across loan
    sizes.
    """
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    amount = frame["loan_amnt"].to_numpy(dtype=float)
    fn = (y == 1) & (pred == 0)
    fp = (y == 0) & (pred == 1)
    tn = (y == 0) & (pred == 0)
    return dict(
        threshold=float(threshold),
        false_negatives=int(fn.sum()), false_positives=int(fp.sum()),
        loss_from_missed_defaults=float((amount[fn] * lgd).sum()),
        forgone_margin_from_false_alarms=float((amount[fp] * margin).sum()),
        margin_earned_on_approved_good=float((amount[tn] * margin).sum()),
        total_cost=float((amount[fn] * lgd).sum() + (amount[fp] * margin).sum()),
        cost_per_loan=float(((amount[fn] * lgd).sum() + (amount[fp] * margin).sum())
                            / len(y)),
        mean_missed_default_size=float(amount[fn].mean()) if fn.any() else np.nan,
        mean_false_alarm_size=float(amount[fp].mean()) if fp.any() else np.nan)


def error_profile(frame, y, p, threshold):
    """What separates the loans the model gets wrong from the ones it gets right.

    For each numeric feature, the mean among false negatives against the mean
    among true positives: the gap says what a missed default looked like. A
    missed default that looks like a repaid loan on every feature is a limit of
    the data; one that looks distinctive is a limit of the model.
    """
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    fn = (y == 1) & (pred == 0)
    tp = (y == 1) & (pred == 1)
    rows = []
    for col in ["int_rate", "loan_amnt", "annual_inc", "dti", "revol_util",
                "inq_last_6mths", "term_months"]:
        v = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        a, b = v[fn], v[tp]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else np.nan
        rows.append(dict(
            feature=C.LABELS.get(col, col), variable=col,
            missed_defaults=float(a.mean()), caught_defaults=float(b.mean()),
            difference=float(a.mean() - b.mean()),
            standardized_difference=float((a.mean() - b.mean()) / pooled)
            if pooled and pooled > 0 else np.nan))
    return pd.DataFrame(rows).sort_values(
        "standardized_difference", key=np.abs, ascending=False).reset_index(drop=True)


def approval_curve(y, p, frame, lgd, margin, grid=60):
    """Portfolio view: for every approval rate, what the approved book earns.

    A lender does not choose a threshold, it chooses how much of the market to
    approve. This maps one to the other, and the maximum of the net column is
    the operating point the study recommends.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    amount = frame["loan_amnt"].to_numpy(dtype=float)
    order = np.argsort(p)
    rows = []
    for share in np.linspace(0.05, 1.0, grid):
        take = order[:int(round(share * len(order)))]
        if len(take) == 0:
            continue
        good, bad = take[y[take] == 0], take[y[take] == 1]
        earned = (amount[good] * margin).sum()
        lost = (amount[bad] * lgd).sum()
        rows.append(dict(
            approval_rate=float(len(take) / len(order)),
            threshold=float(p[take].max()),
            approved=int(len(take)),
            approved_default_rate=float(y[take].mean()),
            principal_lent=float(amount[take].sum()),
            interest_earned=float(earned), losses=float(lost),
            net=float(earned - lost),
            return_on_principal=float((earned - lost) / amount[take].sum())))
    return pd.DataFrame(rows)
