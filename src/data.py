"""Load, verify, and scope the LendingClub accepted-loan file.

The source file is checksummed and row-counted before it is read. Three
filters then turn 887,379 loan records into the study cohort: only loans whose
outcome is final, only loans whose full contractual term had elapsed by the
January 2016 snapshot, and only columns a lender could actually have seen when
it decided to lend. Each filter is measured on the way past, because what the
cohort excludes is as much a result as what it keeps.
"""
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw():
    """Read the source file after verifying it, and attach the parsed dates and
    the binary outcome. Nothing is dropped here; the funnel does that."""
    path = C.DATA_DIR / C.RAW_FILE
    assert _md5(path) == C.RAW_MD5, f"{C.RAW_FILE} failed its MD5 check"
    df = pd.read_csv(path, low_memory=False)
    assert df.shape == (C.RAW_ROWS, C.RAW_COLS), f"unexpected shape {df.shape}"

    df["issue_date"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["earliest_cr_date"] = pd.to_datetime(df["earliest_cr_line"],
                                            format="%b-%Y", errors="coerce")
    df["term_months"] = df["term"].str.extract(r"(\d+)").astype(int)

    known = set(C.STATUS_DEFAULT) | set(C.STATUS_REPAID) | set(C.STATUS_UNRESOLVED)
    unknown = set(df["loan_status"].unique()) - known
    assert not unknown, f"unhandled loan_status values: {unknown}"

    df["resolved"] = df["loan_status"].isin(C.STATUS_DEFAULT + C.STATUS_REPAID)
    df[C.TARGET] = df["loan_status"].isin(C.STATUS_DEFAULT).astype(int)

    # A loan has had its full chance to default only once its contractual term
    # has run. The snapshot is January 2016, so a 36-month loan issued in
    # December 2012 qualifies and one issued in January 2013 does not.
    matured_by = df["issue_date"] + pd.to_timedelta(
        df["term_months"] * C.DAYS_PER_MONTH, unit="D")
    df["matured"] = matured_by <= pd.Timestamp(C.SNAPSHOT)
    return df


def cohort_funnel(df):
    """The three filters, each priced in loans and in default rate.

    The two rows that matter are the ones a reader would otherwise have to take
    on trust: unresolved loans have no outcome to learn from, and resolved but
    unmatured loans carry a maturity bias, because a loan issued in 2015 can
    only have resolved by 2016 if it resolved unusually fast.
    """
    rows = []

    def row(stage, mask, note, rate=True):
        d = df[mask]
        rows.append(dict(
            stage=stage, loans=int(len(d)),
            # The default rate of the whole file is not a default rate. Most of
            # those loans have no outcome yet, so dividing terminal defaults by
            # every record would report 5.3% for a book that defaults at 13.4%.
            default_rate=(float(d[C.TARGET].mean()) if rate and len(d) else np.nan),
            note=note))

    row("All accepted loans", pd.Series(True, index=df.index),
        "outcome unknown for most", rate=False)
    row("Resolved outcome", df["resolved"],
        "fully paid, charged off, or defaulted")
    row("Resolved and unmatured", df["resolved"] & ~df["matured"],
        "excluded: only the fastest resolutions are visible")
    row("Analysis cohort", df["resolved"] & df["matured"],
        "full contractual term elapsed by the snapshot")
    out = pd.DataFrame(rows)
    out["default_rate"] = out["default_rate"].astype(float)
    return out


def status_breakdown(df):
    """Every loan_status value, its size, and how this study treats it."""
    treat = {}
    for s in C.STATUS_DEFAULT:
        treat[s] = "default (positive class)"
    for s in C.STATUS_REPAID:
        treat[s] = "repaid (negative class)"
    for s in C.STATUS_UNRESOLVED:
        treat[s] = "unresolved (excluded)"
    counts = df["loan_status"].value_counts()
    return pd.DataFrame(dict(
        loan_status=counts.index, loans=counts.to_numpy(),
        share=counts.to_numpy() / len(df),
        treatment=[treat[s] for s in counts.index])).reset_index(drop=True)


def get_cohort(df):
    """The analysis cohort: resolved, matured, sorted by issue date."""
    d = df[df["resolved"] & df["matured"]].copy()
    d = d.sort_values("issue_date", kind="mergesort").reset_index(drop=True)
    assert d[C.TARGET].isin([0, 1]).all()
    return d


def price_errors(cohort):
    """What each kind of error costs, computed from realized cash flows.

    Loss given default is the principal never returned, net of recoveries and of
    what collecting them cost, as a share of the amount lent. The margin is the
    interest a repaid loan actually delivered, on the same scale. Both use the
    post-origination columns the feature schema forbids, which is the right way
    round: history prices the mistake, it does not predict it.
    """
    bad = cohort[cohort[C.TARGET] == 1]
    good = cohort[cohort[C.TARGET] == 0]
    lost = (bad["loan_amnt"] - bad["total_rec_prncp"] - bad["recoveries"]
            + bad["collection_recovery_fee"])
    lgd = (lost / bad["loan_amnt"]).clip(*C.COST_CLIP)
    margin = good["total_rec_int"] / good["loan_amnt"]
    return dict(
        lgd_mean=float(lgd.mean()), lgd_median=float(lgd.median()),
        margin_mean=float(margin.mean()), margin_median=float(margin.median()),
        mean_loss_per_default=float(lost.mean()),
        mean_interest_per_repaid=float(good["total_rec_int"].mean()),
        mean_principal=float(cohort["loan_amnt"].mean()),
        # A missed default costs this many times what a false alarm costs.
        cost_ratio=float(lost.mean() / good["total_rec_int"].mean()),
        # Decline when the predicted probability exceeds this. Bayes-optimal for
        # the two costs above, and nowhere near the 0.5 every default report uses.
        cost_optimal_threshold=float(
            good["total_rec_int"].mean() / (good["total_rec_int"].mean() + lost.mean())))


def leakage_audit(cohort):
    """How separable the outcome is from each post-origination column alone.

    Reported as the share of each class on which the column takes a value that
    only one class ever takes. `recoveries` is the clearest: a repaid loan never
    generates one.
    """
    rows = []
    bad = cohort[cohort[C.TARGET] == 1]
    good = cohort[cohort[C.TARGET] == 0]
    checks = [
        ("recoveries", "> 0", lambda d: d["recoveries"] > 0),
        ("collection_recovery_fee", "> 0", lambda d: d["collection_recovery_fee"] > 0),
        ("total_rec_prncp", "< loan_amnt", lambda d: d["total_rec_prncp"] < d["loan_amnt"]),
        ("out_prncp", "> 0", lambda d: d["out_prncp"] > 0),
        ("total_rec_late_fee", "> 0", lambda d: d["total_rec_late_fee"] > 0),
        ("last_pymnt_amnt", "< installment", lambda d: d["last_pymnt_amnt"] < d["installment"]),
    ]
    for col, rule, fn in checks:
        rows.append(dict(column=col, rule=rule,
                         share_of_defaults=float(fn(bad).mean()),
                         share_of_repaid=float(fn(good).mean())))
    return pd.DataFrame(rows)


def describe_cohort(cohort):
    """Vintage table: size, default rate, and mix, by year of issue. The default
    rate is not stable across vintages, which is what makes a random split and a
    forward-chaining split disagree."""
    g = cohort.groupby(cohort["issue_date"].dt.year)
    out = pd.DataFrame(dict(
        loans=g.size(),
        default_rate=g[C.TARGET].mean(),
        mean_loan_amnt=g["loan_amnt"].mean(),
        mean_int_rate=g["int_rate"].mean(),
        share_60_month=g["term_months"].apply(lambda s: float((s == 60).mean())),
    )).reset_index().rename(columns={"issue_date": "issue_year"})
    return out


def get_split(cohort, seed):
    """One 70/30 hold-out, stratified on the outcome. Every hyperparameter,
    encoding constant, imputation constant, threshold, and resampling decision is
    made inside the training partition."""
    train, test = train_test_split(
        cohort, test_size=C.TEST_SIZE, stratify=cohort[C.TARGET], random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def get_temporal_split(cohort):
    """The split a lender actually faces: fit on what has already resolved,
    predict the loans issued afterwards. Compared against the random split to
    show what a random split conceals."""
    cut = pd.Timestamp(C.TEMPORAL_CUTOFF)
    train = cohort[cohort["issue_date"] < cut].reset_index(drop=True)
    test = cohort[cohort["issue_date"] >= cut].reset_index(drop=True)
    return train, test
