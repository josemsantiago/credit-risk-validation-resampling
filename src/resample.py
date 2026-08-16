"""The resampling grid, and the one rule that decides whether it means anything.

Ten treatments of the class imbalance: leaving it alone, reweighting the loss,
two random methods, two synthetic oversamplers, two boundary cleaners, and the
two published combinations of the last two groups. Every one of them is applied
to the training fold and to nothing else. Applying it before the split instead
is the standard mistake in this literature, and `resample_before_split` exists so
the study can measure what that mistake is worth rather than describe it.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import (EditedNearestNeighbours, RandomUnderSampler,
                                     TomekLinks)

import config as C

# What each method actually does, for the table in the report.
DESCRIPTIONS = {
    "None": "training fold left at its observed 13.4% default rate",
    "Class weight": "loss reweighted inversely to class frequency; no rows added or removed",
    "Random over": "minority rows duplicated at random until the classes are equal",
    "Random under": "majority rows discarded at random until the classes are equal",
    "SMOTE": "synthetic minority rows interpolated between a minority row and one of its k nearest minority neighbours",
    "ADASYN": "as SMOTE, but generating more where the local neighbourhood is majority-dominated",
    "Tomek links": "majority row of each mutually-nearest opposite-class pair removed",
    "ENN": "row removed when its k nearest neighbours disagree with its label",
    "SMOTE-Tomek": "SMOTE, then Tomek links applied to the enlarged set",
    "SMOTE-ENN": "SMOTE, then edited nearest neighbours applied to the enlarged set",
}

# Whether the method changes the training rows at all. "None" and "Class weight"
# do not, which is why they are the two references everything else is read
# against.
RESHAPES_DATA = {name: name not in ("None", "Class weight") for name in C.RESAMPLERS}


# Neighbour queries are independent and merged by row index, so parallelism
# here cannot change the answer, unlike a threaded float reduction. BLAS is
# pinned to one thread in run_analysis.py; the cores go here instead.
N_JOBS = -1


def make_resampler(name, seed):
    """The imbalanced-learn estimator behind a method name, or None when the
    method leaves the training rows alone."""
    if name in ("None", "Class weight"):
        return None
    if name == "Random over":
        return RandomOverSampler(random_state=seed)
    if name == "Random under":
        return RandomUnderSampler(random_state=seed)
    if name == "SMOTE":
        return SMOTE(random_state=seed, k_neighbors=C.SMOTE_K, n_jobs=N_JOBS)
    if name == "ADASYN":
        return ADASYN(random_state=seed, n_neighbors=C.SMOTE_K, n_jobs=N_JOBS)
    if name == "Tomek links":
        return TomekLinks(sampling_strategy="majority", n_jobs=N_JOBS)
    if name == "ENN":
        return EditedNearestNeighbours(n_neighbors=C.ENN_K, n_jobs=N_JOBS)
    if name == "SMOTE-Tomek":
        return SMOTETomek(random_state=seed,
                          smote=SMOTE(random_state=seed, k_neighbors=C.SMOTE_K, n_jobs=N_JOBS),
                          tomek=TomekLinks(sampling_strategy="all", n_jobs=N_JOBS))
    if name == "SMOTE-ENN":
        return SMOTEENN(random_state=seed,
                        smote=SMOTE(random_state=seed, k_neighbors=C.SMOTE_K, n_jobs=N_JOBS),
                        enn=EditedNearestNeighbours(n_neighbors=C.ENN_K,
                                                    sampling_strategy="all", n_jobs=N_JOBS))
    raise ValueError(f"unknown resampler {name}")


def apply(name, X, y, seed):
    """Resample one training fold. Returns the new matrix, the new labels, the
    class weight the model should carry, and how long the resampling took."""
    if name == "Class weight":
        return X, y, "balanced", 0.0
    if name == "None":
        return X, y, None, 0.0
    sampler = make_resampler(name, seed)
    t0 = time.time()
    Xr, yr = sampler.fit_resample(X, y)
    return Xr, yr, None, time.time() - t0


def describe(name, y_before, y_after, seconds):
    """One row of the resampling-effect table: what the method did to the
    training fold, in rows rather than in adjectives."""
    b, a = np.bincount(y_before, minlength=2), np.bincount(y_after, minlength=2)
    return dict(
        method=name, description=DESCRIPTIONS[name],
        rows_before=int(b.sum()), rows_after=int(a.sum()),
        majority_before=int(b[0]), majority_after=int(a[0]),
        minority_before=int(b[1]), minority_after=int(a[1]),
        minority_share_before=float(b[1] / b.sum()),
        minority_share_after=float(a[1] / a.sum()),
        rows_added=int(max(0, a.sum() - b.sum())),
        rows_removed=int(max(0, b.sum() - a.sum())),
        synthetic_share=float(max(0, a[1] - b[1]) / a.sum()),
        seconds=float(seconds))


def resample_before_split(X, y, name, seed, test_size, rng):
    """The wrong order, run deliberately.

    Resampling the whole cohort and splitting afterwards puts near-copies of test
    rows into the training set: a duplicated minority row can land on both sides,
    and a synthetic SMOTE point is a blend of real rows that may now be test
    rows. The model is then partly memorizing its own test set. This returns the
    indices that mistake produces so the inflation can be measured.
    """
    Xr, yr, _, _ = apply(name, X, y, seed)
    n = len(yr)
    idx = rng.permutation(n)
    cut = int(round(n * (1 - test_size)))
    return Xr[idx[:cut]], yr[idx[:cut]], Xr[idx[cut:]], yr[idx[cut:]]


def neighbourhood_summary(X, y, name, seed, sample=4000, rng=None):
    """How far a synthetic minority point sits from the real ones it came from.

    A synthetic row that lands on top of an existing minority row adds nothing an
    oversampled duplicate would not; one that lands far away is an assertion about
    a region of feature space where no borrower was observed. The median distance
    to the nearest real minority row, against the median distance between real
    minority rows, says which of the two a method is doing.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = rng or np.random.RandomState(seed)
    Xr, yr, _, _ = apply(name, X, y, seed)
    real_min = X[y == 1]
    if len(Xr) == len(X) and np.array_equal(yr, y):
        return None

    new_min = Xr[yr == 1]
    if len(new_min) <= len(real_min):
        return None                                    # nothing synthetic was added

    take = min(sample, len(new_min))
    probe = new_min[rng.choice(len(new_min), take, replace=False)]
    nn = NearestNeighbors(n_neighbors=1).fit(real_min)
    d_syn = nn.kneighbors(probe, return_distance=True)[0][:, 0]

    base_take = min(sample, len(real_min))
    base = real_min[rng.choice(len(real_min), base_take, replace=False)]
    nn2 = NearestNeighbors(n_neighbors=2).fit(real_min)
    d_real = nn2.kneighbors(base, return_distance=True)[0][:, 1]

    # "Duplicate" is judged against the scale of the data rather than against
    # zero. Two identical 48-dimensional rows come back from a brute-force
    # Euclidean search at around 1e-7 rather than exactly 0, because the
    # implementation expands the squared distance, so an absolute tolerance would
    # call a duplicated row novel.
    typical = float(np.median(d_real))
    tolerance = 1e-4 * typical if typical > 0 else 1e-9
    return dict(method=name,
                median_distance_to_real=float(np.median(d_syn)),
                median_distance_between_real=typical,
                share_duplicate=float(np.mean(d_syn <= tolerance)),
                ratio=float(np.median(d_syn) / typical) if typical > 0 else np.nan)


def summarize_grid(rows, by_model=False):
    """Order the resampling comparison the way the report reads it: the two
    references first, then the random methods, the synthetic ones, the cleaners,
    and the combinations. Alphabetical order would scatter them."""
    df = pd.DataFrame(rows)
    order = {name: i for i, name in enumerate(C.RESAMPLERS)}
    keys = (["model", "_o"] if by_model and "model" in df.columns else ["_o"])
    return df.assign(_o=df["method"].map(order)).sort_values(
        keys, kind="mergesort").drop(columns="_o").reset_index(drop=True)
