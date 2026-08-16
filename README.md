# Credit Default Prediction: Validation Protocols, Resampling, and Error Analysis

A reproducible study of **how a credit-risk model should be validated**, run on
78,245 LendingClub loans whose outcomes are actually final. Seven
cross-validation protocols, ten treatments of the class imbalance, bootstrap
confidence intervals, and an error analysis priced in dollars.

## What this does

Builds a default classifier on LendingClub's published loan book, then spends
almost all of its effort on the procedure rather than on the classifier. It
starts by showing that most of what the file appears to predict is the outcome
written down twice.

**The cohort comes first.** `loan_status` mixes finished loans with in-flight
ones, and "Current" alone is 601,779 of the 887,379 records; treating it as a
non-default labels a loan that has not yet had the chance to default. Keeping
only terminal statuses leaves 256,939. Then a loan that resolved before its term
elapsed only resolved because it resolved fast, which is not independent of the
outcome, so requiring the full contractual term leaves **78,245 loans at a 13.40%
default rate**. The loans that filter removes default at 20.56%.

**Experiments:** three leakage routes priced in ROC-AUC · four imputation
strategies and three treatments of informative missingness · three scalers, with
and without a distance-based resampler in front · RFE, mutual information, and
LASSO asked for fifteen features each · seven cross-validation protocols on
identical rows · ten treatments of the class imbalance applied inside every
training fold · percentile and BCa bootstrap intervals, out-of-bag error, and the
.632 family · errors sliced by borrower and priced in dollars of this cohort's own
principal.

## Quick start

```bash
pip install -r requirements.txt      # scikit-learn imbalanced-learn numpy pandas scipy matplotlib
cd src && python3 run_analysis.py    # ~41 min; regenerates every figure, table, results.json
QUICK=1 python3 run_analysis.py      # ~2 min smoke run that exercises every code path
python3 render_figures.py            # seconds; redraws the figures from cached inputs
python3 render_figures.py --verify    #   and checks they match what the pipeline drew
```

Editing a title, a colour, or a panel layout does not need the full run. The
pipeline caches the arguments the figure code reads into `outputs/plot_inputs.pkl`,
and `render_figures.py` replays them in a couple of seconds; `--verify` MD5s every
figure before and after so a shortcut that stops matching the pipeline says so.

The loan file is a 441 MB public download and is **not** tracked. Fetch it from
the Kaggle dataset named in `data/SOURCE.txt` into `data/loan.csv`; the loader
verifies its MD5 and its exact 887,379 x 74 shape before reading, so a truncated
or substituted file fails loudly rather than trains silently.

## Layout

```
├── src/                     # pipeline (deterministic under master seed)
│   ├── config.py            # seed, checksum, cohort definition, feature schema, excluded columns,
│   │                        #   CV protocols, resampling grid, bootstrap budgets
│   ├── data.py              # verified load, the two cohort filters, leakage audit, error pricing
│   ├── features.py          # derived features, out-of-fold target encoding, imputation, scaling,
│   │                        #   RFE / mutual information / LASSO selection
│   ├── models.py            # model zoo, ranking and threshold metrics kept apart, cost curves
│   ├── validate.py          # seven CV protocols, the stratification sweep, temporal drift
│   ├── resample.py          # the ten treatments, and the wrong order run deliberately
│   ├── bootstrap.py         # percentile and BCa intervals, out-of-bag error, .632 and .632+, stability
│   ├── errors.py            # error slices, risk deciles, threshold rules, cost and approval curves
│   ├── plots.py             # nine publication-quality figures
│   ├── render_figures.py    # redraw the figures from cached inputs, without the full run
│   └── run_analysis.py      # orchestrator -> figures + tables + results.json
├── data/                    # SOURCE.txt only (provenance, checksums, filter arithmetic)
├── figures/                 # 9 PNGs (fig01..fig09)
└── outputs/                 # results.json + CSV tables
```

## Headline result

A penalized logistic model on origination-time features reaches **0.6734
ROC-AUC** on 23,474 held-out loans, with a 95% bootstrap interval of 0.6633 to
0.6825. Add the sixteen columns recorded *after* the money was lent and it
reaches **0.9996**. One of those columns used alone, as the entire model, reaches
0.8651: a repaid loan never generates a recovery, so `recoveries > 0` is true for
72.79% of defaults and 0.00% of repayments.

That gap is the study's spine, because **everything else is smaller than the
confidence interval around a single estimate**:

| Decision | Worth, in ROC-AUC |
|---|---:|
| Post-origination columns in the feature set | **0.3262** |
| Shuffled validation protocol vs forward chaining | 0.0155 |
| Ten treatments of the class imbalance | 0.0055 |
| Three classifiers | 0.0048 |
| Fifteen features instead of 48 | 0.0019 |
| Five shuffled cross-validation protocols | 0.0013 |
| Standardization vs min-max vs robust scaling | 0.0005 |
| Median vs mean vs chained-equation vs kNN imputation | 0.0002 |
| *(the 95% interval around the headline number is 0.0192 wide)* | |

**Resampling did not improve the model.** Ten treatments applied inside every
training fold span 0.0055 of ROC-AUC against a 0.0074-to-0.0090 fold-to-fold
standard deviation, so none is distinguishable from leaving the fold alone. They
appear to transform it: F1 at the 0.5 cut-off rises sixteen-fold, from 0.0187 to
0.3091. Give each model its own cut-off instead and all ten land between 0.3148
and 0.3170, with the untouched model third of ten. What resampling reliably does
change is calibration, which degrades by a factor of 59 to 86; SMOTE-ENN, trained
on a fold that is 65.9% defaults, reports a 61% mean default probability on a
book that defaults at 13%.

**Order of operations matters more than method.** Resampling the cohort and
splitting afterwards, rather than resampling inside the training fold, inflates
the reported ROC-AUC by 0.0031 for random oversampling, 0.0083 for SMOTE, and
0.1052 for SMOTE-ENN, whose cleaning step deletes hard cases out of what becomes
the test set.

**The threshold is where the money is.** Errors priced from the cohort's own cash
flows put a missed default at $6,072 against a false alarm's $1,877, a ratio of
3.24. At the 0.5 cut-off the model declines 55 of 23,474 applicants and is 86.6%
accurate. Maximizing F1 costs $819.82 a loan and Youden's J costs $932.63, both
worse than the $807.30 of doing nothing; only the cost-optimal cut-off improves
on it, at $774.15. And the score does order borrowers: observed default rates
climb monotonically from 3.02% in the safest risk decile to 28.04% in the
riskiest.

## Reproducibility

Every figure, table, and `results.json` value is regenerated by a single
deterministic run of `src/run_analysis.py` under master seed `20260815`. Two
consecutive runs were compared file by file: all nine figures and 33 of the 37
tables are byte-identical, and the four that differ do so only in a `seconds`
column of wall clock. Every computed value is reproduced exactly. That required pinning
the linear-algebra backend to one thread before NumPy loads, because a threaded
reduction leaves its summation order to the scheduler and moves the last bits of
every probability. It also turned out to be much faster: almost every fit here is
small, and a hundred logistic fits on a 299 x 48 matrix take 8.25 seconds across
eight threads against 0.14 seconds on one. The resamplers get their parallelism
from joblib instead, where each neighbour query is independent and the merge is
by row index, so it cannot change the answer.

The loader verifies the source file's MD5 and exact shape, asserts that every
`loan_status` value is one the configuration accounts for, and recomputes the
cohort, split, and class counts rather than trusting the configuration file.

## Data source

*Credit Risk Analysis*, Ranadeep (2021), Kaggle:
https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset
(LendingClub's published accepted-loan file, 887,379 loans issued June 2007 to
December 2015, 74 columns, snapshot January 2016.) Provenance, checksums, the
cohort arithmetic, the excluded-column groups, and the cost model are all in
`data/SOURCE.txt`.

## Note on the written report

The full written analysis and its supporting reference material are kept private
and are intentionally **not** part of this repository (see `.gitignore`). This
repo tracks only the code and the reproducible run artifacts (figures, outputs).
