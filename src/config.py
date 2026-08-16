"""Central configuration for the Week 7 credit-risk validation study.

Every tunable constant lives here so the whole pipeline is deterministic and
auditable from one place: the master seed, the source file and its checksum, the
cohort definition, the origination-time feature schema, the columns excluded as
post-origination leakage, the cross-validation protocols, the resampling grid,
the bootstrap settings, and the cost model that turns errors into money.
"""
from pathlib import Path

# ----------------------------------------------------------------- reproducibility
SEED = 20260815                      # master seed (date of the run)

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"

for _d in (FIG_DIR, OUT_DIR):
    _d.mkdir(exist_ok=True)

# ----------------------------------------------------------------- source data
# The Kaggle mirror of LendingClub's 2007-2015 accepted-loan file. The loader
# verifies this MD5 and the row count before reading, so a truncated or
# substituted file fails loudly rather than trains silently.
RAW_FILE = "loan.csv"
RAW_MD5 = "807c374fa7841310a45e411eb8c399fd"
RAW_ROWS = 887_379
RAW_COLS = 74

# ----------------------------------------------------------------- cohort
# A loan enters the study only if its outcome is final. LendingClub's status
# field mixes terminal states with in-flight ones, and treating "Current" as a
# non-default is the standard way this dataset is misused: it labels a loan that
# has not yet had the chance to default.
STATUS_DEFAULT = [
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
]
STATUS_REPAID = [
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
]
STATUS_UNRESOLVED = [
    "Current", "Issued", "In Grace Period",
    "Late (16-30 days)", "Late (31-120 days)",
]

# The file is a January 2016 snapshot. A resolved loan is kept only if its full
# contractual term had elapsed by then, so the cohort is not enriched with the
# loans that resolved fastest. The maturity filter costs 70% of the resolved
# rows and is measured in the report rather than assumed.
SNAPSHOT = "2016-01-01"
DAYS_PER_MONTH = 30.44               # term is contractual months; issue_d is monthly

TARGET = "default"                   # 1 = charged off or defaulted, 0 = fully repaid

# ----------------------------------------------------------------- leakage
# Recorded after funding, so each is a function of what the borrower went on to
# do; a model that sees them reads the outcome. `recoveries` is the extreme
# case, non-zero for 52.3% of defaults and for no repaid loan.
POST_ORIGINATION = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d", "collections_12_mths_ex_med",
    "funded_amnt", "funded_amnt_inv",
]

# Fields LendingClub only began recording partway through the cohort. Their
# missingness is a property of the reporting date and not of the borrower, so
# keeping them would let a model shortcut from "this field is blank" to "this
# loan was issued before 2012" to that vintage's default rate.
NOT_YET_COLLECTED = [
    "tot_cur_bal", "tot_coll_amt", "total_rev_hi_lim", "open_acc_6m",
    "open_il_6m", "open_il_12m", "open_il_24m", "mths_since_rcnt_il",
    "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc",
    "all_util", "inq_fi", "total_cu_tl", "inq_last_12m",
    "annual_inc_joint", "dti_joint", "verification_status_joint",
    "mths_since_last_major_derog",
]

# Identifiers, free text, and constants that carry no origination-time signal.
NON_PREDICTIVE = [
    "id", "member_id", "url", "desc", "emp_title", "title", "zip_code",
    "policy_code", "application_type", "pymnt_plan", "acc_now_delinq",
    "initial_list_status",
]

# ----------------------------------------------------------------- features
# Recorded at application, used as given.
NUMERIC_RAW = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "term_months",
]

# Built from domain knowledge in features.py. The brief asks for a
# debt-to-income ratio, a credit-utilization rate, payment-history patterns, and
# a credit-age calculation; these are those, plus the ratios a credit analyst
# would compute by hand before looking at anything else.
NUMERIC_DERIVED = [
    "credit_age_years",          # months between earliest credit line and issue
    "log_annual_inc",
    "loan_to_income",            # principal as a multiple of annual income
    "installment_to_income",     # annualized payment as a share of income
    "total_debt_burden",         # reported DTI plus this loan's payment burden
    "revol_bal_to_income",
    "open_acc_ratio",            # share of the borrower's accounts still open
    "log_revol_bal",
    "months_since_delinq",       # imputed at the ceiling when never delinquent
    "months_since_record",
    "emp_length_years",          # ordinal, 0 to 10
]

# Missingness that is informative rather than accidental: a blank
# mths_since_last_delinq means no delinquency was ever recorded, which is a fact
# about the borrower. Contrast NOT_YET_COLLECTED, where a blank is a fact about
# LendingClub.
MISSING_INDICATORS = [
    "never_delinquent", "no_public_record", "emp_length_missing",
]

# Interaction terms between the correlated predictors named in the brief. Each
# pair is chosen because the two members are individually predictive and
# correlated with each other, so their product carries what neither carries
# alone.
INTERACTIONS = [
    ("int_rate", "dti"),
    ("int_rate", "loan_to_income"),
    ("revol_util", "credit_age_years"),
    ("inq_last_6mths", "open_acc_ratio"),
]

# One-hot encoded: few enough levels that a column each is affordable.
ONEHOT = ["home_ownership", "verification_status", "grade"]
HOME_OWNERSHIP_OTHER = ["OTHER", "NONE", "ANY"]      # collapsed into one level

# Target-encoded: too many levels for one-hot without shattering the design
# matrix. The encoding is fitted out of fold inside every training partition,
# which is the only version of target encoding that is not leakage.
TARGET_ENCODE = ["sub_grade", "purpose", "addr_state"]
TE_SMOOTHING = 50.0                  # prior weight, in loans, toward the global rate
TE_FOLDS = 5                         # inner folds used to fit the encoding out of fold

# ----------------------------------------------------------------- partitioning
TEST_SIZE = 0.30                     # stratified hold-out on the outcome
TEMPORAL_CUTOFF = "2012-01-01"       # issue date splitting the temporal hold-out

# ----------------------------------------------------------------- cross-validation
CV_FOLDS = [5, 10]                   # k-fold and stratified k-fold both run at both k
CV_REPEATS = 5                       # repeats used to separate fold noise from seed noise
TS_SPLITS = 5                        # forward-chaining splits over issue date
LOOCV_N = 2_000                      # stratified subsample LOOCV is run on; see below

# Leave-one-out over 54,771 training loans is 54,771 model fits, and it cannot
# produce a per-fold ROC-AUC at all, because a fold holding one loan holds one
# class. It is therefore run on a stratified subsample, at the same size, against
# every other protocol, so the comparison is like for like.

# ----------------------------------------------------------------- resampling
# Every method is applied inside the training fold only. The study also runs the
# incorrect version, resampling before the split, and reports the difference.
RESAMPLERS = [
    "None",                # the untouched imbalance, the reference for everything
    "Class weight",        # cost-sensitive learning instead of resampling
    "Random over",
    "Random under",
    "SMOTE",
    "ADASYN",
    "Tomek links",
    "ENN",
    "SMOTE-Tomek",
    "SMOTE-ENN",
]
SMOTE_K = 5                          # neighbours used to interpolate a synthetic minority point
ENN_K = 3                            # neighbours consulted when cleaning a boundary

# ----------------------------------------------------------------- bootstrap
BOOTSTRAP_N = 1_000                  # resamples for the confidence intervals
BOOTSTRAP_MODEL_N = 200              # resamples on which a model is refitted for OOB
BOOTSTRAP_ALPHA = 0.05               # 95% intervals, percentile and BCa
BOOTSTRAP_JACKKNIFE_N = 2_000        # subsample the BCa acceleration is estimated on

# ----------------------------------------------------------------- models
LOGIT_C = 1.0
LOGIT_MAX_ITER = 2_000
RF_PARAMS = dict(n_estimators=300, min_samples_leaf=25, max_features="sqrt",
                 n_jobs=1, random_state=SEED)
GB_PARAMS = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                 min_samples_leaf=50, l2_regularization=1.0, random_state=SEED)

# ----------------------------------------------------------------- evaluation
CALIBRATION_BINS = 10                # equal-count bins for the reliability curve and ECE

# A missed default loses principal net of recoveries and collection cost; a
# false alarm loses forgone interest. Both are computed in data.py from this
# cohort's realized cash flows, using the columns POST_ORIGINATION hides from
# the model. Pricing an error from history is legitimate; predicting is not.
COST_CLIP = (0.0, 1.0)               # loss given default is bounded to a share of principal

# Human-readable labels for tables and figures.
LABELS = {
    "loan_amnt": "Loan amount", "int_rate": "Interest rate",
    "installment": "Monthly installment", "annual_inc": "Annual income",
    "dti": "Debt-to-income (reported)", "delinq_2yrs": "Delinquencies, 2 years",
    "inq_last_6mths": "Credit inquiries, 6 months", "open_acc": "Open accounts",
    "pub_rec": "Public records", "revol_bal": "Revolving balance",
    "revol_util": "Credit utilization", "total_acc": "Total accounts",
    "term_months": "Term (months)", "credit_age_years": "Credit age (years)",
    "log_annual_inc": "Log annual income", "loan_to_income": "Loan-to-income",
    "installment_to_income": "Payment-to-income",
    "total_debt_burden": "Total debt burden", "revol_bal_to_income": "Revolving-to-income",
    "open_acc_ratio": "Open-account share", "log_revol_bal": "Log revolving balance",
    "months_since_delinq": "Months since delinquency",
    "months_since_record": "Months since public record",
    "emp_length_years": "Employment length (years)",
    "never_delinquent": "Never delinquent", "no_public_record": "No public record",
    "emp_length_missing": "Employment length missing",
    "sub_grade_te": "Sub-grade (target-encoded)", "purpose_te": "Purpose (target-encoded)",
    "addr_state_te": "State (target-encoded)",
}
