"""Publication-quality figures for the report. Pure matplotlib, clean APA-friendly
styling matched to the earlier projects (150 dpi, no top/right spines, subtle grid).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import models as M

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

INK = "#1b2a41"
DEFAULT = "#c0392b"       # the minority class: loans that defaulted
REPAID = "#2c6fbb"        # the majority class
ACCENT = "#4d9078"
WARN = "#e0a458"
SERIES = ["#2c6fbb", "#c0392b", "#4d9078", "#e0a458", "#7d5ba6", "#5f6b7a",
          "#b5651d", "#3f7d8c", "#a63d6b", "#6b8f3a"]

SHORT_METHOD = {
    "Class weight": "Class weight", "Random over": "Random over",
    "Random under": "Random under", "Tomek links": "Tomek",
    "SMOTE-Tomek": "SMOTE-Tomek", "SMOTE-ENN": "SMOTE-ENN",
}
SHORT_PROTOCOL = {
    "K-fold (5)": "K-fold 5", "K-fold (10)": "K-fold 10",
    "Stratified k-fold (5)": "Strat. 5", "Stratified k-fold (10)": "Strat. 10",
    "Repeated stratified (5 x 5)": "Repeated 5x5",
    "Time-series split (5)": "Time-series", "LOOCV": "LOOCV",
}


def _panel(ax, letter, title):
    ax.set_title(f"{letter}. {title}", loc="left", fontweight="bold", fontsize=10)


def _short(name, table=None):
    return (table or SHORT_METHOD).get(name, name)


# ===================================================== fig 1: the cohort
def fig_cohort(raw, cohort, funnel, vintages, path):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))

    ax = axes[0]                                        # the funnel
    d = funnel[funnel["stage"] != "Resolved and unmatured"]
    x = np.arange(len(d))
    ax.bar(x, d["loans"] / 1000, color=[REPAID, "#8fa8c8", ACCENT],
           edgecolor="white")
    for xi, (n, r) in enumerate(zip(d["loans"], d["default_rate"])):
        tail = "outcome unknown" if np.isnan(r) else f"{r*100:.2f}% default"
        ax.text(xi, n / 1000 + 18, f"{n:,}\n{tail}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["All accepted", "Resolved", "Analysis\ncohort"], fontsize=8.5)
    ax.set_ylabel("Loans (thousands)")
    ax.set_ylim(0, max(d["loans"]) / 1000 * 1.30)
    _panel(ax, "A", "Cohort size at each filter stage")

    ax = axes[1]                                        # the maturity bias
    unmatured = float(funnel.loc[funnel["stage"] == "Resolved and unmatured",
                                 "default_rate"].iloc[0])
    matured = float(funnel.loc[funnel["stage"] == "Analysis cohort",
                               "default_rate"].iloc[0])
    ax.bar([0, 1], [matured * 100, unmatured * 100], color=[ACCENT, DEFAULT],
           width=0.55, edgecolor="white")
    for xi, v in enumerate([matured, unmatured]):
        ax.text(xi, v * 100 + 0.4, f"{v*100:.2f}%", ha="center", fontsize=9.5,
                fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Full term elapsed\n(kept)", "Term not elapsed\n(excluded)"],
                       fontsize=8.5)
    ax.set_ylabel("Observed default rate (%)")
    ax.set_ylim(0, max(matured, unmatured) * 100 * 1.25)
    _panel(ax, "B", "Default rate by maturity-filter outcome")

    ax = axes[2]                                        # vintages
    ax2 = ax.twinx()
    ax2.bar(vintages["issue_year"], vintages["loans"] / 1000, color="#c8d4e3",
            edgecolor="white", width=0.72, zorder=1)
    ax2.set_ylabel("Loans issued (thousands)", color="#7b8a9c", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#7b8a9c", labelsize=8)
    # Headroom so the volume bars stay under the default-rate line rather than
    # running through it.
    ax2.set_ylim(0, float(vintages["loans"].max()) / 1000 * 2.1)
    ax2.grid(False)
    ax.plot(vintages["issue_year"], vintages["default_rate"] * 100, "o-",
            color=DEFAULT, lw=2.0, ms=5, zorder=3)
    ax.axhline(cohort[C.TARGET].mean() * 100, color=INK, ls="--", lw=1,
               label=f"cohort {cohort[C.TARGET].mean()*100:.1f}%", zorder=2)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xlabel("Year of issue")
    ax.set_ylabel("Default rate (%)", color=DEFAULT)
    ax.tick_params(axis="y", labelcolor=DEFAULT)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _panel(ax, "C", "Loans issued and default rate by year")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ===================================================== fig 2: leakage
def fig_leakage(audit, leakage, path):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))

    ax = axes[0]                                        # separability of leaky columns
    d = audit.iloc[::-1]
    ypos = np.arange(len(d))
    ax.barh(ypos + 0.19, d["share_of_defaults"] * 100, height=0.36, color=DEFAULT,
            edgecolor="white", label="of defaults")
    ax.barh(ypos - 0.19, d["share_of_repaid"] * 100, height=0.36, color=REPAID,
            edgecolor="white", label="of repaid loans")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{c}\n{r}" for c, r in zip(d["column"], d["rule"])],
                       fontsize=7)
    ax.set_xlabel("Share of the class satisfying the rule (%)")
    ax.set_xlim(0, 108)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _panel(ax, "A", "Class separation by post-origination column")

    ax = axes[1]                                        # AUC with and without
    d = pd.DataFrame(leakage["features"])
    x = np.arange(len(d))
    ax.bar(x, d["roc_auc"], color=[ACCENT, DEFAULT, WARN], width=0.55,
           edgecolor="white")
    for xi, v in enumerate(d["roc_auc"]):
        ax.text(xi, v + 0.012, f"{v:.4f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.axhline(0.5, color="#999", ls="--", lw=1)
    ax.annotate("chance", (len(d) - 0.45, 0.5), fontsize=7.5, color="#888",
                ha="right", va="bottom", textcoords="offset points", xytext=(0, 3))
    ax.set_xticks(x)
    ax.set_xticklabels(["Origination-time\nfeatures", "Plus post-\norigination",
                        "`recoveries` > 0\nalone"], fontsize=8)
    ax.set_ylabel("Hold-out ROC-AUC")
    ax.set_ylim(0.45, 1.06)
    _panel(ax, "B", "Hold-out ROC-AUC by feature set")

    ax = axes[2]                                        # resampling order
    d = pd.DataFrame(leakage["resampling_order"])
    x = np.arange(len(d))
    w = 0.36
    ax.bar(x - w / 2, d["resampled_inside_the_fold"], w, color=ACCENT,
           edgecolor="white", label="resampled inside the fold")
    ax.bar(x + w / 2, d["resampled_before_split"], w, color=DEFAULT,
           edgecolor="white", label="resampled before the split")
    for xi, (a, b) in enumerate(zip(d["resampled_inside_the_fold"],
                                    d["resampled_before_split"])):
        ax.text(xi + w / 2, b + 0.008, f"+{b - a:.3f}", ha="center", fontsize=8,
                color=DEFAULT, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(d["method"], fontsize=8.5)
    ax.set_ylabel("ROC-AUC reported")
    ax.set_ylim(0.5, max(d["resampled_before_split"]) * 1.10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _panel(ax, "C", "Reported ROC-AUC by resampling order")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ===================================================== fig 3: features
def fig_features(design, cohort, selection, imputation, sel_results, path):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2))

    ax = axes[0]                                        # missingness
    informative = set(imputation.get("informative", []))
    rates = pd.Series(imputation["missing_rates"]).sort_values() * 100
    rates = rates[rates > 0]
    colors = [DEFAULT if i in informative else REPAID for i in rates.index]
    ax.barh(np.arange(len(rates)), rates.to_numpy(), color=colors,
            edgecolor="white")
    for yi, v in enumerate(rates.to_numpy()):
        ax.text(v + 1.5, yi, f"{v:.1f}", va="center", fontsize=6.5, color="#555")
    ax.set_yticks(np.arange(len(rates)))
    ax.set_yticklabels([C.LABELS.get(i, i.replace("_", " ").capitalize())
                        for i in rates.index], fontsize=7.5)
    ax.set_xlabel("Missing in the source file (%)")
    ax.set_xlim(0, 112)
    ax.plot([], [], color=REPAID, lw=6, label="accidental gap")
    ax.plot([], [], color=DEFAULT, lw=6, label="blank means something")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _panel(ax, "A", "Missingness by column type")

    ax = axes[1]                                        # what the choices are worth
    d = pd.DataFrame(imputation["rows"])
    base = d["roc_auc"].max()
    ypos, labels, cols = [], [], []
    for i, (_, row) in enumerate(d.iterrows()):
        ypos.append(base - row["roc_auc"])
        labels.append(row["choice"][:26])
        cols.append(REPAID if row["question"].startswith("Sparse") else DEFAULT)
    order = np.arange(len(ypos))
    ax.barh(order, ypos, color=cols, edgecolor="white")
    ax.set_yticks(order)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC given up against the best choice")
    ax.plot([], [], color=REPAID, lw=6, label="sparse accidental gaps")
    ax.plot([], [], color=DEFAULT, lw=6, label="informative blanks")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _panel(ax, "B", "ROC-AUC forgone by preprocessing choice")

    ax = axes[2]                                        # selector agreement
    d = pd.DataFrame(sel_results["agreement"])
    counts = d["n_methods"].value_counts().sort_index()
    ax.bar(counts.index, counts.to_numpy(),
           color=[WARN, "#8fa8c8", ACCENT][:len(counts)], width=0.6,
           edgecolor="white")
    for xi, v in zip(counts.index, counts.to_numpy()):
        ax.text(xi, v + 0.4, str(v), ha="center", fontsize=9.5, fontweight="bold")
    ax.set_xticks(list(counts.index))
    ax.set_xticklabels([f"{i} of 3" for i in counts.index])
    ax.set_xlabel("Selectors that kept the feature")
    ax.set_ylabel("Features")
    ax.set_ylim(0, counts.max() * 1.34)
    perf = pd.DataFrame(sel_results["performance"])
    for i, (_, r) in enumerate(perf.iterrows()):
        ax.text(0.02, 0.955 - i * 0.055, f"{r['method']}: {r['roc_auc']:.4f}",
                transform=ax.transAxes, fontsize=7, color="#555")
    _panel(ax, "C", f"Selector agreement on the top {sel_results['k']} features")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ===================================================== fig 4: cross-validation
def fig_crossval(records, sweep, cv_results, path):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2))

    ax = axes[0]                                        # fold-score spread
    keep = [r for r in records if len(r["folds"])]
    data = [r["folds"]["roc_auc"].to_numpy() for r in keep]
    labels = [_short(r["name"], SHORT_PROTOCOL) for r in keep]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55,
                    medianprops=dict(color=INK, lw=1.4),
                    flierprops=dict(marker="o", ms=3, mfc="#999", mec="none"))
    for patch, col in zip(bp["boxes"], SERIES):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
        patch.set_edgecolor("white")
    for i, r in enumerate(keep):
        ax.scatter(np.full(len(r["folds"]), i + 1), r["folds"]["roc_auc"],
                   s=9, color=INK, alpha=0.5, zorder=3)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=7.5)
    ax.set_ylabel("Fold ROC-AUC")
    _panel(ax, "A", "Fold ROC-AUC by validation protocol")

    ax = axes[1]                                        # what stratification controls
    for scheme, col, mk in [("K-fold", DEFAULT, "o"),
                            ("Stratified k-fold", ACCENT, "s")]:
        d = sweep[sweep["scheme"] == scheme].sort_values("n")
        ax.plot(d["n"], d["fold_rate_sd"] * 100, mk + "-", color=col, lw=1.9,
                ms=5, label=scheme)
    ax.set_xscale("log")
    ax.set_xlabel("Loans available to the ten folds")
    ax.set_ylabel("SD of the fold default rate (percentage points)")
    ax.legend(frameon=False, fontsize=8.5)
    _panel(ax, "B", "Fold default-rate variance by sample size")

    ax = axes[2]                                        # temporal drift
    d = pd.DataFrame(cv_results["temporal_drift"])
    if not len(d):
        _panel(ax, "C", "Forward-chaining ROC-AUC by issue year")
        fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
        return
    ax2 = ax.twinx()
    ax2.bar(d["issue_year"], d["default_rate"] * 100, color="#e6cfa8",
            edgecolor="white", width=0.7, zorder=1)
    ax2.set_ylabel("Default rate (%)", color="#a8823f", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#a8823f", labelsize=8)
    ax2.grid(False)
    ax.plot(d["issue_year"], d["roc_auc"], "o-", color=INK, lw=2.0, ms=5, zorder=3)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xticks(list(d["issue_year"]))
    ax.set_xticklabels([str(int(v)) for v in d["issue_year"]])
    ax2.set_ylim(0, float(d["default_rate"].max()) * 100 * 2.2)
    ax.set_xlabel("Year of issue")
    ax.set_ylabel("ROC-AUC on that vintage")
    _panel(ax, "C", "Forward-chaining ROC-AUC by issue year")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ===================================================== fig 5: what resampling does
def fig_resampling(shapes, grid, neigh, path):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3))

    ax = axes[0]                                        # rows before and after
    d = shapes.iloc[::-1]
    ypos = np.arange(len(d))
    ax.barh(ypos, d["majority_after"] / 1000, color=REPAID, edgecolor="white",
            label="majority (repaid)")
    ax.barh(ypos, d["minority_after"] / 1000, left=d["majority_after"] / 1000,
            color=DEFAULT, edgecolor="white", label="minority (default)")
    before = float(d["rows_before"].iloc[0]) / 1000
    ax.axvline(before, color=INK, ls="--", lw=1,
               label=f"training fold ({before*1000:,.0f} rows)")
    ax.set_yticks(ypos)
    ax.set_yticklabels([_short(m) for m in d["method"]], fontsize=8)
    ax.set_xlabel("Training rows after resampling (thousands)")
    ax.set_xlim(0, float(d["rows_after"].max()) / 1000 * 1.34)
    ax.legend(frameon=False, fontsize=7, loc="lower right", labelspacing=0.3)
    _panel(ax, "A", "Training-fold composition by method")

    ax = axes[1]                                        # ranking vs threshold metric
    d = grid.query("model == 'Logistic regression'").copy()
    ypos = np.arange(len(d))
    ax.scatter(d["roc_auc"], ypos, s=62, color=REPAID, edgecolor="white",
               linewidth=0.8, zorder=3, label="ROC-AUC")
    ax.scatter(d["f1"], ypos, s=62, color=DEFAULT, edgecolor="white",
               linewidth=0.8, marker="s", zorder=3, label="F1 at the 0.5 cut-off")
    base = d[d["method"] == "None"]
    ax.axvline(float(base["roc_auc"].iloc[0]), color=REPAID, ls=":", lw=1.2)
    ax.axvline(float(base["f1"].iloc[0]), color=DEFAULT, ls=":", lw=1.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([_short(m) for m in d["method"]], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Score (dotted lines: the untouched fold)")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    _panel(ax, "B", "ROC-AUC and F1 at 0.5 by method")

    ax = axes[2]                                        # what a synthetic row is
    if len(neigh):
        d = neigh
        ypos = np.arange(len(d))
        ax.barh(ypos, d["ratio"], color=[WARN if r < 0.6 else DEFAULT
                                         for r in d["ratio"]], edgecolor="white")
        ax.axvline(1.0, color=INK, ls="--", lw=1.2,
                   label="as far apart as real defaults are")
        for yi, (r, s) in enumerate(zip(d["ratio"], d["share_duplicate"])):
            ax.text(r + 0.03, yi, f"{r:.2f}" + (f"   {s*100:.0f}% are copies"
                                                if s > 0.01 else ""),
                    va="center", fontsize=7.5, color="#444")
        ax.set_yticks(ypos)
        ax.set_yticklabels([_short(m) for m in d["method"]], fontsize=8.5)
        ax.set_xlabel("Distance to the nearest real default, relative to the\n"
                      "typical distance between real defaults")
        ax.set_xlim(0, max(1.35, float(d["ratio"].max()) * 1.45))
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    _panel(ax, "C", "Distance from synthetic to nearest real default")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ===================================================== fig 6: the effect
def fig_resampling_effect(holdout, resampling, y_te, path):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    show = ["None", "Class weight", "Random under", "SMOTE", "ADASYN", "SMOTE-ENN"]
    picked = [h for h in holdout if h["method"] in show]
    y_te = np.asarray(y_te).astype(int)

    ax = axes[0]                                        # predicted risk distribution
    for h, col in zip(picked, SERIES):
        p = np.sort(h["predictions"])
        ax.plot(p, np.linspace(0, 1, len(p)), color=col, lw=1.7, label=h["method"])
    ax.axvline(0.5, color="#999", ls="--", lw=1)
    ax.annotate("0.5", (0.5, 0.03), fontsize=7.5, color="#888",
                textcoords="offset points", xytext=(3, 0))
    ax.set_xlabel("Predicted probability of default")
    ax.set_ylabel("Cumulative share of held-out loans")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    _panel(ax, "A", "Predicted-risk distribution by method")

    ax = axes[1]                                        # calibration
    for h, col in zip(picked, SERIES):
        rel = M.reliability(y_te, np.asarray(h["predictions"], dtype=float))
        ax.plot(rel["predicted"], rel["observed"], "o-", color=col, lw=1.6, ms=4,
                label=f"{h['method']} ({h['ece']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed default rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    _panel(ax, "B", "Reliability curves and calibration error")

    ax = axes[2]                                        # the control
    d = pd.DataFrame(resampling["threshold_control"])
    ypos = np.arange(len(d))
    ax.barh(ypos - 0.19, d["f1_at_half"], height=0.36, color="#8fa8c8",
            edgecolor="white", label="F1 at the 0.5 cut-off")
    ax.barh(ypos + 0.19, d["f1_at_best"], height=0.36, color=ACCENT,
            edgecolor="white", label="F1 at that model's best cut-off")
    ax.set_yticks(ypos)
    ax.set_yticklabels([_short(m) for m in d["method"]], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("F1 on the out-of-fold predictions")
    ax.set_xlim(0, max(float(d["f1_at_best"].max()), float(d["f1_at_half"].max())) * 1.45)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelspacing=0.3)
    _panel(ax, "C", "F1 at 0.5 against F1 at each model's cut-off")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ===================================================== fig 7: bootstrap
def fig_bootstrap(boot, path):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))

    ax = axes[0]                                        # both intervals per metric
    d = pd.DataFrame(boot["intervals"]).iloc[::-1]
    ypos = np.arange(len(d))
    for yi, (_, r) in enumerate(d.iterrows()):
        ax.plot([r["percentile_lo"], r["percentile_hi"]], [yi + 0.16] * 2,
                color=REPAID, lw=2.6, solid_capstyle="butt")
        ax.plot([r["bca_lo"], r["bca_hi"]], [yi - 0.16] * 2, color=DEFAULT,
                lw=2.6, solid_capstyle="butt")
        ax.scatter([r["observed"]], [yi], s=26, color=INK, zorder=4)
    ax.plot([], [], color=REPAID, lw=2.6, label="percentile")
    ax.plot([], [], color=DEFAULT, lw=2.6, label="BCa")
    ax.set_yticks(ypos)
    ax.set_yticklabels(d["metric"], fontsize=8)
    ax.set_xlabel("95% confidence interval")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _panel(ax, "A", "Percentile and BCa intervals by metric")

    ax = axes[1]                                        # error estimators
    oob = boot["out_of_bag"]
    models = list(oob)
    kinds = [("apparent_error", "Apparent (resubstitution)", WARN),
             ("oob_error", "Out-of-bag", DEFAULT),
             ("error_632", ".632", "#8fa8c8"),
             ("error_632_plus", ".632+", ACCENT)]
    w = 0.8 / len(kinds)
    x = np.arange(len(models))
    for k, (key, label, col) in enumerate(kinds):
        vals = [oob[m][key] for m in models]
        ax.bar(x + (k - (len(kinds) - 1) / 2) * w, vals, w, color=col,
               edgecolor="white", label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=8.5)
    ax.set_ylabel("Misclassification rate")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center")
    ax.set_ylim(0, max(oob[m]["oob_error"] for m in models) * 1.55)
    _panel(ax, "B", "Four estimates of misclassification rate")

    ax = axes[2]                                        # stability
    d = pd.DataFrame(boot["stability"]).head(14).iloc[::-1]
    ypos = np.arange(len(d))
    cols = [ACCENT if s >= 0.99 else (WARN if s >= 0.9 else DEFAULT)
            for s in d["sign_agreement"]]
    ax.barh(ypos, d["mean_coefficient"], color=cols, edgecolor="white")
    ax.errorbar(d["mean_coefficient"], ypos,
                xerr=[d["mean_coefficient"] - d["lo"], d["hi"] - d["mean_coefficient"]],
                fmt="none", ecolor=INK, elinewidth=1.0, capsize=2)
    ax.axvline(0, color="#999", ls=":", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f if len(f) <= 30 else f[:29] + "." for f in d["feature"]],
                       fontsize=6.5)
    ax.set_xlabel("Coefficient across bootstrap refits (95% band)")
    span = max(abs(float(d["lo"].min())), abs(float(d["hi"].max()))) * 1.08
    ax.set_xlim(-span, span * 1.05)
    ax.set_ylim(-0.9, len(d) - 0.5)
    ax.plot([], [], color=ACCENT, lw=6, label="sign never flips")
    ax.plot([], [], color=WARN, lw=6, label="flips rarely")
    ax.plot([], [], color=DEFAULT, lw=6, label="flips often")
    ax.legend(frameon=False, fontsize=6.5, loc="lower center", ncol=3,
              columnspacing=0.8, handletextpad=0.3, handlelength=1.2)
    _panel(ax, "C", "Coefficient stability across 200 refits")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ===================================================== fig 8: errors
def fig_errors(err, y_te, p_te, path):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    labels = ["Repaid", "Default"]

    for k, (key, title) in enumerate([("confusion_at_half", "Confusion matrix at the 0.5 cut-off"),
                                      ("confusion_at_cost", "Confusion matrix at the cost-optimal cut-off")]):
        ax = axes[k]
        cm = np.array(err[key])
        total = cm.sum()
        ax.imshow(cm.astype(float), cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Approve", "Decline"])
        ax.set_yticklabels(labels)
        ax.set_xlabel("Model's decision")
        ax.set_ylabel("What the loan did")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / total * 100:.1f}%)",
                        ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > cm.max() * 0.5 else INK)
        ax.grid(False)
        _panel(ax, "AB"[k], title)

    ax = axes[2]                                        # risk deciles
    d = pd.DataFrame(err["deciles"])
    x = np.arange(len(d))
    ax.bar(x, d["observed_default_rate"] * 100, color=REPAID, edgecolor="white",
           width=0.66, label="observed")
    ax.plot(x, d["mean_predicted"] * 100, "o-", color=DEFAULT, lw=1.9, ms=5,
            label="predicted")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in d["decile"]])
    ax.set_xlabel("Risk decile (1 = safest)")
    ax.set_ylabel("Default rate (%)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _panel(ax, "C", "Observed and predicted default rate by decile")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ===================================================== fig 9: the operating point
def fig_operating_point(err, costs, path):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))

    ax = axes[0]                                        # cost curve
    d = pd.DataFrame(err["cost_curve"])
    ax.plot(d["threshold"], d["cost_per_loan"], color=INK, lw=2.0)
    best = d.loc[d["cost_per_loan"].idxmin()]
    ax.scatter([best["threshold"]], [best["cost_per_loan"]], s=70, color=ACCENT,
               zorder=4, edgecolor="white", linewidth=1.0)
    ax.annotate(f"minimum at {best['threshold']:.3f}\n${best['cost_per_loan']:,.0f} per loan",
                (best["threshold"], best["cost_per_loan"]), fontsize=8,
                textcoords="offset points", xytext=(12, 16), color=ACCENT)
    half = d.iloc[(d["threshold"] - 0.5).abs().argmin()]
    ax.scatter([half["threshold"]], [half["cost_per_loan"]], s=70, color=DEFAULT,
               zorder=4, edgecolor="white", linewidth=1.0)
    ax.annotate(f"0.5 cut-off\n${half['cost_per_loan']:,.0f} per loan",
                (half["threshold"], half["cost_per_loan"]), fontsize=8,
                textcoords="offset points", xytext=(-4, 58), ha="right",
                color=DEFAULT)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected cost per loan ($)")
    ax.set_xlim(0, min(1.0, float(d["threshold"].max())))
    _panel(ax, "A", "Expected cost per loan by threshold")

    ax = axes[1]                                        # precision and recall
    t = pd.DataFrame(err["thresholds"])
    ax.plot(d["threshold"], d["recall"], color=DEFAULT, lw=1.9, label="recall")
    ax.plot(d["threshold"], d["precision"], color=REPAID, lw=1.9, label="precision")
    ax.plot(d["threshold"], d["f1"], color=ACCENT, lw=1.9, ls="--", label="F1")
    # Several rules select nearly the same cut-off, and rotated labels at the
    # same x overlap however they are staggered in height. Rules within 0.01 of
    # each other are therefore collapsed into one label naming all of them.
    groups = []
    for _, row in t.sort_values("threshold").iterrows():
        name = row["rule"].split(" (")[0]
        if groups and abs(row["threshold"] - groups[-1][0]) < 0.01:
            groups[-1][1].append(name)
        else:
            groups.append([float(row["threshold"]), [name]])
    for x, names in groups:
        ax.axvline(x, color="#bbb", lw=0.8, ls=":")
        ax.annotate(" / ".join(names), (x, 1.03), rotation=90, fontsize=6.0,
                    color="#777", ha="right", va="bottom")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(0, min(1.0, float(d["threshold"].max())))
    ax.set_ylim(0, 1.30)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, fontsize=8, loc="center right")
    _panel(ax, "B", "Precision, recall, and F1 by threshold")

    ax = axes[2]                                        # portfolio view
    d = pd.DataFrame(err["approval_curve"])
    ax.plot(d["approval_rate"] * 100, d["net"] / 1e6, color=INK, lw=2.0)
    peak = d.loc[d["net"].idxmax()]
    ax.scatter([peak["approval_rate"] * 100], [peak["net"] / 1e6], s=70,
               color=ACCENT, zorder=4, edgecolor="white", linewidth=1.0)
    ax.annotate(f"approve {peak['approval_rate']*100:.0f}%\n${peak['net']/1e6:.2f}M net",
                (peak["approval_rate"] * 100, peak["net"] / 1e6), fontsize=8,
                textcoords="offset points", xytext=(-10, -28), ha="center",
                color=ACCENT)
    ax.axhline(float(d["net"].iloc[-1]) / 1e6, color=DEFAULT, ls="--", lw=1.2,
               label=f"approve everyone (${d['net'].iloc[-1]/1e6:.2f}M)")
    ax.set_xlabel("Share of applicants approved (%)")
    ax.set_ylabel("Interest earned less losses ($M)")
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    _panel(ax, "C", "Net return by share of applicants approved")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
