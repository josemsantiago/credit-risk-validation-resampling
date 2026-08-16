"""Re-render every figure from cached inputs, without repeating the pipeline.

    python3 render_figures.py            # redraw all nine figures
    python3 render_figures.py --verify   # redraw and check they match the pipeline

The full run takes about forty minutes, nearly all of it in the resampling grid
and the bootstrap, and none of that affects what a figure looks like once the
numbers are fixed. run_analysis.py therefore pickles the arguments the figure
code reads into outputs/plot_inputs.pkl, and this script replays them in
seconds, which makes editing a title or a layout a normal edit-and-look loop.

--verify MD5s every figure before and after. That matters because fig_cohort and
fig_features take raw, design, and selection positionally but never read them,
so None is passed; if that ever stops being true the digests stop matching and
this says so instead of drawing something quietly wrong.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pickle
import hashlib
from pathlib import Path

import config as C
import plots as PL

CACHE = C.OUT_DIR / "plot_inputs.pkl"


def _digests():
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(C.FIG_DIR.glob("*.png"))}


def main(verify=False):
    if not CACHE.exists():
        sys.exit(f"no cache at {CACHE}\nRun `python3 run_analysis.py` once to create it.")

    # Safe to unpickle: written by run_analysis.py on this machine into a
    # gitignored path, never fetched, shared, or committed. Holds pandas frames
    # and numpy arrays, hence pickle rather than JSON. If missing or stale,
    # re-run the pipeline; never source this file from anywhere else.
    d = pickle.loads(CACHE.read_bytes())
    before = _digests() if verify else {}

    PL.fig_cohort(None, {C.TARGET: d["cohort_target"]}, d["funnel"], d["vintages"],
                  C.FIG_DIR / "fig01_cohort.png")
    PL.fig_leakage(d["leak_audit"], d["leakage"],
                   C.FIG_DIR / "fig02_leakage.png")
    PL.fig_features(None, None, None, d["imputation"], d["selection"],
                    C.FIG_DIR / "fig03_features.png")
    PL.fig_crossval(d["cv_records"], d["sweep"], d["cross_validation"],
                    C.FIG_DIR / "fig04_crossvalidation.png")
    PL.fig_resampling(d["resample_shapes"], d["grid_table"], d["neigh"],
                      C.FIG_DIR / "fig05_resampling.png")
    PL.fig_resampling_effect(d["holdout_resample"], d["resampling"], d["y_te"],
                             C.FIG_DIR / "fig06_resampling_effect.png")
    PL.fig_bootstrap(d["bootstrap"], C.FIG_DIR / "fig07_bootstrap.png")
    PL.fig_errors(d["errors"], d["y_te"], d["p_head"],
                  C.FIG_DIR / "fig08_errors.png")
    PL.fig_operating_point(d["errors"], d["costs"],
                           C.FIG_DIR / "fig09_operating_point.png")

    print(f"re-rendered {len(list(C.FIG_DIR.glob('*.png')))} figures from {CACHE.name}")

    if verify:
        after = _digests()
        moved = [k for k in sorted(set(before) | set(after))
                 if before.get(k) != after.get(k)]
        if moved:
            print("  DIFFERS from the pipeline render: " + ", ".join(moved))
            return 1
        print("  byte-identical to the pipeline render for all "
              f"{len(after)} figures")
    return 0


if __name__ == "__main__":
    sys.exit(main(verify="--verify" in sys.argv))
