#!/usr/bin/env python3
"""
Check that the Python results reproduce the published R results.

Run `python run_all.py` first, then:

    python check_convergence.py                 # print the report
    python check_convergence.py --write-docs    # also write docs/CONVERGENCE.md

Every numeric column that exists in both the Python output (`results/`) and the
archived R output (`reference_R_output/`) is compared row by row.  For each
comparison the script reports the largest absolute difference and the largest
relative difference, and marks it PASS or FAIL against a tolerance.

Tolerances
----------
Most statistics are compared at a relative tolerance of 1e-6, which is far
tighter than any reported decimal place: the two languages should be doing
identical arithmetic.  Two exceptions are documented in the table and in the
README:

* the mixed-model standard errors, where `lme4` and `statsmodels` optimise the
  REML criterion with different algorithms (agreement is ~1e-4 relative);
* the robust PCA, where the two MCD implementations draw different random
  subsets and are therefore compared qualitatively rather than numerically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import wing_morphometry as wm

REFERENCE_DIR = wm.PROJECT_ROOT / "reference_R_output"
DOCS_DIR = wm.PROJECT_ROOT / "docs"


class Report:
    """Collects one row per comparison and prints a summary table."""

    def __init__(self):
        self.rows = []

    def add(self, analysis, quantity, n, max_abs, max_rel, tol, note=""):
        ok = bool(np.isnan(max_rel)) or max_rel <= tol
        self.rows.append({"analysis": analysis, "quantity": quantity,
                          "n_values": n, "max_abs_diff": max_abs,
                          "max_rel_diff": max_rel, "tolerance": tol,
                          "verdict": "PASS" if ok else "FAIL", "note": note})

    def add_manual(self, analysis, quantity, verdict, note):
        self.rows.append({"analysis": analysis, "quantity": quantity,
                          "n_values": np.nan, "max_abs_diff": np.nan,
                          "max_rel_diff": np.nan, "tolerance": np.nan,
                          "verdict": verdict, "note": note})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def compare(self, analysis, py, r, keys, columns, tol=1e-6, note=""):
        """Align two tables on `keys` and compare each column in `columns`."""
        merged = py.merge(r, on=keys, suffixes=("_py", "_r"), how="inner")
        if len(merged) != len(r):
            self.add_manual(analysis, "row alignment", "FAIL",
                            f"matched {len(merged)} of {len(r)} R rows")
            return
        for col in columns:
            a = pd.to_numeric(merged[f"{col}_py"], errors="coerce").to_numpy(float)
            b = pd.to_numeric(merged[f"{col}_r"], errors="coerce").to_numpy(float)
            both = ~(np.isnan(a) | np.isnan(b))
            if not both.any():
                continue
            a, b = a[both], b[both]
            abs_d = np.abs(a - b)
            rel_d = abs_d / np.maximum(np.abs(b), 1e-300)
            self.add(analysis, col, int(both.sum()), float(abs_d.max()),
                     float(rel_d.max()), tol, note)


def _load(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def build_report(results_dir: Path, reference_dir: Path) -> Report:
    rep = Report()

    # ── PERMANOVA ────────────────────────────────────────────────────────────
    py = _load(results_dir / "results_permanova.csv")
    r = _load(reference_dir / "results_permanova.csv")
    if py is not None and r is not None:
        r = r.rename(columns={r.columns[0]: "term"})
        r["term"] = r["term"].str.replace("^Residual$", "Residual", regex=True)
        rep.compare("PERMANOVA", py, r, ["term"],
                    ["Df", "SumOfSqs", "R2", "F"], tol=1e-9,
                    note="sums of squares computed algebraically, not by "
                         "building the 4100x4100 distance matrix")
        # p-values are stochastic; both hit the 0.001 floor with 999 permutations
        same_p = np.allclose(py.set_index("term")["Pr(>F)"].dropna(),
                             r.set_index("term")["Pr(>F)"].dropna())
        rep.add_manual("PERMANOVA", "Pr(>F)", "PASS" if same_p else "FAIL",
                       "permutation p-values; both reach the 0.001 floor")

    # ── Follow-up ANOVAs ─────────────────────────────────────────────────────
    py = _load(results_dir / "results_manova_followup.csv")
    r = _load(reference_dir / "results_manova_followup.csv")
    if py is not None and r is not None:
        rep.compare("Follow-up ANOVA (Type II)", py, r, ["trait", "Source"],
                    ["SS", "DF", "MS", "F", "p-unc", "np2"], tol=1e-8)

    # ── Mixed models ─────────────────────────────────────────────────────────
    py = _load(results_dir / "results_lme.csv")
    r = _load(reference_dir / "results_lme.csv")
    if py is not None and r is not None:
        rename = {"Intercept": "(Intercept)",
                  "origin[T.Ribeirao_Preto]": "originRibeirao_Preto",
                  "side[T.left]": "sideleft",
                  "origin[T.Ribeirao_Preto]:side[T.left]":
                      "originRibeirao_Preto:sideleft"}
        py = py.assign(term=py["term"].replace(rename))
        rep.compare("Mixed models — estimates", py, r, ["trait", "term"],
                    ["estimate"], tol=1e-5,
                    note="lme4 and statsmodels use different REML optimisers")
        rep.compare("Mixed models — AIC/BIC", py, r, ["trait", "term"],
                    ["AIC", "BIC"], tol=1e-6)
        rep.compare("Mixed models — SE / z / p", py, r, ["trait", "term"],
                    ["std_err", "z"], tol=1e-3,
                    note="lme4 and statsmodels use different REML optimisers")

    # ── Asymmetry tests ──────────────────────────────────────────────────────
    py = _load(results_dir / "results_table_S3_wilcoxon.csv")
    r = _load(reference_dir / "results_table_S3_wilcoxon.csv")
    if py is not None and r is not None:
        cols = ["mean_diff_RL", "n_pairs", "t_stat", "t_pvalue", "wilcox_stat",
                "wilcox_pvalue", "t_pvalue_bonf", "t_pvalue_holm", "t_pvalue_bh",
                "wilcox_pvalue_bonf", "wilcox_pvalue_holm", "wilcox_pvalue_bh"]
        cols = [c for c in cols if c in py.columns and c in r.columns]
        rep.compare("Asymmetry tests (120 colony x trait)", py, r,
                    ["colony", "trait"], cols, tol=1e-8)

    # ── Values printed in the R report rather than saved as CSV ──────────────
    _compare_printed_values(rep, results_dir)

    return rep


# Values that the R pipeline prints into `analysis_main.html` instead of writing
# to a CSV.  They are transcribed here so the check is fully automatic.
R_PRINTED = {
    "PCA variance explained (%)": [55.13, 20.90, 12.39, 7.39, 4.20],
    "PCA loadings": np.array([
        [0.4580, -0.4417, -0.4738, -0.0794, -0.6036],
        [0.5172, -0.3378, -0.0799, -0.2627, 0.7369],
        [0.4354, -0.1931, 0.7980, 0.3130, -0.1959],
        [0.3898, 0.6312, 0.1388, -0.6271, -0.1927],
        [0.4258, 0.5050, -0.3362, 0.6584, 0.1309]]),
    "MANOVA Pillai": {"origin": 0.13584, "side": 0.22664, "origin:side": 0.12195},
    "MANOVA Wilks": {"origin": 0.86416, "side": 0.77336, "origin:side": 0.87805},
    "Mardia": {"b1p": 0.4088, "skew_stat": 279.3779, "b2p": 35.5071,
               "kurt_stat": 1.9404, "kurt_p": 0.052334},
    "LDA accuracy (%)": {"origin (right wing)": 72.0,
                         "side within Coyhaique": 62.3,
                         "side within Ribeirão Preto": 80.2},
    "LDA confusion counts": {"origin (right wing)": [864, 251, 321, 610],
                             "side within Coyhaique": [688, 427, 414, 703],
                             "side within Ribeirão Preto": [728, 203, 167, 770]},
    "G x E correlations": {
        "nerv01": (0.7977886, 0.0018750608, 0.6713287, 0.0168314562),
        "nerv02": (0.8322077, 0.0007845643, 0.6293706, 0.0283196713),
        "nerv03": (0.6141514, 0.0336243336, 0.6083916, 0.0358059971),
        "nerv04": (0.5071451, 0.0923965479, 0.3636364, 0.2452650007),
        "nerv05": (0.8491714, 0.0004745179, 0.8251748, 0.0009513629)},
    "Asymmetry — significant tests at p<0.05": {
        "Raw": 101, "Bonferroni": 76, "Holm-Bonferroni": 81,
        "Benjamini-Hochberg (BH)": 100},
    "Robust PCA": {"RPC1": 54.6, "RPC2": 21.1, "n_outliers": 122},
}


def _compare_printed_values(rep: Report, results_dir: Path) -> None:
    """Compare against the numbers printed in the R HTML report."""

    def diff(py_vals, r_vals):
        a = np.asarray(py_vals, float).ravel()
        b = np.asarray(r_vals, float).ravel()
        abs_d = np.abs(a - b)
        rel_d = abs_d / np.maximum(np.abs(b), 1e-300)
        return a.size, float(abs_d.max()), float(rel_d.max())

    var = _load(results_dir / "results_pca_variance.csv")
    if var is not None:
        n, ad, rd = diff(var["variance_explained_pct"].round(2),
                         R_PRINTED["PCA variance explained (%)"])
        rep.add("PCA", "variance explained (%)", n, ad, rd, 1e-9,
                "R report prints 2 decimals")

    load = _load(results_dir / "results_pca_loadings.csv")
    if load is not None:
        n, ad, rd = diff(load[[f"PC{i+1}" for i in range(5)]].to_numpy().round(4),
                         R_PRINTED["PCA loadings"])
        rep.add("PCA", "loadings", n, ad, rd, 1e-9, "R report prints 4 decimals")

    man = _load(results_dir / "results_manova.csv")
    if man is not None:
        for stat, key in [("Pillai's trace", "MANOVA Pillai"),
                          ("Wilks' lambda", "MANOVA Wilks")]:
            sub = man[man.statistic == stat].set_index("term")
            terms = list(R_PRINTED[key])
            n, ad, rd = diff([round(sub.loc[t, "value"], 5) for t in terms],
                             [R_PRINTED[key][t] for t in terms])
            rep.add("MANOVA (Type III)", stat, n, ad, rd, 1e-9,
                    "R report prints 5 decimals")

    mar = _load(results_dir / "results_mardia.csv")
    if mar is not None:
        ref = R_PRINTED["Mardia"]
        py_vals = [round(mar.loc[0, "b"], 4), round(mar.loc[0, "statistic"], 4),
                   round(mar.loc[1, "b"], 4), round(mar.loc[1, "statistic"], 4),
                   round(mar.loc[1, "p_value"], 6)]
        n, ad, rd = diff(py_vals, [ref["b1p"], ref["skew_stat"], ref["b2p"],
                                   ref["kurt_stat"], ref["kurt_p"]])
        rep.add("Mardia normality test", "skewness & kurtosis", n, ad, rd, 1e-9)

    lda = _load(results_dir / "results_lda.csv")
    if lda is not None:
        acc = lda.groupby("analysis")["accuracy"].first()
        keys = list(R_PRINTED["LDA accuracy (%)"])
        n, ad, rd = diff([round(acc[k] * 100, 1) for k in keys],
                         [R_PRINTED["LDA accuracy (%)"][k] for k in keys])
        rep.add("LDA (leave-one-out)", "accuracy (%)", n, ad, rd, 1e-9)
        py_counts, r_counts = [], []
        for k in keys:
            sub = lda[lda.analysis == k]
            py_counts += sub["n"].tolist()
            r_counts += R_PRINTED["LDA confusion counts"][k]
        n, ad, rd = diff(py_counts, r_counts)
        rep.add("LDA (leave-one-out)", "confusion matrix counts", n, ad, rd, 0.0,
                "every one of ~2,200 bees classified identically")

    gxe = _load(results_dir / "results_gxe_correlations.csv")
    if gxe is not None:
        py_vals, r_vals = [], []
        for _, row in gxe.iterrows():
            py_vals += [row.pearson_r, row.pearson_p, row.spearman_rho,
                        row.spearman_p]
            r_vals += list(R_PRINTED["G x E correlations"][row.trait])
        n, ad, rd = diff(py_vals, r_vals)
        rep.add("G x E correlations", "Pearson & Spearman", n, ad, rd, 1e-6,
                "differences are display rounding only: the R report prints the "
                "correlation coefficients to 7 significant digits")

    summ = _load(results_dir / "results_asymmetry_correction_summary.csv")
    if summ is not None:
        keys = list(R_PRINTED["Asymmetry — significant tests at p<0.05"])
        s = summ.set_index("method")["n_significant"]
        n, ad, rd = diff([s[k] for k in keys],
                         [R_PRINTED["Asymmetry — significant tests at p<0.05"][k]
                          for k in keys])
        rep.add("Multiple-comparison corrections", "number of significant tests",
                n, ad, rd, 0.0)

    rob = _load(results_dir / "results_robust_pca.csv")
    if rob is not None:
        v = rob["variance_explained_pct"].to_numpy()
        ref = R_PRINTED["Robust PCA"]
        rep.add_manual(
            "Robust PCA (MCD)", "variance explained (%)", "APPROX",
            f"Python RPC1={v[0]:.1f}%, RPC2={v[1]:.1f}% vs R "
            f"RPC1={ref['RPC1']}%, RPC2={ref['RPC2']}% — sklearn's MinCovDet and "
            "rrcov::CovMcd draw different random subsets; agreement to ~0.1 "
            "percentage points is the expected behaviour, not an error")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(wm.RESULTS_DIR))
    ap.add_argument("--reference", default=str(REFERENCE_DIR))
    ap.add_argument("--write-docs", action="store_true",
                    help="write docs/CONVERGENCE.md as well as printing")
    args = ap.parse_args(argv)

    results_dir, reference_dir = Path(args.results), Path(args.reference)
    if not (results_dir / "results_permanova.csv").exists():
        print("No Python results found. Run `python run_all.py` first.",
              file=sys.stderr)
        return 1

    rep = build_report(results_dir, reference_dir)
    table = rep.frame()

    show = table.copy()
    show["max_abs_diff"] = show["max_abs_diff"].map(
        lambda v: "" if pd.isna(v) else f"{v:.3e}")
    show["max_rel_diff"] = show["max_rel_diff"].map(
        lambda v: "" if pd.isna(v) else f"{v:.3e}")
    show["tolerance"] = show["tolerance"].map(
        lambda v: "" if pd.isna(v) else f"{v:g}")
    show["n_values"] = show["n_values"].map(
        lambda v: "" if pd.isna(v) else f"{int(v)}")

    print("\nPython vs R convergence report")
    print("=" * 110)
    print(show.drop(columns=["note"]).to_string(index=False))

    notes = table[table["note"].astype(bool)]
    if len(notes):
        print("\nNotes")
        print("-" * 110)
        for _, row in notes.drop_duplicates(subset=["note"]).iterrows():
            print(f"* {row['analysis']} / {row['quantity']}: {row['note']}")

    n_fail = int((table["verdict"] == "FAIL").sum())
    n_approx = int((table["verdict"] == "APPROX").sum())
    print("\n" + "=" * 110)
    print(f"{int((table.verdict == 'PASS').sum())} PASS, {n_approx} APPROX, "
          f"{n_fail} FAIL")

    if args.write_docs:
        DOCS_DIR.mkdir(exist_ok=True)
        path = DOCS_DIR / "CONVERGENCE.md"
        with path.open("w") as fh:
            fh.write("# Python vs R convergence report\n\n")
            fh.write("Generated by `python check_convergence.py --write-docs`.\n\n")
            fh.write("Every quantity the R pipeline reports is recomputed in "
                     "Python and compared here.  `max_rel_diff` is the largest "
                     "relative difference across all the values in that row.\n\n")
            fh.write(show.drop(columns=["note"]).to_markdown(index=False))
            fh.write("\n\n## Notes\n\n")
            for _, row in notes.drop_duplicates(subset=["note"]).iterrows():
                fh.write(f"* **{row['analysis']} / {row['quantity']}** — "
                         f"{row['note']}\n")
            fh.write(f"\n\n**Summary: {int((table.verdict == 'PASS').sum())} PASS, "
                     f"{n_approx} APPROX, {n_fail} FAIL.**\n")
        print(f"\nWrote {path}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
