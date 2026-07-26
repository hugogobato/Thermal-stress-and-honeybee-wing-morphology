#!/usr/bin/env python3
"""
Run the complete analysis and write every table and figure to `results/`.

Usage
-----
    python run_all.py                 # everything (about one minute)
    python run_all.py --no-figures    # tables only, no plots
    python run_all.py --permutations 9999
    python run_all.py --outdir somewhere/else

Nothing in this script needs editing: it finds the data next to itself.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

import wing_morphometry as wm


def banner(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\n{n:>2}. {title}\n{'=' * 78}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(wm.RESULTS_DIR),
                    help="where to write tables and figures (default: results/)")
    ap.add_argument("--datadir", default=str(wm.DATA_DIR),
                    help="where to read the CSV files from (default: data/)")
    ap.add_argument("--permutations", type=int, default=999,
                    help="PERMANOVA permutations (default: 999, as in the paper)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    ap.add_argument("--no-figures", action="store_true",
                    help="compute the tables but skip the plots")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figures = not args.no_figures
    wm.set_plot_style()
    t0 = time.time()

    print("Apis mellifera wing morphometry — Python pipeline")
    print(f"data    : {Path(args.datadir).resolve()}")
    print(f"results : {outdir.resolve()}")

    banner(1, "Load and inspect the data")
    df, df_complete, df_bp = wm.load_data(args.datadir)

    if figures:
        banner(2, "Boxplots")
        wm.fig_boxplot_media(df_bp, outdir)
        wm.fig_boxplot_traits(df_complete, outdir)
        wm.fig_boxplots_per_trait(df_complete, outdir)
        plt.close("all")

    banner(3, "Principal Component Analysis")
    pca_res = wm.run_pca(df_complete, outdir)
    if figures:
        wm.fig_pca(pca_res, outdir)
        plt.close("all")

    banner(4, "MANOVA (Type III) and follow-up univariate ANOVAs")
    wm.run_manova(df_complete, outdir)
    wm.run_anova_followup(df_complete, outdir)

    banner(5, "Mardia's test of multivariate normality")
    wm.mardia_test(df_complete, outdir)

    banner(6, "PERMANOVA (two-way, Euclidean, distribution-free)")
    wm.run_permanova(df_complete, n_permutations=args.permutations,
                     seed=args.seed, outdir=outdir)

    banner(7, "Linear Discriminant Analysis with leave-one-out cross-validation")
    lda_res = wm.run_lda(df_complete, outdir)
    if figures:
        wm.fig_lda(lda_res, outdir)
        plt.close("all")

    banner(8, "Linear mixed-effects models")
    wm.run_lme(df_complete, outdir)

    banner(9, "Robust PCA and outlier diagnostics")
    rob = wm.run_robust_pca(df_complete, seed=args.seed, outdir=outdir)
    if figures:
        wm.fig_robust_pca(rob, outdir)
        plt.close("all")

    banner(10, "Asymmetry tests (paired t-test and Wilcoxon)")
    asym = wm.run_asymmetry(df_complete, outdir)
    if figures:
        wm.fig_asymmetry_heatmaps(asym, outdir)
        wm.fig_asymmetry_pvalue_comparison(asym, outdir)
        plt.close("all")

    if figures:
        banner(11, "Hierarchical clustering of colony means")
        wm.fig_clustering(df_complete, outdir)
        plt.close("all")

        banner(12, "Reaction norms")
        wm.fig_reaction_norms(df_complete, outdir)
        plt.close("all")

    banner(13, "Genotype x Environment interaction")
    gxe_table, wide = wm.run_gxe(df_complete, outdir)
    if figures:
        wm.fig_gxe(gxe_table, wide, outdir)
        plt.close("all")

    print(f"\n{'=' * 78}")
    print(f"Finished in {time.time() - t0:.1f} s. Everything is in {outdir.resolve()}")
    print("Next step: `python check_convergence.py` compares these results with "
          "the published R output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
