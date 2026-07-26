# Results index

Everything in this folder is produced by `python/run_all.py`. It is committed to
the repository so the outputs can be inspected without running anything; re-running
the pipeline simply overwrites it.

The published R equivalents live in [`../reference_R_output/`](../reference_R_output),
and [`../docs/CONVERGENCE.pdf`](../docs/CONVERGENCE.pdf) compares the two number by
number.

## Tables

| File | What it contains |
|---|---|
| `results_pca_variance.csv` | Percentage of variance explained by each principal component. |
| `results_pca_loadings.csv` | How much each vein trait contributes to each component. |
| `results_manova.csv` | Two-way MANOVA (Type III), all four test statistics: Pillai, Wilks, Hotelling-Lawley, Roy. |
| `results_manova_followup.csv` | One Type II ANOVA per trait, with partial η² (`np2`) as the effect size. |
| `results_mardia.csv` | Mardia's multivariate skewness and kurtosis test. |
| `results_permanova.csv` | Two-way PERMANOVA: sums of squares, R², F and permutation p-values. |
| `results_lda.csv` | Leave-one-out confusion matrices and accuracies for the three discriminant analyses. |
| `results_lme.csv` | Mixed-model fixed effects per trait: estimate, standard error, Wald z, p, AIC, BIC. |
| `results_robust_pca.csv` | Variance explained by the robust (MCD-based) components. |
| `results_table_S3_wilcoxon.csv` | **Table S3.** All 120 colony × trait paired tests: mean right−left difference, number of bee pairs, t and Wilcoxon statistics, and p-values raw and after Bonferroni, Holm-Bonferroni and Benjamini-Hochberg correction. |
| `results_asymmetry_correction_summary.csv` | How many of the 120 tests stay significant under each correction. |
| `results_colony_mahalanobis.csv` | Pairwise Mahalanobis distances between the 24 colony means. |
| `results_gxe_correlations.csv` | Pearson and Spearman correlations between each colony pair's mean in the two localities. |

## Figures

| File | What it shows |
|---|---|
| `plot_boxplot_media.jpg` | Mean vein length per colony pair, right wing, coloured by origin. |
| `plot_boxplot_traits.jpg` | The same, one panel per trait (M1–M5). |
| `traits_plots/boxplot_nerv0*.jpg` | Each of those panels as a standalone figure. |
| `pca_plots/scree_plot.jpg` | Variance explained by each principal component. |
| `pca_plots/pca_origin.jpg` | PC1 vs PC2, coloured by locality, with 95% ellipses. |
| `pca_plots/pca_side.jpg` | PC1 vs PC2, coloured by wing side. |
| `pca_plots/pca_colony_pair.jpg` | PC1 vs PC2, coloured by colony pair (● Coyhaique, ▲ Ribeirão Preto). |
| `pca_plots/plot_pca_combined.jpg` | All four PCA panels in one figure. |
| `plot_lda.jpg` | LD1 score distributions for origin, and for wing side within each locality, annotated with cross-validated accuracy. |
| `plot_robust_pca.jpg` | Robust PCA scores, sorted Mahalanobis distances with the outlier cutoff, and a χ² Q-Q plot. |
| `plot_asymmetry_heatmap.jpg` | **Right − left differences** by colony and trait, with significance stars from the raw Wilcoxon p-values. Red = right wing larger, blue = left wing larger. |
| `plot_asymmetry_heatmap_raw.jpg` and `_bonferroni` / `_holm_bonferroni` / `_benjamini_hochberg_bh` | The same heatmap under each multiple-comparison correction, with the adjusted p-value printed in every cell. |
| `plot_asymmetry_heatmap_corrections.jpg` | All four corrections side by side. |
| `plot_asymmetry_pvalue_comparison.jpg` | All 120 tests as −log₁₀(p) bars under each correction, with the p = 0.05 line. |
| `plot_cluster_heatmap.jpg` | Colony means clustered by Euclidean distance (Ward linkage), with an origin annotation bar. |
| `plot_cluster_heatmap_mahalanobis.jpg` | The same, clustered by Mahalanobis distance instead. |
| `plot_reaction_norms.jpg` | One line per colony pair joining its mean in Coyhaique to its mean in Ribeirão Preto. Crossing lines indicate genotype × environment interaction. |
| `plot_gxe_scatter.jpg` | Coyhaique mean vs Ribeirão Preto mean per colony pair, with the 1:1 line (dashed) and the fitted line (red). |
