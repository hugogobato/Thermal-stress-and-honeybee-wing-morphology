# Archived R output

This is the output of `R/analysis_main.Rmd` and `R/permanova.R` exactly as it
was produced for the paper. **It is the reference the Python pipeline is checked
against**, so please do not overwrite it: `python/check_convergence.py` reads
the CSV files here.

* `results_*.csv` — the tables the R pipeline writes.
* `permanova_log.txt` — the printed `vegan::adonis2` output.
* `figures/` — every figure the R pipeline produces, at 300 dpi.

A handful of quantities are printed into the R HTML report rather than saved as
CSV (PCA loadings, MANOVA statistics, Mardia's test, LDA confusion matrices,
G×E correlations, robust-PCA variance). Those are transcribed into the
`R_PRINTED` dictionary at the top of `python/check_convergence.py` so that the
comparison covers them too.
