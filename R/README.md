# R pipeline

This is the implementation that produced the results reported in the paper. Its
output is archived in [`../reference_R_output/`](../reference_R_output).

## Running it

Open `analysis_main.Rmd` in RStudio and click **Knit**. The first chunk installs
any missing packages automatically. Knitting produces `analysis_main.html`, a
single report with every table and figure inline.

Figures and CSV files are written **into this `R/` folder**, next to the Rmd.
That is deliberate: it keeps a fresh run from overwriting the archived
`reference_R_output/`, which `python/check_convergence.py` compares against.

## The PERMANOVA is a separate script

`vegan::adonis2` builds the full 4,100 × 4,100 distance matrix and needs about
**twenty minutes** on this dataset, so the PERMANOVA chunk inside the Rmd is set
to `eval = FALSE`. Run it on its own when you want it:

```bash
cd R
Rscript permanova.R
```

It writes `permanova_log.txt` and `results_permanova.csv`.

The Python implementation in `../python/wing_morphometry.py` computes the same
quantities algebraically in under a second and agrees with this script to ten
significant digits — see [`../docs/CONVERGENCE.md`](../docs/CONVERGENCE.md).

## Packages used

`ggplot2`, `dplyr`, `tidyr`, `readr`, `stringr`, `purrr`, `tibble`, `forcats`,
`patchwork`, `RColorBrewer`, `car` (Type II/III ANOVA and MANOVA), `vegan`
(PERMANOVA), `MASS` (LDA), `lme4` (mixed models), `rrcov` (robust PCA),
`pheatmap` (clustered heatmaps), `ggrepel`.
