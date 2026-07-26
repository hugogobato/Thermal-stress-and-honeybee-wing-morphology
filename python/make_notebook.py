#!/usr/bin/env python3
"""
Regenerate `notebooks/analysis_main.ipynb` from the cell definitions below.

The notebook is a thin, step-by-step wrapper around `wing_morphometry.py`:
every cell calls a function from that file, so the notebook and the script can
never drift apart.  Run this only if you want to change the notebook's
structure; ordinary users just open the notebook.
"""

import json
from pathlib import Path

import wing_morphometry as wm

NB_PATH = wm.PROJECT_ROOT / "notebooks" / "analysis_main.ipynb"

CELLS = [
    ("md", """# *Apis mellifera* wing morphometry — full analysis

**Phenotypic plasticity, canalization, and developmental stability of the *Apis mellifera* wing: effects of developmental temperature on sister broods**

Gruber CV, Polido M, Souto HG, Soares AEE, Araneda Duran X, Del Lama MA (2026)

---

This notebook runs every analysis in the paper, one step at a time, and shows
each figure as it is produced.

**How to use it:** click on the first cell and press <kbd>Shift</kbd>+<kbd>Enter</kbd>
repeatedly, or choose *Run → Run All Cells* from the menu. Nothing needs to be
edited. The whole notebook takes about a minute.

Every step calls a function from `python/wing_morphometry.py`, where the
statistical details and the comments live. If you want to know exactly what a
step does, open that file and read the function with the same name."""),

    ("code", """import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "python"))

import wing_morphometry as wm
wm.set_plot_style()

OUT = wm.RESULTS_DIR          # where tables and figures are written
print("Results will be written to:", OUT)"""),

    ("md", """## 1. Load the data

`data_clean.csv` has one row per measured wing. `df_complete` keeps only the
wings where all five vein measurements are present — that is the dataset every
analysis below uses."""),

    ("code", "df, df_complete, df_bp = wm.load_data()\ndf_complete.head(10)"),

    ("md", """## 2. Boxplots

Raw measurements, colony pair by colony pair. Blue is Coyhaique (cool
development), red is Ribeirão Preto (warm development)."""),

    ("code", "wm.fig_boxplot_media(df_bp, OUT);"),
    ("code", "wm.fig_boxplot_traits(df_complete, OUT);"),
    ("code", """# one standalone figure per trait, written to results/traits_plots/
wm.fig_boxplots_per_trait(df_complete, OUT);"""),

    ("md", """## 3. Principal Component Analysis

PCA compresses the five correlated vein measurements into a few uncorrelated
axes. PC1 is the overall "size" axis (all five loadings have the same sign);
the later components describe shape."""),

    ("code", "pca_res = wm.run_pca(df_complete, OUT)\nwm.fig_pca(pca_res, OUT);"),

    ("md", """## 4. MANOVA and follow-up ANOVAs

MANOVA asks whether the five traits *taken together* differ by origin, by wing
side, and whether the effect of side depends on origin (the interaction). The
follow-up ANOVAs then say which individual trait drives each effect, with
partial eta-squared (`np2`) as the effect size."""),

    ("code", "manova_table = wm.run_manova(df_complete, OUT)"),
    ("code", "anova_table = wm.run_anova_followup(df_complete, OUT)\nanova_table"),

    ("md", """## 5. Is the data multivariate normal?

MANOVA assumes multivariate normality. Mardia's test checks it. Skewness is
rejected here, which is why the PERMANOVA in the next step matters: it asks the
same question without that assumption."""),

    ("code", "wm.mardia_test(df_complete, OUT)"),

    ("md", """## 6. PERMANOVA

A permutation-based, assumption-free version of the MANOVA. It reshuffles the
bees 999 times to build the null distribution.

The equivalent R code (`vegan::adonis2`) takes about twenty minutes on this
dataset; this version takes under a second and gives the same numbers — see
`docs/CONVERGENCE.pdf`."""),

    ("code", "wm.run_permanova(df_complete, n_permutations=999, seed=42, outdir=OUT)"),

    ("md", """## 7. Linear Discriminant Analysis

How well can a bee's origin — or which side of its body a wing came from — be
predicted from the five measurements? Accuracy is leave-one-out
cross-validated, so it is an honest out-of-sample figure."""),

    ("code", "lda_res = wm.run_lda(df_complete, OUT)\nwm.fig_lda(lda_res, OUT);"),

    ("md", """## 8. Linear mixed-effects models

One model per trait. The random intercept for `pair_id` accounts for the fact
that the two colonies of a pair share a queen, so their workers are not
independent observations."""),

    ("code", "lme_table = wm.run_lme(df_complete, OUT)\nlme_table"),

    ("md", """## 9. Robust PCA and outliers

A sanity check: does the ordinary PCA reflect the bulk of the bees, or is it
being pulled around by a few extreme wings?"""),

    ("code", "rob = wm.run_robust_pca(df_complete, outdir=OUT)\nwm.fig_robust_pca(rob, OUT);"),

    ("md", """## 10. Asymmetry: right wing versus left wing

For every bee, the right and left measurements of the same wing vein are
compared. Consistent right-minus-left differences indicate directional
asymmetry, a classic index of developmental instability.

Because 120 tests are run, the p-values are also shown after three
multiple-comparison corrections."""),

    ("code", "asym = wm.run_asymmetry(df_complete, OUT)\nasym.head(12)"),
    ("code", "wm.fig_asymmetry_heatmaps(asym, OUT);"),
    ("code", "wm.fig_asymmetry_pvalue_comparison(asym, OUT);"),

    ("md", """## 11. Hierarchical clustering of colony means

Do colonies group by where they developed, or by which queen they came from?"""),

    ("code", "_, _, mahal = wm.fig_clustering(df_complete, OUT)\nmahal.round(2)"),

    ("md", """## 12. Reaction norms

One line per colony pair, joining its mean in Coyhaique to its mean in Ribeirão
Preto. Parallel lines mean every genotype responded the same way to the change
of environment; crossing lines are genotype × environment interaction."""),

    ("code", "wm.fig_reaction_norms(df_complete, OUT);"),

    ("md", """## 13. Genotype × Environment interaction

The same question again, as a correlation: does a colony that produces large
wings in Coyhaique also produce large wings in Ribeirão Preto?"""),

    ("code", "gxe_table, wide = wm.run_gxe(df_complete, OUT)\nwm.fig_gxe(gxe_table, wide, OUT);"),

    ("md", """## 14. Check against the published R results

This runs the same comparison as `python check_convergence.py` and prints the
PASS/FAIL table."""),

    ("code", """import check_convergence
check_convergence.main([])"""),
]


def build():
    cells = []
    for kind, source in CELLS:
        lines = source.splitlines(keepends=True)
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": lines})
        else:
            cells.append({"cell_type": "code", "execution_count": None,
                          "metadata": {}, "outputs": [], "source": lines})

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"Wrote {NB_PATH} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
