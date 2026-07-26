# Thermal stress and honeybee wing morphology

Data and complete analysis code for:

> **Phenotypic plasticity, canalization, and developmental stability of the *Apis mellifera* wing: effects of developmental temperature on sister broods**
>
> Caroline Vivian Gruber<sup>1,¥</sup>, Mariah Polido<sup>1,¥</sup>, Hugo Gobato Souto<sup>2</sup>, Ademilson Espencer E. Soares<sup>3</sup>, Ximena Araneda Duran<sup>4</sup>, Marco Antonio Del Lama<sup>1</sup>
>
> <sup>¥</sup> shared first authors
>
> <sup>1</sup> Departamento de Genética e Evolução, Universidade Federal de São Carlos, São Carlos, Brazil
> <sup>2</sup> Dell, Austin — TX, USA
> <sup>3</sup> Departamento de Genética, Faculdade de Medicina de Ribeirão Preto, Universidade de São Paulo, Brazil
> <sup>4</sup> *(affiliation to be completed)*

| Author | ORCID |
|---|---|
| Caroline Vivian Gruber | — |
| Mariah Polido | [0000-0001-9965-5127](https://orcid.org/0000-0001-9965-5127) |
| Hugo Gobato Souto | [0000-0002-7039-0572](https://orcid.org/0000-0002-7039-0572) |
| Ademilson Espencer E. Soares | — |
| Ximena Araneda Duran | [0000-0002-7876-6909](https://orcid.org/0000-0002-7876-6909) |
| Marco Antonio Del Lama | [0000-0002-3329-8953](https://orcid.org/0000-0002-3329-8953) |

---

## What this study asked

Twelve honeybee queens each produced two broods of daughters: one reared in
**Coyhaique, Chile** (cool) and one in **Ribeirão Preto, Brazil** (warm).
Because the two broods of a pair share a mother, the workers are sisters, and
any difference between them is caused by where they developed rather than by
who their mother was.

On each worker we measured five linear distances on the wing venation
(**M1–M5**), on both the right and the left forewing. Three questions follow:

1. **Plasticity** — does developmental temperature change wing shape and size?
2. **Canalization** — do some traits resist that change more than others?
3. **Developmental stability** — does thermal stress make the right and left
   wings of the same bee less alike (asymmetry)?

The dataset is **4,100 wings** from **2,065 bees** across **24 colonies**
(12 matched pairs).

---

## What is in this repository

```
├── data/                    the two data files, plus a data dictionary
├── python/                  the Python analysis  ← recommended
├── notebooks/               the same analysis as a Jupyter notebook
├── R/                       the original R analysis
├── results/                 output of the Python pipeline (tables + figures)
├── reference_R_output/      output of the R pipeline, as published
└── docs/CONVERGENCE.pdf     proof that the two agree, number by number
```

The **R pipeline produced the published results**. The **Python pipeline is an
independent reimplementation** written to check them: it was written against
the R output, and every quantity is compared automatically. See
[Do the two agree?](#do-the-two-agree) below — the answer is yes.

---

## Running the analysis

You do not need to know how to program. Pick whichever language you are more
comfortable with; both produce the same numbers.

### Option A — Python, one command (recommended)

**1. Install Python.** If you do not already have it, download it from
[python.org/downloads](https://www.python.org/downloads/) (any version 3.10 or
newer) and — on Windows — tick **"Add Python to PATH"** during installation.

**2. Download this repository.** Click the green **Code** button at the top of
this page → **Download ZIP**, then unzip it. (Or, if you use git:
`git clone https://github.com/hugogobato/Thermal-stress-and-honeybee-wing-morphology.git`)

**3. Open a terminal** in the unzipped folder.
On Windows: open the folder in File Explorer, type `cmd` in the address bar and
press Enter. On macOS: right-click the folder → *Services* → *New Terminal at
Folder*.

**4. Install the required packages** (once):

```bash
pip install -r requirements.txt
```

**5. Run everything:**

```bash
cd python
python run_all.py
```

That takes about a minute and writes every table and figure into `results/`.

**6. Check the results against the published R output:**

```bash
python check_convergence.py
```

This prints a PASS/FAIL table comparing all 38 sets of numbers. The comparison
is already typeset in [`docs/CONVERGENCE.pdf`](docs/CONVERGENCE.pdf); if you
want to regenerate that PDF, add `--write-docs`. That step is the only one in
the repository that needs LaTeX installed (TeX Live, MacTeX or MiKTeX) — the
comparison itself prints to the screen without it.

### Option B — Python, step by step in a notebook

If you prefer to see each analysis and its figure one at a time, open
`notebooks/analysis_main.ipynb`:

```bash
pip install -r requirements.txt
jupyter lab notebooks/analysis_main.ipynb
```

Then choose **Run → Run All Cells**, or press <kbd>Shift</kbd>+<kbd>Enter</kbd>
through the notebook. Every cell is explained in plain language.

### Option C — R, in RStudio

1. Open `R/analysis_main.Rmd` in RStudio.
2. Click **Knit**. The first run installs any missing packages automatically
   and produces an HTML report with every table and figure inline.
3. The PERMANOVA chunk is deliberately switched off inside the report because
   `vegan::adonis2` needs about **twenty minutes** on this dataset. Run it
   separately when you want it:

```bash
Rscript R/permanova.R
```

> R writes its figures into the folder you run it from. The published R output
> is preserved in `reference_R_output/` — do not overwrite it if you want the
> comparison script to keep working.

---

## The analyses, in plain language

| # | Analysis | The question it answers | Main output |
|---|---|---|---|
| 1 | **Boxplots** | What do the raw measurements look like, colony by colony? | `plot_boxplot_media.jpg`, `plot_boxplot_traits.jpg` |
| 2 | **PCA** | Can five correlated measurements be summarised by a couple of axes? Do Coyhaique and Ribeirão Preto bees sit in different places on them? | `pca_plots/` |
| 3 | **MANOVA + follow-up ANOVAs** | Do the five traits *together* differ by origin, by wing side, and does side behave differently in the two places? Which individual trait drives it, and how strongly (partial η²)? | `results_manova.csv`, `results_manova_followup.csv` |
| 4 | **Mardia's test** | Are the data multivariate normal — i.e. is the MANOVA entitled to its p-values? | `results_mardia.csv` |
| 5 | **PERMANOVA** | The same question as the MANOVA, but making no distributional assumption: the bees are reshuffled 999 times to build the null distribution by brute force. | `results_permanova.csv` |
| 6 | **LDA** | How accurately can a bee's origin — or which side of the body a wing came from — be predicted from its five measurements? Accuracy is leave-one-out cross-validated, so it is honest out-of-sample performance. | `results_lda.csv`, `plot_lda.jpg` |
| 7 | **Mixed models** | Effect sizes per trait, while accounting for the fact that sisters within a colony pair are not independent observations. | `results_lme.csv` |
| 8 | **Robust PCA** | Is the separation seen in the ordinary PCA real, or created by a handful of extreme wings? | `plot_robust_pca.jpg` |
| 9 | **Asymmetry tests** | Within each bee, is the right wing consistently different from the left? Tested per colony and per trait, with three multiple-comparison corrections. | `results_table_S3_wilcoxon.csv`, `plot_asymmetry_heatmap*.jpg` |
| 10 | **Clustering** | Do colonies group by where they developed, or by which queen they came from? | `plot_cluster_heatmap*.jpg` |
| 11 | **Reaction norms & G×E** | Did every genotype respond to the change of environment in the same way? Parallel lines mean no genotype × environment interaction; crossing lines mean there is one. | `plot_reaction_norms.jpg`, `plot_gxe_scatter.jpg` |

### Reading the asymmetry heatmaps

Each cell is one colony × one trait. The number is the mean **right minus left**
difference in millimetres — **red** means the right wing is larger, **blue**
means the left wing is larger, white means no difference. Stars mark
significance (`*` p<0.05, `**` p<0.01, `***` p<0.001).

Because 120 tests are run at once, some will look significant by chance alone.
Four versions of the heatmap are therefore produced — raw p-values, and after
Bonferroni, Holm-Bonferroni and Benjamini-Hochberg correction — so you can see
exactly which findings survive. `plot_asymmetry_pvalue_comparison.jpg` shows all
120 tests side by side under each correction.

---

## Do the two agree?

Yes. `python check_convergence.py` compares every quantity the R pipeline
reports against the Python recomputation and prints a PASS/FAIL table. The
typeset version of that table is
[`docs/CONVERGENCE.pdf`](docs/CONVERGENCE.pdf) — open it for the full
comparison. The current status is **37 PASS, 1 APPROX, 0 FAIL**. Highlights:

| Quantity | Agreement |
|---|---|
| PCA variance explained and all 25 loadings | exact to every printed digit |
| MANOVA Pillai / Wilks / Hotelling-Lawley / Roy | exact to every printed digit |
| Follow-up ANOVA sums of squares, F, p, partial η² | relative difference < 1e-12 |
| PERMANOVA sums of squares, R², F | relative difference < 1e-13 |
| Mardia's skewness and kurtosis | exact to every printed digit |
| LDA leave-one-out confusion matrices | **every one of ~2,200 bees classified identically** |
| All 120 asymmetry tests (t, Wilcoxon, and all corrections) | relative difference < 1e-12 |
| Mixed-model fixed effects, AIC, BIC | relative difference < 1e-6 |
| G×E Pearson and Spearman correlations | agree to the last digit R prints |

**The one exception**, marked APPROX rather than PASS: the **robust PCA**.
`sklearn.covariance.MinCovDet` and R's `rrcov::CovMcd` both implement the
Minimum Covariance Determinant estimator, but they draw different random
subsets internally. Python reports RPC1 = 54.7% / RPC2 = 21.1% and 126 outliers;
R reports 54.6% / 21.1% and 122 outliers, out of 4,100 wings. The two are
qualitatively identical and nothing in the paper depends on the difference.
This is inherent to the method, not a bug in either implementation.

### Where the two languages needed care

Making two statistics packages agree is not automatic. Three defaults had to be
matched deliberately; each is flagged with a `# MATCH-R:` comment in
`python/wing_morphometry.py`:

1. **Standardisation** uses the *population* standard deviation (divide by *n*),
   not the sample one (*n*−1). The R script has a `scale_pop()` helper for this.
2. **`MASS::lda(CV = TRUE)`** holds the class priors fixed at the full-sample
   proportions across all leave-one-out folds. Re-estimating them inside each
   fold — what a naive scikit-learn loop does — reclassifies one or two bees that
   sit exactly on the decision boundary. With the priors handled correctly, the
   confusion matrices match to the individual bee.
3. **Type III MANOVA** is only meaningful with sum-to-zero contrasts, hence
   `C(origin, Sum) * C(side, Sum)` in the Python formula. With the default
   treatment coding, the "origin" row would silently test something else.

### A speed note on the PERMANOVA

The R implementation builds the full 4,100 × 4,100 distance matrix and takes
about **twenty minutes**. With Euclidean distances the sums of squares are
algebraically identical to the classical multivariate Type I sums of squares, so
the Python version obtains them by projecting the centred data onto an
orthonormal basis of each model term. It finishes in **under a second** and
agrees with R to ten significant digits.

---

## Two corrections relative to an earlier Python draft

An earlier, superseded Python notebook of this analysis contained two
methodological problems. Both were fixed in the R pipeline and both fixes are
carried through here:

1. **PERMANOVA design.** The old notebook ran three separate one-way PERMANOVAs
   — one for origin, one for side, and one on a concatenated `origin_side`
   factor. That is not a two-way design and cannot test the interaction. Both
   pipelines now fit a proper two-way model, `Y ~ origin * side`.
2. **Pairing in the asymmetry tests.** The old notebook truncated the right and
   left measurement arrays to equal length *by row position*, which could
   compare one bee's right wing with a different bee's left wing. Both
   pipelines now match right and left on the bee identifier `amost` with an
   inner join inside each colony, so every paired test operates on the same
   bee's two wings.

---

## Frequently asked

**Do I need R at all?** No. `reference_R_output/` already contains everything
the R pipeline produced, so the convergence check works with Python alone.

**`pip: command not found` / `python: command not found`.** Try `pip3` and
`python3`. On Windows, re-run the Python installer and make sure *"Add Python to
PATH"* is ticked.

**Can I run it on my own data?** Yes, if your file has the same columns as
`data/data_clean.csv` (see [`data/README.md`](data/README.md)). Point the script
at it with `python run_all.py --datadir /path/to/your/folder`.

**Will running it overwrite the published results?** No. Python writes to
`results/`; the R output lives untouched in `reference_R_output/`.

**How do I get more permutations?** `python run_all.py --permutations 9999`.
It stays fast.

---

## Citing

If you use this code or data, please cite the paper. A machine-readable
citation is in [`CITATION.cff`](CITATION.cff); GitHub renders it as a
*"Cite this repository"* button in the sidebar.

## Licence

Code is released under the [MIT Licence](LICENSE). The data are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — reuse freely with
attribution to the paper.

## Contact

Questions about the code: Hugo Gobato Souto —
[hugogobatosouto@gmail.com](mailto:hugogobatosouto@gmail.com).
