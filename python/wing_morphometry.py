"""
Apis mellifera wing morphometry — Python analysis library
=========================================================

Paper
-----
Phenotypic plasticity, canalization, and developmental stability of the
*Apis mellifera* wing: effects of developmental temperature on sister broods.
Gruber CV, Polido M, Souto HG, Soares AEE, Araneda Duran X, Del Lama MA (2026).

What this file is
-----------------
Every statistical analysis and every figure in the paper, written as small,
independent functions.  Nothing here runs on import — you call the functions
you want.  The usual way to run everything at once is:

    python run_all.py

Each function is documented with (a) what it does in plain words, (b) which
figure/table of the paper it produces, and (c) the R function it mirrors, so
that the Python and R pipelines can be checked against each other line by line.

Design rule followed throughout
-------------------------------
The R pipeline (`R/analysis_main.Rmd`) is the reference implementation used for
the published results.  Wherever R and Python have different *defaults* for the
same statistic, this file explicitly reproduces R's choice.  The three places
where that matters are flagged in the code with `# MATCH-R:` comments:

  1. Standardisation uses the population SD (divide by n), matching
     `sklearn.StandardScaler`; the R code has a `scale_pop()` helper for this.
  2. `MASS::lda(CV = TRUE)` keeps the class priors fixed at the full-sample
     proportions across all leave-one-out folds.  Re-estimating them inside
     each fold (what a naive scikit-learn loop does) changes 1-2 borderline
     bees out of ~2,200.
  3. MANOVA uses Type III sums of squares, which are only meaningful with
     sum-to-zero contrasts, hence `C(origin, Sum) * C(side, Sum)`.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
try:                       # inside Jupyter/IPython, keep the interactive backend
    __IPYTHON__            # noqa: F821
except NameError:          # plain script: render to files without needing a screen
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multitest import multipletests


# ═════════════════════════════════════════════════════════════════════════════
# Constants — colours, labels, paths
# ═════════════════════════════════════════════════════════════════════════════

#: The five linear wing-vein measurements (millimetres).
VEIN_COLS = ["nerv01", "nerv02", "nerv03", "nerv04", "nerv05"]

#: Short names used in the figures of the paper.
VEIN_LABELS = {"nerv01": "M1", "nerv02": "M2", "nerv03": "M3",
               "nerv04": "M4", "nerv05": "M5"}

#: `origin` is stored in the data with the historical codes below.
ORIGIN_LABELS = {"Chile": "Coyhaique", "Ribeirao_Preto": "Ribeirão Preto"}

ORIGIN_PAL = {"Coyhaique": "#2166AC", "Ribeirão Preto": "#B2182B"}
SIDE_PAL = {"right": "#1B7837", "left": "#762A83"}

#: ColorBrewer "Set3" (12 classes) — one colour per colony pair.
PAIR_PAL = ["#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3", "#FDB462",
            "#B3DE69", "#FCCDE5", "#D9D9D9", "#BC80BD", "#CCEBC5", "#FFED6F"]

#: ColorBrewer "RdBu" (11 classes); reversed for heatmaps (blue = low, red = high).
RDBU_11 = ["#67001F", "#B2182B", "#D6604D", "#F4A582", "#FDDBC7", "#F7F7F7",
           "#D1E5F0", "#92C5DE", "#4393C3", "#2166AC", "#053061"]

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DPI = 300


def _relabel_colony(code: str) -> str:
    """`CH07` -> `COY07`, `RP07` -> `RPR07` (the labels used in the paper)."""
    if code.startswith("CH"):
        return "COY" + code[2:]
    if code.startswith("RP"):
        return "RPR" + code[2:]
    return code


# ═════════════════════════════════════════════════════════════════════════════
# Plot style — a matplotlib approximation of ggplot2's theme_bw()
# ═════════════════════════════════════════════════════════════════════════════

def set_plot_style() -> None:
    """Apply the shared figure style. Call once, before making any figure."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": DPI,
        "figure.dpi": 110,
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#EBEBEB",
        "grid.linewidth": 0.8,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "font.size": 11,
        "legend.frameon": True,
        "legend.edgecolor": "#CCCCCC",
        "legend.fontsize": 9,
        "legend.title_fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def _save(fig, path, dpi: int = DPI) -> Path:
    """Save a figure as JPEG and report the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  saved: {path.relative_to(PROJECT_ROOT)}")
    return path


def _confidence_ellipse(ax, x, y, level: float = 0.95, **kwargs):
    """
    Draw the same 95% ellipse that ggplot2's `stat_ellipse(type = "norm")` draws.

    ggplot2 uses an F-distribution radius (not chi-squared) with denominator
    degrees of freedom n - 1, applied to the Cholesky factor of the covariance
    matrix.  Reproducing it exactly keeps the Python and R PCA panels
    superimposable.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 3:
        return
    cov = np.cov(np.vstack([x, y]))
    radius = np.sqrt(2 * stats.f.ppf(level, 2, n - 1))
    chol = np.linalg.cholesky(cov).T  # upper triangular, as R's chol()
    theta = np.linspace(0, 2 * np.pi, 201)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    pts = circle @ chol * radius + np.array([x.mean(), y.mean()])
    ax.plot(pts[:, 0], pts[:, 1], **kwargs)


def _bw_nrd0(x) -> float:
    """R's default kernel-density bandwidth (`bw.nrd0`), so density curves match."""
    x = np.asarray(x, float)
    n = len(x)
    sd = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    lo = min(sd, iqr / 1.349) or sd or abs(x[0]) or 1.0
    return 0.9 * lo * n ** (-0.2)


def _kde_curve(x, n_grid: int = 512, cut: float = 3.0):
    """Gaussian KDE evaluated on a grid, using R's bandwidth rule."""
    x = np.asarray(x, float)
    bw = _bw_nrd0(x)
    kde = stats.gaussian_kde(x, bw_method=bw / np.std(x, ddof=1))
    grid = np.linspace(x.min() - cut * bw, x.max() + cut * bw, n_grid)
    return grid, kde(grid)


def scale_pop(X) -> np.ndarray:
    """
    Standardise columns to mean 0 / SD 1 using the *population* SD (divide by n).

    MATCH-R: this is what `sklearn.StandardScaler` does and what the R helper
    `scale_pop()` reproduces; base R's `scale()` uses the sample SD (n - 1) and
    would shift the PCA percentages in the third decimal.
    """
    return StandardScaler().fit_transform(np.asarray(X, float))


# ═════════════════════════════════════════════════════════════════════════════
# 1. Load the data
# ═════════════════════════════════════════════════════════════════════════════

def load_data(data_dir=DATA_DIR, verbose: bool = True):
    """
    Read the two data files and return `(df, df_complete, df_boxplot)`.

    * `df`           — every measured wing (4,130 rows).
    * `df_complete`  — only wings with all five vein measurements (4,100 rows).
                       This is the analysis dataset used everywhere below.
    * `df_boxplot`   — one row per bee with the mean of the five veins,
                       used for the colony-pair boxplot (Figure 1).
    """
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "data_clean.csv")
    df_bp = pd.read_csv(data_dir / "data_boxplot_media_long.csv")

    df["origin"] = pd.Categorical(df["origin"], categories=["Chile", "Ribeirao_Preto"])
    df["side"] = pd.Categorical(df["side"], categories=["right", "left"])
    df["pair_id"] = df["pair_id"].astype(str).str.zfill(2)
    df["origin_label"] = df["origin"].map(ORIGIN_LABELS).astype(str)
    df["colony"] = df["colony_label"].map(_relabel_colony)

    df_complete = df.dropna(subset=VEIN_COLS).copy()

    if verbose:
        print(f"  all wings measured : {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"  complete cases     : {df_complete.shape[0]} rows")
        print("  missing values per vein:")
        for c in VEIN_COLS:
            print(f"    {c}: {int(df[c].isna().sum())}")
        print("  group sizes (origin x side):")
        for (o, s), k in df_complete.groupby(["origin", "side"], observed=True).size().items():
            print(f"    {o:15s} {s:6s} {k}")
    return df, df_complete, df_bp


# ═════════════════════════════════════════════════════════════════════════════
# 2. Boxplots  (Figure 1 and Figure S-traits)
# ═════════════════════════════════════════════════════════════════════════════

def fig_boxplot_media(df_bp, outdir=RESULTS_DIR):
    """Mean vein length per colony pair, right wing — `plot_boxplot_media.jpg`."""
    d = df_bp.copy()
    d["origin_label"] = d["colony_code"].str[:2].map(
        {"CH": "Coyhaique", "RP": "Ribeirão Preto"})
    d["colony_num"] = d["colony_code"].str[2:]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=d, x="colony_num", y="mean_nerv", hue="origin_label",
                hue_order=["Coyhaique", "Ribeirão Preto"], palette=ORIGIN_PAL,
                width=0.65, linewidth=0.8, fliersize=2.0,
                flierprops={"alpha": 0.4}, ax=ax)
    ax.set_xlabel("Colony Pair")
    ax.set_ylabel("Mean Vein Length (nerv01–05, mm)")
    ax.set_title("Wing Vein Measurements by Colony Pair and Origin (Right Wing)")
    ax.legend(title="Origin", loc="best")
    ax.xaxis.grid(False)
    fig.tight_layout()
    return _save(fig, Path(outdir) / "plot_boxplot_media.jpg")


def _right_wing_long(df_complete):
    """Right-wing measurements reshaped to one row per (bee, trait)."""
    d = df_complete[df_complete["side"] == "right"]
    long = d.melt(id_vars=["pair_id", "origin", "origin_label", "colony_label"],
                  value_vars=VEIN_COLS, var_name="trait", value_name="value")
    long["trait"] = long["trait"].map(VEIN_LABELS)
    return long


def fig_boxplot_traits(df_complete, outdir=RESULTS_DIR):
    """One panel per vein trait, right wing — `plot_boxplot_traits.jpg`."""
    long = _right_wing_long(df_complete)
    labels = [VEIN_LABELS[c] for c in VEIN_COLS]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.ravel()
    for ax, tr in zip(axes, labels):
        sns.boxplot(data=long[long.trait == tr], x="pair_id", y="value",
                    hue="origin_label", hue_order=["Coyhaique", "Ribeirão Preto"],
                    palette=ORIGIN_PAL, width=0.6, linewidth=0.7, fliersize=1.5,
                    flierprops={"alpha": 0.3}, ax=ax)
        ax.set_title(tr)
        ax.set_xlabel("Colony Pair")
        ax.set_ylabel("Measurement (mm)")
        ax.get_legend().remove()
    axes[-1].axis("off")
    handles = [Patch(facecolor=ORIGIN_PAL[k], label=k) for k in ORIGIN_PAL]
    axes[-1].legend(handles=handles, title="Origin", loc="center", frameon=True)
    fig.suptitle("Vein Measurements by Colony Pair and Origin (Right Wing)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, Path(outdir) / "plot_boxplot_traits.jpg")


def fig_boxplots_per_trait(df_complete, outdir=RESULTS_DIR):
    """One standalone figure per trait, written to `results/traits_plots/`."""
    long = _right_wing_long(df_complete)
    paths = []
    for col in VEIN_COLS:
        tr = VEIN_LABELS[col]
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=long[long.trait == tr], x="pair_id", y="value",
                    hue="origin_label", hue_order=["Coyhaique", "Ribeirão Preto"],
                    palette=ORIGIN_PAL, width=0.6, linewidth=0.7, fliersize=1.5,
                    flierprops={"alpha": 0.3}, ax=ax)
        ax.set_xlabel("Colony Pair")
        ax.set_ylabel("Measurement (mm)")
        ax.set_title(tr)
        ax.legend(title="Origin", loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        paths.append(_save(fig, Path(outdir) / "traits_plots" / f"boxplot_{col}.jpg"))
        plt.close(fig)
    return paths


# ═════════════════════════════════════════════════════════════════════════════
# 3. Principal Component Analysis  (Figure 2)
# ═════════════════════════════════════════════════════════════════════════════

def run_pca(df_complete, outdir=RESULTS_DIR, verbose: bool = True):
    """
    PCA of the five vein traits on standardised data.

    Returns a dict with the variance explained, the loadings table and a
    dataframe of PC scores joined back to the design columns.
    Mirrors R's `prcomp()` on `scale_pop()`-standardised data; PC signs are
    aligned the same way in both languages (largest loading forced positive).
    """
    X = scale_pop(df_complete[VEIN_COLS])
    pca = PCA(n_components=5).fit(X)
    scores = pca.transform(X)

    var_exp = pca.explained_variance_ratio_ * 100
    loadings = pd.DataFrame(pca.components_.T, index=VEIN_COLS,
                            columns=[f"PC{i+1}" for i in range(5)])

    pca_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(5)],
                          index=df_complete.index)
    for c in ["origin", "side", "pair_id", "colony_label", "origin_label"]:
        pca_df[c] = df_complete[c].values

    if verbose:
        print("  variance explained per PC (%):",
              ", ".join(f"{v:.2f}" for v in var_exp))
        print(loadings.round(4).to_string())

    loadings.to_csv(Path(outdir) / "results_pca_loadings.csv")
    pd.DataFrame({"PC": range(1, 6), "variance_explained_pct": var_exp}).to_csv(
        Path(outdir) / "results_pca_variance.csv", index=False)
    return {"pca": pca, "var_exp": var_exp, "loadings": loadings, "scores": pca_df}


def _panel_scree(ax, var_exp):
    ax.bar(range(1, 6), var_exp, color="#4393C3", edgecolor="#2166AC",
           linewidth=0.8, width=0.7, zorder=2)
    ax.plot(range(1, 6), var_exp, "-o", color="#2166AC", linewidth=1.6,
            markersize=6, zorder=3)
    for i, v in enumerate(var_exp, start=1):
        ax.text(i, v + max(var_exp) * 0.03, f"{v:.1f}%", ha="center",
                fontsize=9, fontweight="bold")
    ax.set_xticks(range(1, 6))
    ax.set_ylim(0, max(var_exp) * 1.15)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("Scree Plot")


def _panel_pca_by(ax, pca_df, var_exp, column, palette, legend_title, title):
    for key, colour in palette.items():
        m = pca_df[column] == key
        if not m.any():
            continue
        ax.scatter(pca_df.loc[m, "PC1"], pca_df.loc[m, "PC2"], s=6, alpha=0.3,
                   color=colour, label=key, linewidths=0)
        _confidence_ellipse(ax, pca_df.loc[m, "PC1"], pca_df.loc[m, "PC2"],
                            color=colour, linestyle="--", linewidth=1.8)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    ax.set_title(title)
    handles = [Line2D([], [], marker="o", linestyle="", color=c, label=k)
               for k, c in palette.items()]
    ax.legend(handles=handles, title=legend_title, loc="best")


def _panel_pca_pairs(ax, pca_df, var_exp):
    pair_ids = sorted(pca_df["pair_id"].unique())
    colours = dict(zip(pair_ids, PAIR_PAL))
    markers = {"Chile": "o", "Ribeirao_Preto": "^"}
    for pid in pair_ids:
        for orig, mk in markers.items():
            m = (pca_df["pair_id"] == pid) & (pca_df["origin"] == orig)
            if not m.any():
                continue
            ax.scatter(pca_df.loc[m, "PC1"], pca_df.loc[m, "PC2"], s=14,
                       alpha=0.5, color=colours[pid], marker=mk,
                       linewidths=0.2, edgecolors="#555555")
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    ax.set_title("PCA — by Colony Pair (● Coyhaique, ▲ RPR)")
    handles = [Line2D([], [], marker="o", linestyle="", color=colours[p],
                      markeredgecolor="#555555", label=p) for p in pair_ids]
    ax.legend(handles=handles, title="Colony Pair", ncol=2, fontsize=7,
              loc="best")


def fig_pca(pca_res, outdir=RESULTS_DIR):
    """The four PCA panels, saved individually and as one combined figure."""
    pca_df, var_exp = pca_res["scores"], pca_res["var_exp"]
    outdir = Path(outdir) / "pca_plots"

    builders = {
        "scree_plot": lambda ax: _panel_scree(ax, var_exp),
        "pca_origin": lambda ax: _panel_pca_by(
            ax, pca_df, var_exp, "origin_label", ORIGIN_PAL, "Origin",
            "PCA — by Origin"),
        "pca_side": lambda ax: _panel_pca_by(
            ax, pca_df, var_exp, "side", SIDE_PAL, "Side", "PCA — by Wing Side"),
        "pca_colony_pair": lambda ax: _panel_pca_pairs(ax, pca_df, var_exp),
    }
    for name, build in builders.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        build(ax)
        fig.tight_layout()
        _save(fig, outdir / f"{name}.jpg")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for ax, build in zip(axes.ravel(), builders.values()):
        build(ax)
    fig.suptitle("Principal Component Analysis — Wing Vein Measurements",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, outdir / "plot_pca_combined.jpg")


# ═════════════════════════════════════════════════════════════════════════════
# 4. MANOVA and follow-up univariate ANOVAs  (Table 1)
# ═════════════════════════════════════════════════════════════════════════════

def run_manova(df_complete, outdir=RESULTS_DIR, verbose: bool = True):
    """
    Two-way MANOVA of the five traits on `origin * side`.

    MATCH-R: Type III sums of squares require sum-to-zero contrasts, hence
    `C(origin, Sum) * C(side, Sum)`.  With the default treatment coding the
    "origin" row would instead test origin *at the reference level of side*,
    which is not the main effect and does not agree with `car::Manova(type=3)`.
    """
    formula = ("nerv01 + nerv02 + nerv03 + nerv04 + nerv05 ~ "
               "C(origin, Sum) * C(side, Sum)")
    res = MANOVA.from_formula(formula, data=df_complete).mv_test()

    pretty = {"Intercept": "(Intercept)", "C(origin, Sum)": "origin",
              "C(side, Sum)": "side", "C(origin, Sum):C(side, Sum)": "origin:side"}
    rows = []
    for term, out in res.results.items():
        tab = out["stat"]
        for statname in ["Pillai's trace", "Wilks' lambda",
                         "Hotelling-Lawley trace", "Roy's greatest root"]:
            r = tab.loc[statname]
            rows.append({"term": pretty.get(term, term), "statistic": statname,
                         "value": r["Value"], "num_df": r["Num DF"],
                         "den_df": r["Den DF"], "F": r["F Value"],
                         "p_value": r["Pr > F"]})
    table = pd.DataFrame(rows)

    if verbose:
        for statname in table["statistic"].unique():
            print(f"  --- {statname} ---")
            sub = table[table.statistic == statname]
            for _, r in sub.iterrows():
                print(f"    {r['term']:12s} stat={r['value']:.5f}  "
                      f"F={r['F']:.3f}  df=({r['num_df']:.0f},{r['den_df']:.0f})  "
                      f"p={r['p_value']:.3g}")
    table.to_csv(Path(outdir) / "results_manova.csv", index=False)
    return table


def run_anova_followup(df_complete, outdir=RESULTS_DIR, verbose: bool = True):
    """
    One Type II two-way ANOVA per trait, with partial eta-squared.

    Mirrors `car::Anova(type = "II")`.  Partial eta-squared is defined here as
    SS_effect / (SS_effect + SS_residual), which is the same definition used by
    the R script and by `pingouin`.
    """
    rows = []
    for trait in VEIN_COLS:
        fit = smf.ols(f"{trait} ~ origin * side", data=df_complete).fit()
        aov = sm.stats.anova_lm(fit, typ=2)
        ss_resid = aov.loc["Residual", "sum_sq"]
        for src in [i for i in aov.index if i != "Residual"]:
            ss = aov.loc[src, "sum_sq"]
            dfree = aov.loc[src, "df"]
            rows.append({"trait": trait, "Source": src, "SS": ss, "DF": dfree,
                         "MS": ss / dfree, "F": aov.loc[src, "F"],
                         "p-unc": aov.loc[src, "PR(>F)"],
                         "np2": ss / (ss + ss_resid)})
    table = pd.DataFrame(rows)
    if verbose:
        print(table.to_string(index=False))
    table.to_csv(Path(outdir) / "results_manova_followup.csv", index=False)
    return table


# ═════════════════════════════════════════════════════════════════════════════
# 5. Mardia's test of multivariate normality
# ═════════════════════════════════════════════════════════════════════════════

def mardia_test(df_complete, outdir=RESULTS_DIR, verbose: bool = True):
    """
    Mardia's multivariate skewness and kurtosis test.

    This is the justification for also running PERMANOVA: if the traits are not
    multivariate normal, the MANOVA p-values rest on an assumption the data do
    not satisfy, and a distribution-free test should agree with them.
    """
    X = df_complete[VEIN_COLS].to_numpy(float)
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    S_inv = np.linalg.inv(np.cov(X, rowvar=False))  # ddof=1, as R's cov()
    D = Xc @ S_inv @ Xc.T

    b1p = float((D ** 3).mean())
    stat_skew = n * b1p / 6.0
    df_skew = p * (p + 1) * (p + 2) / 6.0
    p_skew = float(stats.chi2.sf(stat_skew, df_skew))

    b2p = float((np.diag(D) ** 2).mean())
    stat_kurt = (b2p - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
    p_kurt = float(2 * stats.norm.sf(abs(stat_kurt)))

    out = pd.DataFrame([
        {"test": "skewness", "b": b1p, "statistic": stat_skew, "df": df_skew,
         "p_value": p_skew, "verdict": "NON-NORMAL" if p_skew < 0.05 else "normal"},
        {"test": "kurtosis", "b": b2p, "statistic": stat_kurt, "df": np.nan,
         "p_value": p_kurt, "verdict": "NON-NORMAL" if p_kurt < 0.05 else "normal"},
    ])
    if verbose:
        print(f"  skewness  b1p={b1p:.4f}  stat={stat_skew:.4f}  df={df_skew:.0f}"
              f"  p={p_skew:.6f}  {out.verdict[0]}")
        print(f"  kurtosis  b2p={b2p:.4f}  stat={stat_kurt:.4f}"
              f"              p={p_kurt:.6f}  {out.verdict[1]}")
    out.to_csv(Path(outdir) / "results_mardia.csv", index=False)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 6. PERMANOVA  (distribution-free counterpart of the MANOVA)
# ═════════════════════════════════════════════════════════════════════════════

def run_permanova(df_complete, n_permutations: int = 999, seed: int = 42,
                  outdir=RESULTS_DIR, verbose: bool = True):
    """
    Two-way PERMANOVA on Euclidean distances — equivalent to
    `vegan::adonis2(Y ~ origin * side, method = "euclidean", by = "terms")`.

    How it works, in words
    ----------------------
    PERMANOVA asks the same question as MANOVA ("do the groups differ?") but
    without assuming normality.  It splits the total spread of the data into
    the part explained by origin, by side, by their interaction, and the rest;
    it then reshuffles the bees thousands of times to see how large those parts
    would get purely by chance.

    Why this is fast here
    ---------------------
    The R version builds the full 4,100 x 4,100 distance matrix and takes about
    twenty minutes.  With Euclidean distances the sums of squares are algebraically
    identical to the classical multivariate Type I sums of squares, so they can be
    obtained by projecting the centred response onto an orthonormal basis of each
    model term.  That runs in well under a second and gives the same numbers to
    ten significant digits.

    The permutation p-value is `(1 + #{F_permuted >= F_observed}) / (1 + n_perm)`,
    the same estimator vegan uses, so the smallest attainable p-value with 999
    permutations is 0.001.
    """
    Y = df_complete[VEIN_COLS].to_numpy(float)
    Yc = Y - Y.mean(axis=0)
    n = len(Y)

    origin = (df_complete["origin"].to_numpy() == "Ribeirao_Preto").astype(float)
    side = (df_complete["side"].to_numpy() == "left").astype(float)
    term_names = ["origin", "side", "origin:side"]
    blocks = [origin[:, None], side[:, None], (origin * side)[:, None]]

    # Sequential ("Type I") orthonormal bases: each term is orthogonalised
    # against the intercept and every term entered before it.
    basis, spanned = [], np.ones((n, 1)) / np.sqrt(n)
    for B in blocks:
        resid = B - spanned @ (spanned.T @ B)
        q, _ = np.linalg.qr(resid)
        basis.append(q)
        spanned = np.hstack([spanned, q])

    ss_total = float((Yc ** 2).sum())
    ss_terms = np.array([float(((q.T @ Yc) ** 2).sum()) for q in basis])
    ss_resid = ss_total - ss_terms.sum()
    df_terms = np.ones(len(blocks))
    df_resid = n - 1 - len(blocks)
    F_obs = (ss_terms / df_terms) / (ss_resid / df_resid)

    rng = np.random.default_rng(seed)
    ge = np.zeros(len(blocks))
    for _ in range(n_permutations):
        Yp = Yc[rng.permutation(n)]
        ssp = np.array([float(((q.T @ Yp) ** 2).sum()) for q in basis])
        F_perm = ssp / ((ss_total - ssp.sum()) / df_resid)
        ge += F_perm >= F_obs
    p_values = (ge + 1) / (n_permutations + 1)

    table = pd.DataFrame({
        "term": term_names + ["Residual", "Total"],
        "Df": list(df_terms.astype(int)) + [df_resid, n - 1],
        "SumOfSqs": list(ss_terms) + [ss_resid, ss_total],
        "R2": list(ss_terms / ss_total) + [ss_resid / ss_total, 1.0],
        "F": list(F_obs) + [np.nan, np.nan],
        "Pr(>F)": list(p_values) + [np.nan, np.nan],
    })
    if verbose:
        print(f"  {n_permutations} permutations, Euclidean distance, n = {n}")
        print(table.to_string(index=False))
    table.to_csv(Path(outdir) / "results_permanova.csv", index=False)
    return table


# ═════════════════════════════════════════════════════════════════════════════
# 7. Linear Discriminant Analysis with leave-one-out cross-validation (Figure 3)
# ═════════════════════════════════════════════════════════════════════════════

def _lda_loo(X, y, labels):
    """
    Leave-one-out LDA classification, reproducing `MASS::lda(CV = TRUE)`.

    MATCH-R: the class priors are estimated once on the full sample and held
    fixed across folds (only the means and the pooled covariance are refitted
    without the held-out bee).  Re-estimating priors inside every fold — which
    is what a plain scikit-learn cross-validation loop does — reclassifies one
    or two bees that sit exactly on the decision boundary.
    """
    X = np.asarray(X, float)
    y = np.asarray(y)
    n, p = X.shape
    prior = np.array([(y == c).mean() for c in labels])
    log_prior = np.log(prior)
    pred = np.empty(n, dtype=object)

    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        Xi, yi = X[keep], y[keep]
        means, W, dfree = [], np.zeros((p, p)), 0
        for c in labels:
            Xc = Xi[yi == c]
            mu = Xc.mean(axis=0)
            means.append(mu)
            Xd = Xc - mu
            W += Xd.T @ Xd
            dfree += len(Xc) - 1
        S_inv = np.linalg.inv(W / dfree)
        score = [means[k] @ S_inv @ X[i] - 0.5 * means[k] @ S_inv @ means[k]
                 + log_prior[k] for k in range(len(labels))]
        pred[i] = labels[int(np.argmax(score))]
    return pred


def _lda_ld1(X, y, labels, negative_group):
    """
    LD1 scores from an LDA fitted on all the data (used for the density panels).

    MATCH-R: `MASS::lda` scales the discriminant axis so that the *within-group*
    variance of LD1 is 1 (not its total variance) and centres the scores on the
    prior-weighted mean of the group means.  Reproducing both conventions puts
    the Python and R density curves on the same axis.  `negative_group` fixes
    the arbitrary sign of the axis, exactly as the R script does.
    """
    X = np.asarray(X, float)
    y = np.asarray(y)
    p = X.shape[1]
    means, W, dfree, priors = [], np.zeros((p, p)), 0, []
    for c in labels:
        Xc = X[y == c]
        mu = Xc.mean(axis=0)
        means.append(mu)
        priors.append(len(Xc) / len(X))
        Xd = Xc - mu
        W += Xd.T @ Xd
        dfree += len(Xc) - 1
    S = W / dfree
    direction = np.linalg.solve(S, means[1] - means[0])   # two-group LDA axis
    direction = direction / np.sqrt(direction @ S @ direction)  # unit within-SD
    centre = sum(pi * mu for pi, mu in zip(priors, means))
    ld1 = (X - centre) @ direction
    if ld1[y == negative_group].mean() > 0:
        ld1 = -ld1
    return ld1


def run_lda(df_complete, outdir=RESULTS_DIR, verbose: bool = True):
    """
    Three LDA analyses:
      (a) Coyhaique vs Ribeirão Preto, right wings only;
      (b) right vs left wing within Coyhaique;
      (c) right vs left wing within Ribeirão Preto.

    Accuracy is leave-one-out cross-validated, so it is an honest out-of-sample
    number rather than a resubstitution figure.
    """
    results, rows = {}, []

    right = df_complete[df_complete["side"] == "right"]
    X, y = right[VEIN_COLS].to_numpy(), right["origin"].astype(str).to_numpy()
    labels = ["Chile", "Ribeirao_Preto"]
    pred = _lda_loo(X, y, labels)
    acc = float((pred == y).mean())
    cm = pd.crosstab(pd.Series(y, name="Actual"), pd.Series(pred, name="Predicted")
                     ).reindex(index=labels, columns=labels, fill_value=0)
    results["origin"] = {"accuracy": acc, "confusion": cm, "labels": labels,
                         "ld1": _lda_ld1(X, y, labels, "Chile"), "y": y}
    if verbose:
        print(f"  origin (right wing, LOO-CV): accuracy = {acc*100:.1f}%")
        print(cm.to_string())
    for a in labels:
        for b in labels:
            rows.append({"analysis": "origin (right wing)", "actual": a,
                         "predicted": b, "n": int(cm.loc[a, b]), "accuracy": acc})

    for orig in labels:
        sub = df_complete[df_complete["origin"] == orig]
        Xs, ys = sub[VEIN_COLS].to_numpy(), sub["side"].astype(str).to_numpy()
        side_labels = ["right", "left"]
        pr = _lda_loo(Xs, ys, side_labels)
        acc_s = float((pr == ys).mean())
        cms = pd.crosstab(pd.Series(ys, name="Actual"),
                          pd.Series(pr, name="Predicted")
                          ).reindex(index=side_labels, columns=side_labels, fill_value=0)
        results[f"side_{orig}"] = {"accuracy": acc_s, "confusion": cms,
                                   "labels": side_labels,
                                   "ld1": _lda_ld1(Xs, ys, side_labels, "left"), "y": ys}
        if verbose:
            print(f"  side within {ORIGIN_LABELS[orig]} (LOO-CV): "
                  f"accuracy = {acc_s*100:.1f}%")
            print(cms.to_string())
        for a in side_labels:
            for b in side_labels:
                rows.append({"analysis": f"side within {ORIGIN_LABELS[orig]}",
                             "actual": a, "predicted": b, "n": int(cms.loc[a, b]),
                             "accuracy": acc_s})

    pd.DataFrame(rows).to_csv(Path(outdir) / "results_lda.csv", index=False)
    return results


def fig_lda(lda_res, outdir=RESULTS_DIR):
    """LD1 density curves for the three LDA analyses — `plot_lda.jpg`."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    r = lda_res["origin"]
    for key, colour in [("Chile", ORIGIN_PAL["Coyhaique"]),
                        ("Ribeirao_Preto", ORIGIN_PAL["Ribeirão Preto"])]:
        vals = r["ld1"][r["y"] == key]
        g, dens = _kde_curve(vals)
        axes[0].fill_between(g, dens, alpha=0.35, color=colour,
                             label=ORIGIN_LABELS[key])
        axes[0].plot(g, dens, color=colour, linewidth=0.9)
    axes[0].set_title(f"Coyhaique vs RPR (acc: {r['accuracy']*100:.1f}%)")
    axes[0].legend(title="Origin")

    for ax, orig in zip(axes[1:], ["Chile", "Ribeirao_Preto"]):
        r = lda_res[f"side_{orig}"]
        for key in ["right", "left"]:
            vals = r["ld1"][r["y"] == key]
            g, dens = _kde_curve(vals)
            ax.fill_between(g, dens, alpha=0.35, color=SIDE_PAL[key], label=key)
            ax.plot(g, dens, color=SIDE_PAL[key], linewidth=0.9)
        ax.set_title(f"Side within {ORIGIN_LABELS[orig]} "
                     f"(acc: {r['accuracy']*100:.1f}%)")
        ax.legend(title="Side")

    for ax in axes:
        ax.set_xlabel("LD1 Score")
        ax.set_ylabel("Density")
    fig.suptitle("Linear Discriminant Analysis", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, Path(outdir) / "plot_lda.jpg")


# ═════════════════════════════════════════════════════════════════════════════
# 8. Linear mixed-effects models  (Table 2)
# ═════════════════════════════════════════════════════════════════════════════

def run_lme(df_complete, outdir=RESULTS_DIR, verbose: bool = True):
    """
    One mixed model per trait: `trait ~ origin * side + (1 | pair_id)`, REML.

    The random intercept for `pair_id` accounts for the fact that the two
    colonies of a pair share a queen, so their workers are not independent
    observations.  P-values use the Wald-z approximation (z = estimate / SE),
    which is what the R script reports as well.
    """
    rows = []
    for trait in VEIN_COLS:
        fit = smf.mixedlm(f"{trait} ~ origin * side", df_complete,
                          groups=df_complete["pair_id"]).fit(reml=True)
        n_obs = len(df_complete)
        # AIC/BIC on the REML criterion, matching lme4's convention.
        k = len(fit.params) - 1 + 1  # fixed effects + residual/group variance
        llf = fit.llf
        aic = -2 * llf + 2 * (k + 1)
        bic = -2 * llf + np.log(n_obs) * (k + 1)
        for term in fit.params.index:
            if term in ("Group Var", "groups RE"):
                continue
            rows.append({"trait": trait, "term": term,
                         "estimate": fit.params[term], "std_err": fit.bse[term],
                         "z": fit.tvalues[term], "p_value": fit.pvalues[term],
                         "AIC": aic, "BIC": bic})
    table = pd.DataFrame(rows)
    if verbose:
        print(table.to_string(index=False))
    table.to_csv(Path(outdir) / "results_lme.csv", index=False)
    return table


# ═════════════════════════════════════════════════════════════════════════════
# 9. Robust PCA and outlier diagnostics  (Figure 5)
# ═════════════════════════════════════════════════════════════════════════════

def run_robust_pca(df_complete, seed: int = 42, outdir=RESULTS_DIR,
                   verbose: bool = True):
    """
    PCA based on the Minimum Covariance Determinant (MCD) estimator.

    Purpose: check that the separation between origins seen in the ordinary PCA
    reflects the bulk of the bees and is not created by a handful of extreme
    wings.  Bees whose robust Mahalanobis distance exceeds the 97.5% point of a
    chi-squared distribution with 5 degrees of freedom are flagged as outliers.

    NOTE ON REPRODUCIBILITY: this is the one analysis whose numbers are not
    bit-identical between the two languages.  `sklearn.covariance.MinCovDet`
    and `rrcov::CovMcd` both implement MCD but draw different random subsets,
    so the variance explained differs by ~0.1 percentage points and the outlier
    count by a handful of bees out of 4,100.  The conclusion is unaffected.
    """
    X = scale_pop(df_complete[VEIN_COLS])
    mcd = MinCovDet(random_state=seed).fit(X)
    cov, centre = mcd.covariance_, mcd.location_

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    var_exp = eigvals / eigvals.sum() * 100
    scores = (X - centre) @ eigvecs

    dist = mcd.mahalanobis(X)
    cutoff = float(stats.chi2.ppf(0.975, df=5))
    n_out = int((dist > cutoff).sum())

    if verbose:
        print(f"  variance explained: RPC1={var_exp[0]:.1f}%, RPC2={var_exp[1]:.1f}%")
        print(f"  chi-squared(df=5) 97.5% cutoff: {cutoff:.2f}")
        print(f"  outliers flagged: {n_out} of {len(X)} "
              f"({n_out/len(X)*100:.1f}%)")

    pd.DataFrame({"RPC": range(1, 6), "variance_explained_pct": var_exp}).to_csv(
        Path(outdir) / "results_robust_pca.csv", index=False)
    return {"scores": scores, "var_exp": var_exp, "mahalanobis": dist,
            "cutoff": cutoff, "n_outliers": n_out,
            "origin_label": df_complete["origin_label"].to_numpy()}


def fig_robust_pca(rob, outdir=RESULTS_DIR):
    """Robust PCA scores, sorted distances and a chi-squared Q-Q plot."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    scores, var_exp = rob["scores"], rob["var_exp"]
    for key, colour in ORIGIN_PAL.items():
        m = rob["origin_label"] == key
        axes[0].scatter(scores[m, 0], scores[m, 1], s=5, alpha=0.3, color=colour,
                        label=key, linewidths=0)
        _confidence_ellipse(axes[0], scores[m, 0], scores[m, 1], color=colour,
                            linestyle="--", linewidth=1.8)
    axes[0].set_xlabel(f"RPC1 ({var_exp[0]:.1f}%)")
    axes[0].set_ylabel(f"RPC2 ({var_exp[1]:.1f}%)")
    axes[0].set_title("Robust PCA — by Origin")
    axes[0].legend(handles=[Line2D([], [], marker="o", linestyle="", color=c,
                                   label=k) for k, c in ORIGIN_PAL.items()],
                   title="Origin")

    d_sorted = np.sort(rob["mahalanobis"])
    axes[1].scatter(np.arange(1, len(d_sorted) + 1), d_sorted, s=3, alpha=0.4,
                    color="#4393C3", linewidths=0)
    axes[1].axhline(rob["cutoff"], color="red", linestyle="--", linewidth=1.2)
    axes[1].text(len(d_sorted) * 0.05, rob["cutoff"] * 1.06,
                 f"Cutoff = {rob['cutoff']:.1f}", color="red", fontsize=9)
    axes[1].set_xlabel("Sample (sorted)")
    axes[1].set_ylabel("Mahalanobis Distance")
    axes[1].set_title(f"Outlier Detection ({rob['n_outliers']} outliers)")

    probs = np.linspace(0.001, 0.999, len(d_sorted))
    theoretical = np.sort(stats.chi2.ppf(probs, df=5))
    axes[2].scatter(theoretical, d_sorted, s=3, alpha=0.3, color="#4393C3",
                    linewidths=0)
    lims = [min(theoretical.min(), d_sorted.min()),
            max(theoretical.max(), d_sorted.max())]
    axes[2].plot(lims, lims, "r--", linewidth=1.4)
    axes[2].set_xlim(lims)
    axes[2].set_ylim(lims)
    axes[2].set_xlabel("$\\chi^2$ (df=5) quantiles")
    axes[2].set_ylabel("Observed Mahalanobis Distance")
    axes[2].set_title("Q-Q Plot — Multivariate Normality")

    fig.suptitle("Robust PCA & Outlier Diagnostics", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, Path(outdir) / "plot_robust_pca.jpg")


# ═════════════════════════════════════════════════════════════════════════════
# 10. Asymmetry tests  (Table S3, Figure 4)
# ═════════════════════════════════════════════════════════════════════════════

def run_asymmetry(df_complete, outdir=RESULTS_DIR, verbose: bool = True):
    """
    Right-minus-left differences, tested per colony and per trait.

    Each bee contributes one paired difference: its right and left wings are
    matched on the bee identifier `amost` with an inner join inside each colony,
    so a test never compares one bee's right wing with another bee's left wing.
    Both a paired t-test and a paired Wilcoxon signed-rank test are reported.

    Because 120 tests are run (24 colonies x 5 traits), the p-values are also
    reported after three multiple-comparison corrections: Bonferroni (strictest),
    Holm-Bonferroni (same guarantee, uniformly more powerful) and
    Benjamini-Hochberg (controls the false discovery rate instead).

    `wilcox_stat` is V, the sum of the positive signed ranks, matching R's
    `wilcox.test()` output.
    """
    rows = []
    for trait in VEIN_COLS:
        for colony in sorted(df_complete["colony_label"].unique()):
            sub = df_complete[df_complete["colony_label"] == colony]
            right = sub[sub["side"] == "right"][["amost", trait]].rename(
                columns={trait: "val_r"})
            left = sub[sub["side"] == "left"][["amost", trait]].rename(
                columns={trait: "val_l"})
            paired = right.merge(left, on="amost", how="inner").dropna()
            if len(paired) < 3:
                continue

            diff = (paired["val_r"] - paired["val_l"]).to_numpy()
            t_res = stats.ttest_rel(paired["val_r"], paired["val_l"])
            try:
                w_res = stats.wilcoxon(paired["val_r"], paired["val_l"],
                                       zero_method="wilcox", correction=True,
                                       method="approx", alternative="two-sided")
                w_p = float(w_res.pvalue)
            except ValueError:
                w_p = np.nan
            nz = diff[diff != 0]
            ranks = stats.rankdata(np.abs(nz))
            v_stat = float(ranks[nz > 0].sum()) if len(nz) else np.nan

            rows.append({
                "colony": _relabel_colony(colony),
                "origin": "Coyhaique" if colony.startswith("CH") else "Ribeirão Preto",
                "trait": trait,
                "mean_diff_RL": float(diff.mean()),
                "n_pairs": len(paired),
                "t_stat": float(t_res.statistic),
                "t_pvalue": float(t_res.pvalue),
                "wilcox_stat": v_stat,
                "wilcox_pvalue": w_p,
            })

    table = pd.DataFrame(rows)

    methods = {"bonf": "bonferroni", "holm": "holm", "bh": "fdr_bh"}
    t_p = table["t_pvalue"].fillna(1.0).to_numpy()
    for suffix, method in methods.items():
        table[f"t_pvalue_{suffix}"] = multipletests(t_p, method=method)[1]

    valid = table["wilcox_pvalue"].notna().to_numpy()
    for suffix, method in methods.items():
        col = np.full(len(table), np.nan)
        col[valid] = multipletests(table.loc[valid, "wilcox_pvalue"].to_numpy(),
                                   method=method)[1]
        table[f"wilcox_pvalue_{suffix}"] = col

    if verbose:
        print(f"  {len(table)} colony x trait paired tests")
        for label, col in [("raw", "wilcox_pvalue"),
                           ("Bonferroni", "wilcox_pvalue_bonf"),
                           ("Holm-Bonferroni", "wilcox_pvalue_holm"),
                           ("Benjamini-Hochberg", "wilcox_pvalue_bh")]:
            print(f"    significant at p<0.05, {label:18s}: "
                  f"{int((table[col] < 0.05).sum())} / {int(table[col].notna().sum())}")

    table.to_csv(Path(outdir) / "results_table_S3_wilcoxon.csv", index=False)
    return table


def _sig_stars(p) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _fmt_pval(p) -> str:
    if pd.isna(p):
        return ""
    if p >= 1:
        return "1"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def _draw_asymmetry_grid(ax, sub, colonies, traits, show_pvalue, cmap, norm):
    """Shared drawing routine for every asymmetry heatmap variant."""
    pivot = sub.pivot(index="colony", columns="trait", values="mean_diff_RL"
                      ).reindex(index=colonies, columns=traits)
    ax.imshow(pivot.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(traits)), traits)
    ax.set_yticks(range(len(colonies)), colonies)
    ax.grid(False)
    for i, colony in enumerate(colonies):
        for j, trait in enumerate(traits):
            r = sub[(sub.colony == colony) & (sub.trait == trait)]
            if r.empty:
                continue
            r = r.iloc[0]
            txt = f"{r['mean_diff_RL']:.4f}"
            if show_pvalue:
                txt += f"\np={_fmt_pval(r['pval'])}"
                if r["sig"]:
                    txt += f" {r['sig']}"
            elif r["sig"]:
                txt += f"\n{r['sig']}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.8,
                    linespacing=0.95)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(traits), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(colonies), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)


def fig_asymmetry_heatmaps(asym, outdir=RESULTS_DIR):
    """
    The asymmetry heatmaps: one with raw p-values (Figure 4), one panel per
    correction method, and a combined four-panel figure.

    Colour encodes the mean right-minus-left difference in millimetres: red
    means the right wing is larger, blue means the left wing is larger.
    """
    outdir = Path(outdir)
    traits = [VEIN_LABELS[c] for c in VEIN_COLS]
    colonies = sorted(asym["colony"].unique())

    base = asym.copy()
    base["trait"] = base["trait"].map(VEIN_LABELS)

    vmax = float(np.abs(base["mean_diff_RL"]).max())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "RdBu_paper", ["#053061", "#FFFFFF", "#67001F"])
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # --- Figure 4: raw Wilcoxon p-values, stars only ------------------------
    sub = base.assign(pval=base["wilcox_pvalue"])
    sub["sig"] = sub["pval"].map(_sig_stars)
    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_asymmetry_grid(ax, sub, colonies, traits, False, cmap, norm)
    ax.set_xlabel("Trait")
    ax.set_ylabel("Colony")
    ax.set_title("Wing Asymmetry (Right − Left) by Colony and Trait\n"
                 "* p<0.05  ** p<0.01  *** p<0.001 (Wilcoxon)", fontsize=11)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label="Mean Difference\n(Right − Left)")
    fig.tight_layout()
    _save(fig, outdir / "plot_asymmetry_heatmap.jpg")
    plt.close(fig)

    methods = {
        "Raw": "wilcox_pvalue",
        "Bonferroni": "wilcox_pvalue_bonf",
        "Holm-Bonferroni": "wilcox_pvalue_holm",
        "Benjamini-Hochberg (BH)": "wilcox_pvalue_bh",
    }

    # --- one heatmap per correction method ----------------------------------
    for name, col in methods.items():
        sub = base.assign(pval=base[col])
        sub["sig"] = sub["pval"].map(_sig_stars)
        fig, ax = plt.subplots(figsize=(10, 8))
        _draw_asymmetry_grid(ax, sub, colonies, traits, True, cmap, norm)
        ax.set_xlabel("Trait")
        ax.set_ylabel("Colony")
        ax.set_title(f"Wing Asymmetry — {name} correction\n"
                     "Cell text: mean(R−L) and adjusted Wilcoxon p-value;  "
                     "* p<0.05  ** p<0.01  *** p<0.001", fontsize=11)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     label="Mean Difference\n(Right − Left)")
        fig.tight_layout()
        slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
        slug = "_".join(filter(None, slug.split("_")))
        _save(fig, outdir / f"plot_asymmetry_heatmap_{slug}.jpg")
        plt.close(fig)

    # --- all four side by side ----------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(20, 8), sharey=True)
    for ax, (name, col) in zip(axes, methods.items()):
        sub = base.assign(pval=base[col])
        sub["sig"] = sub["pval"].map(_sig_stars)
        _draw_asymmetry_grid(ax, sub, colonies, traits, True, cmap, norm)
        ax.set_title(name, fontweight="bold", fontsize=11)
        ax.set_xlabel("Trait")
    axes[0].set_ylabel("Colony")
    fig.suptitle("Wing Asymmetry by Colony × Trait under each correction\n"
                 "Cell text: mean(R−L) and adjusted Wilcoxon p-value; "
                 "* p<0.05  ** p<0.01  *** p<0.001", fontsize=12)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                 label="Mean Difference\n(Right − Left)", fraction=0.02)
    return _save(fig, outdir / "plot_asymmetry_heatmap_corrections.jpg")


def fig_asymmetry_pvalue_comparison(asym, outdir=RESULTS_DIR):
    """
    Bar chart of -log10(p) for all 120 tests under each correction method.

    This shows directly how many findings survive each correction; the dashed
    line is the p = 0.05 threshold.
    """
    methods = {
        "Raw": "wilcox_pvalue",
        "Bonferroni": "wilcox_pvalue_bonf",
        "Holm-Bonferroni": "wilcox_pvalue_holm",
        "Benjamini-Hochberg (BH)": "wilcox_pvalue_bh",
    }
    base = asym.copy()
    base["trait"] = base["trait"].map(VEIN_LABELS)
    base["label"] = base["colony"] + " | " + base["trait"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=False)
    summary = []
    for ax, (name, col) in zip(axes.ravel(), methods.items()):
        d = base[["label", col]].rename(columns={col: "pval"}).copy()
        d["neg_log10p"] = -np.log10(d["pval"].clip(lower=1e-10))
        d["sig"] = d["pval"] < 0.05
        d = d.sort_values("neg_log10p")
        ax.barh(d["label"], d["neg_log10p"],
                color=np.where(d["sig"], "#B2182B", "#9E9E9E"), height=0.8)
        ax.axvline(-np.log10(0.05), color="red", linestyle="--", linewidth=0.9)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("$-\\log_{10}(p_{adj})$")
        ax.set_ylabel("Colony × Trait")
        ax.tick_params(axis="y", labelsize=4.5)
        ax.margins(y=0.005)
        summary.append({"method": name, "n_total": int(d["pval"].notna().sum()),
                        "n_significant": int(d["sig"].sum())})
    handles = [Patch(facecolor="#B2182B", label="p < 0.05"),
               Patch(facecolor="#9E9E9E", label="n.s.")]
    axes[0, 1].legend(handles=handles, loc="lower right")
    fig.suptitle("Wilcoxon asymmetry tests — effect of multiple-comparison "
                 "correction", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = _save(fig, Path(outdir) / "plot_asymmetry_pvalue_comparison.jpg")
    summary = pd.DataFrame(summary)
    print(summary.to_string(index=False))
    summary.to_csv(Path(outdir) / "results_asymmetry_correction_summary.csv",
                   index=False)
    return path


# ═════════════════════════════════════════════════════════════════════════════
# 11. Hierarchical clustering of colony means  (Figure S3)
# ═════════════════════════════════════════════════════════════════════════════

def colony_means(df_complete):
    """Mean of each vein trait per colony, right wings only."""
    right = df_complete[df_complete["side"] == "right"]
    means = (right.groupby(["colony_label", "origin", "pair_id"], observed=True)
             [VEIN_COLS].mean().reset_index())
    means["colony"] = means["colony_label"].map(_relabel_colony)
    means["origin_label"] = means["origin"].map(ORIGIN_LABELS).astype(str)
    return means


def _clustermap(matrix, row_labels, col_labels, row_linkage, col_linkage,
                row_annotation, title, path):
    """A pheatmap-style clustered heatmap: dendrograms + annotation bar + cells."""
    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 0.14, 5.0],
                          height_ratios=[0.9, 5.0], wspace=0.03, hspace=0.03)

    ax_col = fig.add_subplot(gs[0, 2])
    dn_col = dendrogram(col_linkage, ax=ax_col, color_threshold=0,
                        above_threshold_color="#444444", no_labels=True)
    ax_col.axis("off")

    ax_row = fig.add_subplot(gs[1, 0])
    dn_row = dendrogram(row_linkage, ax=ax_row, orientation="left",
                        color_threshold=0, above_threshold_color="#444444",
                        no_labels=True)
    # scipy's "left" dendrogram draws its first leaf at the bottom, while
    # `imshow` draws the first row at the top.  Flipping the dendrogram axis
    # lines the two up *and* reproduces pheatmap's row order, so the Python and
    # R heatmaps can be laid side by side.
    ax_row.invert_yaxis()
    ax_row.axis("off")

    row_order = dn_row["leaves"]
    col_order = dn_col["leaves"]
    data = matrix[np.ix_(row_order, col_order)]
    rlabs = [row_labels[i] for i in row_order]
    clabs = [col_labels[i] for i in col_order]

    ax_ann = fig.add_subplot(gs[1, 1])
    ann_colours = [ORIGIN_PAL[row_annotation[i]] for i in row_order]
    ax_ann.imshow(np.arange(len(rlabs)).reshape(-1, 1), aspect="auto",
                  cmap=matplotlib.colors.ListedColormap(ann_colours))
    ax_ann.set_xticks([])
    ax_ann.set_yticks([])
    ax_ann.grid(False)

    ax = fig.add_subplot(gs[1, 2])
    vmax = float(np.abs(data).max())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "RdBu_r", RDBU_11[::-1])
    im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(clabs)), clabs)
    ax.yaxis.tick_right()
    ax.set_yticks(range(len(rlabs)), rlabs, fontsize=9)
    ax.grid(False)
    for i in range(len(rlabs)):
        for j in range(len(clabs)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    ax.set_xticks(np.arange(-0.5, len(clabs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rlabs), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.09)
    cbar.set_label("Standardised\nmean", fontsize=9)
    legend = [Patch(facecolor=c, label=k) for k, c in ORIGIN_PAL.items()]
    fig.legend(handles=legend, title="Origin", loc="upper left",
               bbox_to_anchor=(0.005, 0.99), fontsize=9)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    return _save(fig, path)


def fig_clustering(df_complete, outdir=RESULTS_DIR, verbose: bool = True):
    """
    Two clustered heatmaps of the 24 colony means.

    * `plot_cluster_heatmap.jpg` — Euclidean distance, Ward linkage.
      (`scipy`'s "ward" is the same algorithm as R's "ward.D2".)
    * `plot_cluster_heatmap_mahalanobis.jpg` — Mahalanobis distance between
      colony means, using the pooled within-colony covariance matrix estimated
      from the individual bees.  Mahalanobis distance rescales each direction by
      how variable it is, so a difference along a tightly-canalised trait counts
      for more than the same difference along a variable one.
    """
    outdir = Path(outdir)
    means = colony_means(df_complete)
    mat = scale_pop(means[VEIN_COLS])
    row_labels = means["colony"].tolist()
    col_labels = [VEIN_LABELS[c] for c in VEIN_COLS]
    annotation = means["origin_label"].tolist()

    fig1 = _clustermap(
        mat, row_labels, col_labels,
        linkage(pdist(mat, metric="euclidean"), method="ward"),
        linkage(pdist(mat.T, metric="euclidean"), method="ward"),
        annotation,
        "Hierarchical Clustering — Colony Mean Vein Measurements (Right Wing)",
        outdir / "plot_cluster_heatmap.jpg")
    plt.close("all")

    # Pooled within-colony covariance from bee-level right-wing data
    right = df_complete[df_complete["side"] == "right"]
    p = len(VEIN_COLS)
    pooled, weight = np.zeros((p, p)), 0
    for _, grp in right.groupby("colony_label", observed=True):
        w = len(grp) - 1
        pooled += np.cov(grp[VEIN_COLS].to_numpy(), rowvar=False) * w
        weight += w
    S_inv = np.linalg.inv(pooled / weight)

    raw = means[VEIN_COLS].to_numpy()
    n = len(raw)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = raw[i] - raw[j]
            D[i, j] = D[j, i] = np.sqrt(v @ S_inv @ v)
    mahal_df = pd.DataFrame(D, index=row_labels, columns=row_labels)
    mahal_df.round(6).to_csv(outdir / "results_colony_mahalanobis.csv")
    if verbose:
        print("  pairwise Mahalanobis distances between colony means "
              "-> results_colony_mahalanobis.csv")

    # Traits are clustered by correlation distance here, independent of the rows.
    trait_dist = 1 - np.abs(np.corrcoef(mat, rowvar=False))
    fig2 = _clustermap(
        mat, row_labels, col_labels,
        linkage(squareform(D, checks=False), method="ward"),
        linkage(squareform(trait_dist, checks=False), method="ward"),
        annotation,
        "Hierarchical Clustering — Mahalanobis distance (rows), Ward linkage\n"
        "Colony Mean Vein Measurements (Right Wing)",
        outdir / "plot_cluster_heatmap_mahalanobis.jpg")
    plt.close("all")
    return fig1, fig2, mahal_df


# ═════════════════════════════════════════════════════════════════════════════
# 12. Reaction norms  (Figure S2)
# ═════════════════════════════════════════════════════════════════════════════

def pair_means(df_complete):
    """Mean of each trait per colony pair and origin (right wings only)."""
    right = df_complete[df_complete["side"] == "right"]
    return (right.groupby(["pair_id", "origin"], observed=True)[VEIN_COLS]
            .mean().reset_index())


def fig_reaction_norms(df_complete, outdir=RESULTS_DIR):
    """
    One line per colony pair, joining its mean in Coyhaique to its mean in
    Ribeirão Preto.  Parallel lines mean every genotype responded to the change
    of environment in the same way; crossing lines are genotype x environment
    interaction.
    """
    rn = pair_means(df_complete)
    pair_ids = sorted(rn["pair_id"].unique())
    colours = dict(zip(pair_ids, PAIR_PAL))

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for idx, (ax, col) in enumerate(zip(axes, VEIN_COLS)):
        for pid in pair_ids:
            sub = rn[rn["pair_id"] == pid].set_index("origin")
            xs, ys = [], []
            for pos, orig in enumerate(["Chile", "Ribeirao_Preto"]):
                if orig in sub.index:
                    xs.append(pos)
                    ys.append(sub.loc[orig, col])
            ax.plot(xs, ys, "-o", color=colours[pid], linewidth=2.0,
                    markersize=7, markeredgecolor="#555555",
                    markeredgewidth=0.4, label=pid)
        ax.set_xticks([0, 1], ["Coyhaique", "Ribeirão Preto"], rotation=15,
                      ha="right")
        ax.set_xlim(-0.3, 1.3)
        ax.set_xlabel("Origin")
        ax.set_ylabel("Mean measurement (mm)" if idx == 0 else "")
        ax.set_title(VEIN_LABELS[col])
    axes[0].legend(title="Colony Pair", fontsize=6.5, ncol=2, loc="best")
    fig.suptitle("Reaction Norms — Colony Pair Means Across Environments "
                 "(Right Wing)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, Path(outdir) / "plot_reaction_norms.jpg")


# ═════════════════════════════════════════════════════════════════════════════
# 13. Genotype x Environment interaction  (Figure 6)
# ═════════════════════════════════════════════════════════════════════════════

def run_gxe(df_complete, outdir=RESULTS_DIR, verbose: bool = True):
    """
    Correlate each colony pair's mean in Coyhaique with its mean in Ribeirão Preto.

    A high correlation means the ranking of genotypes was preserved across the
    two environments (little G x E); a low one means the ranking changed.
    Pearson r measures a linear relationship, Spearman rho only the ranking.
    """
    rn = pair_means(df_complete)
    wide = rn.pivot(index="pair_id", columns="origin", values=VEIN_COLS)
    rows = []
    for trait in VEIN_COLS:
        ch = wide[(trait, "Chile")].to_numpy(float)
        rp = wide[(trait, "Ribeirao_Preto")].to_numpy(float)
        pear = stats.pearsonr(ch, rp)
        spear = stats.spearmanr(ch, rp)
        rows.append({"trait": trait, "pearson_r": pear[0], "pearson_p": pear[1],
                     "spearman_rho": spear[0], "spearman_p": spear[1]})
    table = pd.DataFrame(rows)
    if verbose:
        print(table.to_string(index=False))
    table.to_csv(Path(outdir) / "results_gxe_correlations.csv", index=False)
    return table, wide


def fig_gxe(gxe_table, wide, outdir=RESULTS_DIR):
    """Scatter of Coyhaique mean vs Ribeirão Preto mean, one panel per trait."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for idx, (ax, trait) in enumerate(zip(axes, VEIN_COLS)):
        ch = wide[(trait, "Chile")].to_numpy(float)
        rp = wide[(trait, "Ribeirao_Preto")].to_numpy(float)
        pids = wide.index.tolist()

        lo, hi = min(ch.min(), rp.min()) * 0.98, max(ch.max(), rp.max()) * 1.02
        ax.plot([lo, hi], [lo, hi], "--", color="black", alpha=0.3, linewidth=1.2)
        slope, intercept = np.polyfit(ch, rp, 1)
        xs = np.linspace(lo, hi, 50)
        ax.plot(xs, intercept + slope * xs, color="#B2182B", linewidth=1.8)
        ax.scatter(ch, rp, s=45, facecolor="#4393C3", edgecolor="#2166AC",
                   zorder=3)
        for x, y, pid in zip(ch, rp, pids):
            ax.annotate(pid, (x, y), fontsize=6.5, xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Coyhaique mean (mm)")
        ax.set_ylabel("RPR mean (mm)" if idx == 0 else "")
        r = gxe_table.loc[gxe_table.trait == trait].iloc[0]
        ax.set_title(f"{VEIN_LABELS[trait]}\nr={r.pearson_r:.2f}, "
                     f"p={r.pearson_p:.3f}", fontsize=10)
    fig.suptitle("G×E Interaction — Coyhaique vs Ribeirão Preto Colony Means",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, Path(outdir) / "plot_gxe_scatter.jpg")
