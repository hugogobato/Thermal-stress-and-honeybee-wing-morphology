# PERMANOVA — 2-way design (origin × side)
# Run this separately; it takes ~20 min on the full dataset (n=4130, 999 perms).
#
# Usage:  cd R && Rscript permanova.R

library(vegan)
library(readr)
library(dplyr)

sink("permanova_log.txt")

# ── Load data ─────────────────────────────────────────────────────────────────
VEIN_COLS <- c("nerv01", "nerv02", "nerv03", "nerv04", "nerv05")

df <- read_csv("../data/data_clean.csv", show_col_types = FALSE) |>
  mutate(
    origin  = factor(origin,  levels = c("Chile", "Ribeirao_Preto")),
    side    = factor(side,    levels = c("right", "left"))
  )
df_complete <- df |> tidyr::drop_na(all_of(VEIN_COLS))

cat(sprintf("n = %d rows\n", nrow(df_complete)))

# ── 2-way PERMANOVA ───────────────────────────────────────────────────────────
# FIX: The Python notebook ran three separate one-way PERMANOVAs (one for
# origin, one for side, and one for a combined origin_side factor) instead of
# a true 2-way design.  Here we use a proper 2-way PERMANOVA with
# vegan::adonis2(Y ~ origin * side, by = "terms").
set.seed(42)
Y_perm    <- as.matrix(df_complete[VEIN_COLS])
perm_full <- vegan::adonis2(
  Y_perm ~ origin * side,
  data         = df_complete,
  method       = "euclidean",
  permutations = 999,
  by           = "terms"
)

cat("=== 2-way PERMANOVA (euclidean distance, 999 permutations) ===\n")
print(perm_full)

sink()

# ── Save results ──────────────────────────────────────────────────────────────
write.csv(as.data.frame(perm_full), "results_permanova.csv", row.names = TRUE)
cat("Saved: results_permanova.csv\n")
