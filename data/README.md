# Data dictionary

Two files, both plain comma-separated text. You can open either one in Excel,
LibreOffice, R or Python without any conversion.

---

## `data_clean.csv` — one row per measured wing

4,130 rows (2,065 bees × 2 wings), 12 columns.

| Column | Type | Meaning |
|---|---|---|
| `pair_id` | integer 1–12 | Colony **pair** identifier. The Coyhaique colony and the Ribeirão Preto colony that share a pair number came from the same queen, so their workers are sisters. |
| `origin` | text | Where the brood developed: `Chile` (Coyhaique, cool) or `Ribeirao_Preto` (warm). These are the raw codes stored in the file; the figures relabel them "Coyhaique" and "Ribeirão Preto". |
| `colony_own` | integer 1–12 | Colony number within its own locality (equal to `pair_id`). |
| `colony_label` | text | Colony code, `CH01`–`CH12` or `RP01`–`RP12`. Relabelled `COY…` / `RPR…` in the figures and result tables. |
| `amost` | integer | Bee identifier ("amostra" = sample). **A bee's right and left wing share the same `amost` within a colony**, which is what makes the paired asymmetry tests possible. |
| `side` | text | `right` or `left` forewing. |
| `hamuli` | text | Number of hamuli (wing-coupling hooks) counted under the stereomicroscope. Stored as text because a few records are `--` (not countable). Not used in any multivariate analysis — it is a count, not a length, and violates the normality assumptions those methods rest on. |
| `nerv01` … `nerv05` | number (mm) | The five linear wing-venation measurements, taken from digitised images. Labelled **M1–M5** in the figures. |

### The five vein measurements

Vein nomenclature follows Engel (2001): A = anal, C = costal, Cu = cubital,
M = medial, Rs = radial sector.

| Column | Figure label | Description |
|---|---|---|
| `nerv01` | M1 | Distal portion of A, between cross-veins Cu-a and Cu-2. |
| `nerv02` | M2 | Proximal portion of Cu, between Cu-a and 1m-cu. |
| `nerv03` | M3 | Proximal portion of M, between the first abscissa of Rs and the intersection of distal M+Cu with Cu. |
| `nerv04` | M4 | Length of cross-vein 2m-cu, between Cu1 and M. |
| `nerv05` | M5 | Distance between two points of the second medial cell bounded by M and Cu1. |

### Missing values

`nerv04` is missing for 2 wings and `nerv05` for 30 wings (a damaged or
unreadable region of the image). Every analysis uses **complete cases only**:
wings with all five measurements present. That leaves **4,100 wings** —
Coyhaique 1,115 right + 1,117 left, Ribeirão Preto 931 right + 937 left.

---

## `data_boxplot_media_long.csv` — one row per bee (right wing)

2,400 rows. Used only for the colony-pair boxplot (`plot_boxplot_media.jpg`).

| Column | Type | Meaning |
|---|---|---|
| `AMOSTRA` | integer | Bee identifier. |
| `colony_code` | text | `CH01`–`CH12` / `RP01`–`RP12`. |
| `mean_nerv` | number (mm) | Mean of that bee's five right-wing vein measurements. |
| `origin` | text | `Chile` or `Ribeirão Preto`. |
| `colony_num` | integer | Colony pair number, taken from `colony_code`. |

---

## Colony renaming

Colony numbers in these files reflect the final naming agreed with the
co-authors (the mapping recorded in *Renomeação dos ninhos.xlsx*). That
renaming has **already been applied** to the files shipped here, so no
preparation step is needed: the two CSV files above are exactly what the R and
Python pipelines read.
