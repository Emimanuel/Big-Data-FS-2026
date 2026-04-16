 # =============================================================================
# FireCastRL: Wildfire Forecasting (GRIDMET + IRWIN, 2014-2025)
# R implementation of the wildfire-forecasting-get-started.ipynb notebook
# =============================================================================
# Requirements:
#   install.packages(c("data.table", "ggplot2", "caret", "keras3"))
#   keras3::install_keras()   # installs TensorFlow backend

# -----------------------------------------------------------------------------
# 0. Libraries
# -----------------------------------------------------------------------------
library(data.table)   # fast I/O and data manipulation
library(ggplot2)      # visualisation
library(future.apply)
plan(multisession)

#install.packages("rlang")
#install.packages("devtools")
#pak::pak("benyamindsmith/RKaggle")
dir.create("data/intermediate", recursive = TRUE, showWarnings = FALSE)
# -----------------------------------------------------------------------------
# 1. Load the dataset
# -----------------------------------------------------------------------------
library(RKaggle)
df <- RKaggle::get_dataset("firecastrl/us-wildfire-dataset")
df <- as.data.table(df)
cat("Shape:", nrow(df), "x", ncol(df), "\n")
cat("Columns:", paste(names(df), collapse = ", "), "\n")
print(head(df))

# -----------------------------------------------------------------------------
# 1.b Unique ID
# -----------------------------------------------------------------------------
SEQ_LEN <- 75L

df[, row_in_group := (seq_len(.N) - 1L) %% SEQ_LEN + 1L]
df[, seq_id       := (seq_len(.N) - 1L) %/% SEQ_LEN + 1L]

cat("Unique sequence IDs:", uniqueN(df$seq_id), "\n")

# -----------------------------------------------------------------------------
# 4. Prepare 75-day sequences
# -----------------------------------------------------------------------------
FEATURES <- c("pr", "rmax", "rmin", "sph", "srad", "tmmn", "tmmx",
              "vs", "bi", "fm100", "fm1000", "erc", "etr", "pet", "vpd")

# -----------------------------------------------------------------------------
# 4.5 prepare wide
# -----------------------------------------------------------------------------
anchor       <- 61L
static_cols  <- c("latitude", "longitude", "datetime", "Wildfire", "row_in_group")
dynamic_cols <- setdiff(names(df), c(static_cols, "seq_id", "t_idx"))

# Offset relative to anchor so columns are t-60, t-1, t0, t+1 etc.
df[, t_idx := seq_len(.N) - anchor, by = seq_id]
df <- df[t_idx <= 0]
# Extract static info at anchor row only
static_part <- df[t_idx == 0, c("seq_id", ..static_cols)]

# Reshape only dynamic columns
flat_dynamic <- dcast(
  df,
  seq_id ~ t_idx,
  value.var = dynamic_cols
)

# Join static anchor info back
flat_df <- static_part[flat_dynamic, on = "seq_id"]

##remove enoded NF and numeric outcome
flat_df <- flat_df[rowSums(flat_df == 32767, na.rm = TRUE) == 0, ]
flat_df$Wildfire <- ifelse(flat_df$Wildfire == "Yes", 1, 0)

# -----------------------------------------------------------------------------
# 5 reduces size
# -----------------------------------------------------------------------------
# Count Yes and No
n_yes <- sum(flat_df$Wildfire == "Yes")
n_no  <- sum(flat_df$Wildfire == "No")

# How many No rows to discard
n_discard <- n_no - n_yes

# Row indices to drop (sampled from No rows)
no_indices    <- which(flat_df$Wildfire == "No")
discard_indices <- sample(no_indices, size = n_discard)

# Keep everything except discarded rows
balanced_df <- flat_df[-discard_indices, ]

# Verify
table(balanced_df$Wildfire)


rm(flat_dynamic, static_part)


saveRDS(balanced_df, "data/intermediate/wildfire_cleaned_balanced.rds")
saveRDS(flat_df, "data/intermediate/wildfire_cleaned_flat.rds")
saveRDS(df, "data/intermediate/wildfire_cleaned.rds")