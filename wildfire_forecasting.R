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
library(caret)        # createDataPartition / confusionMatrix
library(keras3)       # LSTM via TensorFlow/Keras

# -----------------------------------------------------------------------------
# 1. Load the dataset
# -----------------------------------------------------------------------------
setwd("C:/Users/emanu/OneDrive - Universität Zürich UZH/Lukas Andreas Gebhardt's files - Big Data;)")
df <- fread("Wildfire_Dataset.csv", data.table = TRUE)
df[, datetime := as.Date(datetime)]

cat("Shape:", nrow(df), "x", ncol(df), "\n")
cat("Columns:", paste(names(df), collapse = ", "), "\n")
print(head(df))

# -----------------------------------------------------------------------------
# 1.b Unique ID
# -----------------------------------------------------------------------------
SEQ_LEN <- 75L
setorder(df, latitude, longitude, datetime)

df[, row_in_group := seq_len(.N), by = .(latitude, longitude)]
df[, block_num    := (row_in_group - 1L) %/% SEQ_LEN]
df[, seq_id       := .GRP, by = .(latitude, longitude, block_num)]

df[, c("row_in_group", "block_num") := NULL]  # drop helpers

cat("Unique sequence IDs:", uniqueN(df$seq_id), "\n")
# -----------------------------------------------------------------------------
# 2. Class balance
# -----------------------------------------------------------------------------
class_counts <- df[, .N, by = Wildfire]
print(class_counts)

ggplot(df, aes(x = Wildfire, fill = Wildfire)) +
  geom_bar() +
  scale_fill_manual(values = c("No" = "#E41A1C", "Yes" = "#377EB8")) +
  labs(title = "Wildfire Ignition Distribution", x = "Wildfire", y = "Count") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")
# -----------------------------------------------------------------------------
# 3. Feature exploration  (correlation heatmap + distributions)
# -----------------------------------------------------------------------------
FEATURES <- c("pr", "rmax", "rmin", "sph", "srad", "tmmn", "tmmx",
              "vs", "bi", "fm100", "fm1000", "erc", "etr", "pet", "vpd")

# Correlation heatmap
cor_mat  <- cor(df[, ..FEATURES], use = "pairwise.complete.obs")
cor_long <- as.data.table(as.table(cor_mat))
setnames(cor_long, c("Var1", "Var2", "Correlation"))

ggplot(cor_long, aes(Var1, Var2, fill = Correlation)) +
  geom_tile(colour = "white") +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#D6604D",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(title = "Feature Correlation Heatmap") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Feature distributions by class
df_long <- melt(df[, c("Wildfire", ..FEATURES), with = FALSE],
                id.vars = "Wildfire",
                variable.name = "Feature",
                value.name   = "Value")

ggplot(df_long, aes(x = Value, fill = Wildfire)) +
  geom_histogram(alpha = 0.6, bins = 40, position = "identity") +
  facet_wrap(~ Feature, scales = "free", ncol = 5) +
  scale_fill_manual(values = c("No" = "#E41A1C", "Yes" = "#377EB8")) +
  labs(title = "Feature Distributions by Wildfire Class") +
  theme_minimal(base_size = 9)

# -----------------------------------------------------------------------------
# 4. Prepare 75-day sequences
# -----------------------------------------------------------------------------
FILL_VALUE <- 32767.0
SEQ_LEN    <- 75L

# Remove fill-value rows
mask <- !apply(df, 1, function(row) any(row == FILL_VALUE, na.rm = TRUE))
df   <- df[mask]

# Sort and assign sequence IDs (every 75 consecutive rows = one sequence)
setorder(df, latitude, longitude, datetime)
df[, seq_id := (.I - 1L) %/% SEQ_LEN]

# Build 3-D array: (n_seq, 75, 15)
groups <- split(df, by = "seq_id", keep.by = FALSE)

seqs <- simplify2array(
  lapply(groups, function(g) as.matrix(g[, ..FEATURES]))
)
# simplify2array produces (75, 15, n_seq) → transpose to (n_seq, 75, 15)
seqs <- aperm(seqs, c(3L, 1L, 2L))

# Label: 1 if any wildfire in the 75-day block, 0 otherwise
labels <- vapply(groups,
                 function(g) as.integer(any(g$Wildfire == "Yes")),
                 integer(1L))

cat("Sequences:", paste(dim(seqs), collapse = " x "),
    " Labels:", length(labels), "\n")

# -----------------------------------------------------------------------------
# 4.5 prepare wide
# -----------------------------------------------------------------------------
flatten_sequence <- function(mat) {
  anchor  <- 61L
  offsets <- seq_len(nrow(mat)) - anchor
  
  static_cols  <- c("latitude", "longitude", "datetime")
  dynamic_cols <- setdiff(colnames(mat), static_cols)
  
  # Single values for lat/lon (take from anchor row)
  static_part  <- mat[anchor, ..static_cols]
  
  # Expand dynamic columns over time
  dynamic_part <- mat[, ..dynamic_cols]
  col_names    <- as.vector(outer(dynamic_cols, offsets,
                                  function(var, t) sprintf("%s_t%+d", var, t)))
  
  dynamic_flat <- matrix(as.vector(t(dynamic_part)), nrow = 1,
                         dimnames = list(NULL, col_names))
  
  cbind(static_part, dynamic_flat)
}

flat_list <- lapply(groups, flatten_sequence)
flat_df   <- as.data.table(do.call(rbind, flat_list))



# -----------------------------------------------------------------------------
# 5. Train / Test split (80 / 20, stratified)
# -----------------------------------------------------------------------------
set.seed(42)
train_idx <- createDataPartition(labels, p = 0.8, list = FALSE)[, 1]
test_idx  <- setdiff(seq_along(labels), train_idx)

X_train <- seqs[train_idx, , ]
y_train <- labels[train_idx]
X_test  <- seqs[test_idx,  , ]
y_test  <- labels[test_idx]

cat("Train sequences:", nrow(X_train),
    "  Positives:", sum(y_train), "/", length(y_train), "\n")
cat("Test sequences: ", nrow(X_test),
    "  Positives:", sum(y_test),  "/", length(y_test),  "\n")

# -----------------------------------------------------------------------------
# 6. LSTM Baseline Model
# -----------------------------------------------------------------------------

# --- class weights (balanced) -----------------------------------------------
n_neg    <- sum(y_train == 0)
n_pos    <- sum(y_train == 1)
n_total  <- length(y_train)
w_neg    <- n_total / (2 * n_neg)
w_pos    <- n_total / (2 * n_pos)

sample_weights <- ifelse(y_train == 1, w_pos, w_neg)

# --- build model -------------------------------------------------------------
input_dim  <- length(FEATURES)  # 15
hidden_dim <- 64L
num_layers <- 2L
output_dim <- 2L

model <- keras_model_sequential(name = "LSTMClassifier") |>
  layer_lstm(units = hidden_dim, return_sequences = TRUE,
             input_shape = c(SEQ_LEN, input_dim)) |>
  layer_lstm(units = hidden_dim, return_sequences = FALSE) |>
  layer_dropout(rate = 0.3) |>
  layer_dense(units = output_dim, activation = "softmax")

model |> compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),
  loss      = "sparse_categorical_crossentropy",
  metrics   = list("accuracy")
)

summary(model)

# -----------------------------------------------------------------------------
# 7. Training
# -----------------------------------------------------------------------------
epochs     <- 5L
batch_size <- 64L

history <- model |> fit(
  x               = X_train,
  y               = y_train,
  epochs          = epochs,
  batch_size      = batch_size,
  sample_weight   = sample_weights,
  validation_split = 0.1,
  verbose         = 1
)

# Plot training history
plot(history) +
  labs(title = "LSTM Training History") +
  theme_minimal()

# -----------------------------------------------------------------------------
# 8. Evaluation
# -----------------------------------------------------------------------------
probs  <- predict(model, X_test, batch_size = batch_size)
y_pred <- max.col(probs) - 1L          # 0-indexed: 0 = No, 1 = Yes

# Confusion matrix
cm <- confusionMatrix(
  factor(y_pred, levels = c(0, 1)),
  factor(y_test, levels = c(0, 1)),
  positive = "1"
)
print(cm)

# Detailed per-class metrics (mirrors sklearn classification_report)
classes <- c("No (0)", "Yes (1)")
for (cls in c("0", "1")) {
  tp <- sum(y_pred == cls & y_test == cls)
  fp <- sum(y_pred == cls & y_test != cls)
  fn <- sum(y_pred != cls & y_test == cls)
  prec   <- if ((tp + fp) > 0) tp / (tp + fp) else 0
  rec    <- if ((tp + fn) > 0) tp / (tp + fn) else 0
  f1     <- if ((prec + rec) > 0) 2 * prec * rec / (prec + rec) else 0
  supp   <- sum(y_test == cls)
  cat(sprintf("Class %s  |  Precision: %.3f  Recall: %.3f  F1: %.3f  Support: %d\n",
              cls, prec, rec, f1, supp))
}
acc <- mean(y_pred == y_test)
cat(sprintf("Overall Accuracy: %.3f\n", acc))
