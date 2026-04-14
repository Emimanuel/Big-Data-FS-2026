# -----------------------------------------------------------------------------
# 0. Libraries
# -----------------------------------------------------------------------------
library(ggplot2)
library(here)
library(tidymodels)
library(ranger)
library(vip)
library(future)
plan(multisession, workers = parallel::detectCores() - 1)

# -----------------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------------
set.seed(1)
flat_df <- readRDS(here("data/intermediate/wildfire_cleaned_flat.rds"))
#flat_df <- flat_df[sample.int(nrow(flat_df), 500), ]

model_df <- flat_df |>
  mutate(Wildfire = factor(Wildfire, levels = c("No", "Yes"))) |>
  select(-seq_id, -latitude, -longitude, -datetime)

# -----------------------------------------------------------------------------
# 2. Train/test split
# -----------------------------------------------------------------------------
set.seed(1)
split <- initial_split(model_df, prop = 0.8, strata = Wildfire)
train <- training(split)
test  <- testing(split)

# -----------------------------------------------------------------------------
# 3. Preprocessing
# -----------------------------------------------------------------------------
rec <- recipe(Wildfire ~ ., data = train) |>
  step_zv(all_predictors()) |>
  step_impute_median(all_numeric_predictors())

# -----------------------------------------------------------------------------
# 4. Model spec
# -----------------------------------------------------------------------------
rf_spec <- rand_forest(
  trees = tune(),
  mtry  = tune(),
  min_n = tune()
) |>
  set_engine(
    "ranger",
    importance      = "impurity",   # enables vip() later
    num.threads     = 1             # parallelism handled by future/doParallel
  ) |>
  set_mode("classification")

# -----------------------------------------------------------------------------
# 5. Workflow
# -----------------------------------------------------------------------------
rf_wf <- workflow() |>
  add_recipe(rec) |>
  add_model(rf_spec)

# -----------------------------------------------------------------------------
# 6. Cross-validation  (identical folds to XGB script)
# -----------------------------------------------------------------------------
folds <- vfold_cv(train, v = 10, strata = Wildfire)

# -----------------------------------------------------------------------------
# 7. Tuning grid
# -----------------------------------------------------------------------------
rf_grid <- grid_space_filling(
  trees(range = c(200, 1000)),
  mtry(range  = c(5, floor(ncol(train) * 0.8))),
  min_n(range = c(5, 30)),
  size = 30
)

# -----------------------------------------------------------------------------
# 8. Tune
# -----------------------------------------------------------------------------
doParallel::registerDoParallel()

tune_res <- tune_grid(
  rf_wf,
  resamples = folds,
  grid      = rf_grid,
  metrics   = metric_set(roc_auc, pr_auc, f_meas),
  control   = control_grid(save_pred = TRUE, verbose = TRUE, event_level = "second")
)

# -----------------------------------------------------------------------------
# 9. Finalize and evaluate
# ----------------------------------------------------

-------------------------
best_params <- select_best(tune_res, metric = "roc_auc")
final_wf    <- finalize_workflow(rf_wf, best_params)
final_fit   <- last_fit(final_wf, split)

# Metrics
collect_metrics(final_fit)

# Confusion matrix
collect_predictions(final_fit) |>
  conf_mat(truth = Wildfire, estimate = .pred_class)

# ROC curve
collect_predictions(final_fit) |>
  roc_curve(Wildfire, .pred_Yes, event_level = "second") |>
  autoplot()

# Variable importance
final_fit |>
  extract_fit_parsnip() |>
  vip(num_features = 30)