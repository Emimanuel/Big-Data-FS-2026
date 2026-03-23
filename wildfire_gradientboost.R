# -----------------------------------------------------------------------------
# 0. Libraries
# -----------------------------------------------------------------------------
library(data.table)   # fast I/O and data manipulation
library(ggplot2)      # visualisation
library(caret)        # createDataPartition / confusionMatrix
library(keras3)       # LSTM via TensorFlow/Keras
library(future.apply)
library(here)
library(tidymodels)
library(xgboost)
library(vip)


flat_df <- readRDS(here("data/intermediate/wildfire_cleaned_flat.rds"))

model_df <- flat_df |>
  mutate(Wildfire = factor(Wildfire, levels = c("No", "Yes"))) |>
  select(-seq_id, -latitude, -longitude, -datetime)


##train test split
set.seed(1)
split <- initial_split(model_df, prop = 0.8, strata = Wildfire)
train <- training(split)
test  <- testing(split)

##preprocessing
rec <- recipe(Wildfire ~ ., data = train) |>
  step_zv(all_predictors()) |>        # remove zero-variance columns
  step_impute_median(all_numeric_predictors())  # handle any NAs

##model
xgb_spec <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  min_n          = tune(),
  loss_reduction = tune(),
  sample_size    = tune(),
  mtry           = tune()
) |>
  set_engine("xgboost", nthread = parallel::detectCores() - 1) |>
  set_mode("classification")


##Workflow
xgb_wf <- workflow() |>
  add_recipe(rec) |>
  add_model(xgb_spec)

##CV
# Given class imbalance likely in wildfire data, use stratified CV
folds <- vfold_cv(train, v = 10, strata = Wildfire)

# Latin hypercube grid — efficient for 7 params
set.seed(42)
xgb_grid <- grid_space_filling(
  trees(range = c(200, 1000)),
  tree_depth(range = c(3, 8)),
  learn_rate(range = c(-2, -1)),   # log10 scale: 0.01 to 0.1
  min_n(range = c(5, 30)),
  loss_reduction(),
  sample_size = sample_prop(range = c(0.5, 1.0)),
  mtry(range = c(50, 300)),
  size = 30
)

# Use ROC-AUC given binary outcome + likely imbalance
doParallel::registerDoParallel()

tune_res <- tune_grid(
  xgb_wf,
  resamples = folds,
  grid      = xgb_grid,
  metrics   = metric_set(roc_auc, pr_auc, f_meas),
  control   = control_grid(save_pred = TRUE, verbose = TRUE)
)


##select
best_params <- select_best(tune_res, metric = "roc_auc")

final_wf <- finalize_workflow(xgb_wf, best_params)
final_fit <- last_fit(final_wf, split)


# Metrics
collect_metrics(final_fit)

# Confusion matrix
collect_predictions(final_fit) |>
  conf_mat(truth = Wildfire, estimate = .pred_class)

# ROC curve
collect_predictions(final_fit) |>
  roc_curve(Wildfire, .pred_Yes) |>
  autoplot()


# Fit list
final_fit |>
  extract_fit_parsnip() |>
  vip(num_features = 30)


