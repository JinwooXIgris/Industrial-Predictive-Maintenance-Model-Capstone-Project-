"""
config.py
Centralized configuration script defining paths, directory structures,
hyperparameter search grids, and global random seeds for reproducibility.
"""

import os

# -------------------------------------------------------------------------
# 1. PATHS AND DIRECTORY STRUCTURE
# -------------------------------------------------------------------------
# Absolute path to the raw dataset on the local machine
DATA_PATH = r"C:\Users\DELL\Documents\Projects\Notebooks\Capstone(Data Pioneers) files\predictive_maintenance.csv"

# Output directories for artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Ensure directories exist
for directory in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# -------------------------------------------------------------------------
# 2. SEED AND SPLIT CONSTANTS
# -------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # Out of the original dataset, leading to a 70/15/15 split
CV_FOLDS = 5

# Target columns
RAW_TARGET = "Target"
CLEAN_TARGET = "failure"
FAILURE_TYPE_COL = "Failure Type"

# Feature definitions
RAW_NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
CLEAN_NUMERIC_FEATURES = [
    "air_temp",
    "proc_temp",
    "rot_speed",
    "torque",
    "tool_wear"
]
RAW_CATEGORICAL_FEATURE = "Type"
CLEAN_CATEGORICAL_FEATURE = "machine_type"

# Engineered features
ENGINEERED_FEATURES = [
    "temp_diff",
    "power_proxy",
    "wear_per_speed"
]

# -------------------------------------------------------------------------
# 3. HYPERPARAMETER SEARCH GRIDS
# -------------------------------------------------------------------------
# Grid search parameters optimized for F1-Score

LOGISTIC_REGRESSION_GRID = {
    "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "penalty": ["l2"],
    "solver": ["liblinear", "lbfgs"],
    "max_iter": [1000]
}

DECISION_TREE_GRID = {
    "max_depth": [3, 5, 7, 10, 12, 15],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "criterion": ["gini", "entropy"]
}

RANDOM_FOREST_GRID = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [5, 10, 15, 20],
    "min_samples_split": [5, 10, 20],
    "min_samples_leaf": [2, 4, 8],
    "bootstrap": [True]
}

# Histogram Gradient Boosting (Scikit-Learn native version, no compilation required)
HIST_GRADIENT_BOOSTING_GRID = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [10, 20, 50],
    "max_iter": [100, 200],
    "l2_regularization": [0.0, 0.1, 1.0]
}

# Optional XGBoost parameters if the user installs xgboost
XGBOOST_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}
