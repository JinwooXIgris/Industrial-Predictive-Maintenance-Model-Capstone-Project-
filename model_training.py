"""
model_training.py
Module for model initialization, stratified 5-fold cross-validation, hyperparameter
tuning via GridSearchCV (optimizing for F1-Score), and model serialization.
"""

import os
import sys
import pickle
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, classification_report

# Import project configurations
try:
    import config
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config

# Gracefully check for XGBoost availability
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("[INFO] XGBoost is available and will be included in training.")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost is NOT installed. Skipping XGBoost; using HistGradientBoostingClassifier for boosting.")


class ModelTuner:
    """
    Orchestrates the training, hyperparameter tuning, and saving of machine
    learning models for predictive maintenance.
    """
    def __init__(self):
        self.cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        self.models: Dict[str, Any] = {}
        self.grids: Dict[str, Dict[str, Any]] = {}
        self.best_estimators: Dict[str, Any] = {}
        self.tuning_results: Dict[str, Dict[str, Any]] = {}
        
        self._initialize_models_and_grids()

    def _initialize_models_and_grids(self) -> None:
        """
        Sets up the model configurations and corresponding hyperparameter search grids.
        """
        # 1. Logistic Regression
        self.models["logistic_regression"] = LogisticRegression(
            random_state=config.RANDOM_STATE,
            class_weight="balanced"
        )
        self.grids["logistic_regression"] = config.LOGISTIC_REGRESSION_GRID
        
        # 2. Decision Tree
        self.models["decision_tree"] = DecisionTreeClassifier(
            random_state=config.RANDOM_STATE,
            class_weight="balanced"
        )
        self.grids["decision_tree"] = config.DECISION_TREE_GRID
        
        # 3. Random Forest
        self.models["random_forest"] = RandomForestClassifier(
            random_state=config.RANDOM_STATE,
            class_weight="balanced"
        )
        self.grids["random_forest"] = config.RANDOM_FOREST_GRID
        
        # 4. HistGradientBoosting (native scikit-learn, supports class weights via sample_weight)
        self.models["hist_gradient_boosting"] = HistGradientBoostingClassifier(
            random_state=config.RANDOM_STATE
        )
        self.grids["hist_gradient_boosting"] = config.HIST_GRADIENT_BOOSTING_GRID
        
        # 5. XGBoost (conditional implementation)
        if XGBOOST_AVAILABLE:
            # Note: We can handle class imbalance in XGBoost using scale_pos_weight
            self.models["xgboost"] = XGBClassifier(
                random_state=config.RANDOM_STATE,
                use_label_encoder=False,
                eval_metric="logloss"
            )
            self.grids["xgboost"] = config.XGBOOST_GRID

    def tune_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Executes GridSearchCV for each model on the training set, optimizing for F1-Score.
        """
        for model_name, model in self.models.items():
            grid_params = self.grids[model_name]
            print(f"\n[INFO] Starting Grid Search for {model_name}...")
            print(f"   Parameter Grid: {grid_params}")
            
            # Initialize GridSearchCV optimizing F1-Score
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=grid_params,
                scoring="f1",
                cv=self.cv,
                n_jobs=-1,
                verbose=1
            )
            
            # Fit Grid Search
            grid_search.fit(X_train, y_train)
            
            self.best_estimators[model_name] = grid_search.best_estimator_
            self.tuning_results[model_name] = {
                "best_params": grid_search.best_params_,
                "best_cv_f1": grid_search.best_score_,
                "cv_results": grid_search.cv_results_
            }
            
            print(f"[SUCCESS] {model_name} Tuning Complete!")
            print(f"   Best Parameters: {grid_search.best_params_}")
            print(f"   Best Cross-Validation F1-Score: {grid_search.best_score_:.4f}")
            
        return self.best_estimators

    def save_best_models(self, folder_path: str) -> None:
        """
        Serializes best models to pickle files.
        """
        os.makedirs(folder_path, exist_ok=True)
        for model_name, model in self.best_estimators.items():
            file_path = os.path.join(folder_path, f"best_{model_name}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(model, f)
            print(f"[INFO] Saved best model: {file_path}")
            
        # Also save the training/tuning summary as a JSON or text file
        summary_path = os.path.join(config.RESULTS_DIR, "tuning_summary.txt")
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        with open(summary_path, "w") as f:
            f.write("Model Tuning Summary\n")
            f.write("====================\n")
            for model_name, res in self.tuning_results.items():
                f.write(f"\nModel: {model_name}\n")
                f.write(f"Best CV F1-Score: {res['best_cv_f1']:.4f}\n")
                f.write(f"Best Parameters: {res['best_params']}\n")
        print(f"[INFO] Tuning summary written to: {summary_path}")


if __name__ == "__main__":
    from data_loader import load_and_validate_data, split_data
    from feature_engineering import prepare_features_pipeline
    
    try:
        # Load and split
        df = load_and_validate_data(config.DATA_PATH)
        train, val, test = split_data(df)
        
        # Preprocess features
        X_tr, y_tr, X_v, y_v, X_te, y_te, feats = prepare_features_pipeline(train, val, test, save_artifacts=False)
        
        # Instantiate tuner and run on a small subset for quick local test verification
        print("\n[INFO] Running dry-run verification on a subset of the dataset...")
        tuner = ModelTuner()
        
        # Take a subset to verify execution speed
        X_tr_sub = X_tr.iloc[:500]
        y_tr_sub = y_tr.iloc[:500]
        
        # Modify grids for ultra-fast dry-run
        tuner.grids["logistic_regression"] = {"C": [1.0]}
        tuner.grids["decision_tree"] = {"max_depth": [5]}
        tuner.grids["random_forest"] = {"n_estimators": [10], "max_depth": [5]}
        tuner.grids["hist_gradient_boosting"] = {"max_iter": [10], "max_leaf_nodes": [15]}
        if "xgboost" in tuner.models:
            tuner.grids["xgboost"] = {"n_estimators": [10], "max_depth": [3]}
            
        tuner.tune_models(X_tr_sub, y_tr_sub)
        print("[SUCCESS] Model training dry-run completed successfully!")
    except Exception as e:
        print(f"[ERROR] Model training verification failed: {e}")
