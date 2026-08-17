"""
feature_engineering.py
Module for feature engineering, scaling, categorical encoding, and class
imbalance handling using SMOTE. Prevents data leakage by fitting transformers
only on the training set.
"""

import os
import sys
import pickle
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

# Import project configurations
try:
    import config
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features based on physical principles of mechanical failure.
    
    Business Rationale:
    1. Temperature Difference (temp_diff):
       Calculated as proc_temp - air_temp. If heat dissipation is low,
       the temperature differential increases, indicating higher thermal stress
       and risking Heat Dissipation Failure (HDF).
    2. Power Proxy (power_proxy):
       Calculated as rotational_speed * torque. Proportional to the mechanical power
       exerted on the machine. High power can indicate excessive mechanical stress
       associated with Power Failure (PWF).
    3. Tool Wear Rate (wear_per_speed):
       Calculated as tool_wear * rotational_speed. Captures high frictional stress
       where high speed is coupled with advanced tool wear, leading to Tool Wear Failure (TWF).
    """
    # Create a deep copy to avoid modifying original dataframe
    df = df.copy()
    
    # 1. Thermal stress indicator
    df["temp_diff"] = df["proc_temp"] - df["air_temp"]
    
    # 2. Power proxy (Torque * rotational speed is proportional to power)
    # P = Torque * (2 * pi * RPM / 60) -> we skip constants as they scale linearly
    df["power_proxy"] = df["rot_speed"] * df["torque"]
    
    # 3. Tool wear rate proxy
    df["wear_per_speed"] = df["tool_wear"] * df["rot_speed"]
    
    return df


class FeaturePipeline:
    """
    A production-grade pipeline class to scale numerical columns,
    encode categorical columns, and persist fitted transformers.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.feature_names: List[str] = []

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Fits scaling and encoding transformers on the training dataset.
        """
        # Create physical features
        processed_train = engineer_features(train_df)
        
        # 1. Fit Numeric Scaler
        numeric_cols = config.CLEAN_NUMERIC_FEATURES + config.ENGINEERED_FEATURES
        self.scaler.fit(processed_train[numeric_cols])
        
        # 2. Fit Categorical Encoder
        cat_cols = [config.CLEAN_CATEGORICAL_FEATURE]
        self.encoder.fit(processed_train[cat_cols])
        
        # 3. Store Feature Names for downstream interpretability
        cat_feature_names = self.encoder.get_feature_names_out(cat_cols).tolist()
        self.feature_names = numeric_cols + cat_feature_names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw data using the fitted scaler and encoder.
        """
        processed_df = engineer_features(df)
        
        # Scale numeric features
        numeric_cols = config.CLEAN_NUMERIC_FEATURES + config.ENGINEERED_FEATURES
        scaled_numeric = self.scaler.transform(processed_df[numeric_cols])
        
        # Encode categorical features
        cat_cols = [config.CLEAN_CATEGORICAL_FEATURE]
        encoded_cat = self.encoder.transform(processed_df[cat_cols])
        
        # Combine scaled numeric and encoded categorical features
        combined_features = np.hstack([scaled_numeric, encoded_cat])
        
        # Convert back to DataFrame to preserve feature names
        return pd.DataFrame(combined_features, columns=self.feature_names)

    def save(self, folder_path: str) -> None:
        """
        Saves the fitted pipeline transformers to pickle files.
        """
        os.makedirs(folder_path, exist_ok=True)
        
        scaler_path = os.path.join(folder_path, "scaler.pkl")
        encoder_path = os.path.join(folder_path, "encoder.pkl")
        pipeline_meta_path = os.path.join(folder_path, "pipeline_meta.pkl")
        
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(encoder_path, "wb") as f:
            pickle.dump(self.encoder, f)
        with open(pipeline_meta_path, "wb") as f:
            pickle.dump(self.feature_names, f)
            
        print(f"[INFO] Feature pipeline artifacts saved to: {folder_path}")


def balance_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies SMOTE (Synthetic Minority Over-sampling Technique) to resolve
    class imbalance in the training dataset.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Preprocessed training features.
    y_train : pd.Series
        Training targets.
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        SMOTE-balanced training features and labels.
    """
    print(f"[INFO] Class balance before SMOTE: {np.bincount(y_train)}")
    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"[INFO] Class balance after SMOTE: {np.bincount(y_resampled)}")
    
    # Restore as DataFrame to keep column names
    X_resampled_df = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled_series = pd.Series(y_resampled, name=y_train.name)
    
    return X_resampled_df, y_resampled_series


def prepare_features_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_artifacts: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """
    Orchestrates the entire feature engineering, scaling, encoding, and SMOTE balancing workflow.
    Fits transformers on train data only, saving artifacts, and returning processed splits.
    """
    target_col = config.CLEAN_TARGET
    
    # Extract labels
    y_train = train_df[target_col].copy()
    y_val = val_df[target_col].copy()
    y_test = test_df[target_col].copy()
    
    # Initialize and fit FeaturePipeline on training data
    pipeline = FeaturePipeline()
    pipeline.fit(train_df)
    
    # Transform all datasets
    X_train = pipeline.transform(train_df)
    X_val = pipeline.transform(val_df)
    X_test = pipeline.transform(test_df)
    
    # Save pipeline artifacts
    if save_artifacts:
        pipeline.save(config.MODELS_DIR)
        
    # Balance training data only (leave validation and test sets completely pristine)
    X_train_bal, y_train_bal = balance_training_data(X_train, y_train)
    
    return X_train_bal, y_train_bal, X_val, y_val, X_test, y_test, pipeline.feature_names


if __name__ == "__main__":
    from data_loader import load_and_validate_data, split_data
    try:
        # Load and split
        df = load_and_validate_data(config.DATA_PATH)
        train, val, test = split_data(df)
        
        # Pipeline execution
        X_tr, y_tr, X_v, y_v, X_te, y_te, feats = prepare_features_pipeline(train, val, test)
        print(f"[SUCCESS] Feature engineering pipeline executed successfully!")
        print(f"Features: {feats}")
    except Exception as e:
        print(f"[ERROR] Feature engineering module verification failed: {e}")
