"""
data_loader.py
Module for loading the predictive maintenance dataset, validating its integrity,
and performing stratified train/validation/test splits.
"""

import os
import sys
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

# Import project configurations
try:
    import config
except ModuleNotFoundError:
    # Handle paths when executing from different directories
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config


def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """
    Loads the predictive maintenance CSV file and performs rigorous data integrity checks.
    
    Parameters:
    -----------
    filepath : str
        Absolute or relative path to the CSV file.
        
    Returns:
    --------
    pd.DataFrame
        Cleaned and validated DataFrame.
        
    Raises:
    -------
    FileNotFoundError
        If the file does not exist at the specified path.
    ValueError
        If key expected columns are missing or if the dataset structure is invalid.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at path: {filepath}. Please verify config.py path.")

    print(f"[INFO] Loading dataset from: {filepath}...")
    df = pd.read_csv(filepath)
    
    # 1. Validate Shape
    print(f"[INFO] Initial dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    if df.empty:
        raise ValueError("Dataset is empty.")
    
    # 2. Check Expected Columns
    expected_cols = [config.RAW_TARGET, config.RAW_CATEGORICAL_FEATURE] + config.RAW_NUMERIC_FEATURES
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # 3. Check for Missing Values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("[WARNING] Missing values detected during validation:")
        print(null_counts[null_counts > 0])
        # Note: According to foundational analysis, this dataset typically has no missing values.
        # But we handle it robustly by printing a warning.
    else:
        print("[INFO] No missing values detected.")

    # 4. Check for Duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"[WARNING] Detected {duplicate_count} duplicate rows. Removing duplicates...")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("[INFO] No duplicate rows detected.")

    # 5. Rename Columns to Clean Snake Case
    rename_mapping = {
        config.RAW_CATEGORICAL_FEATURE: config.CLEAN_CATEGORICAL_FEATURE,
        config.RAW_TARGET: config.CLEAN_TARGET
    }
    # Add mapping for numeric columns
    for raw_name, clean_name in zip(config.RAW_NUMERIC_FEATURES, config.CLEAN_NUMERIC_FEATURES):
        rename_mapping[raw_name] = clean_name
        
    df = df.rename(columns=rename_mapping)
    
    # 6. Drop Irrelevant ID columns if they exist
    cols_to_drop = ["UDI", "Product ID", config.FAILURE_TYPE_COL]
    existing_drops = [col for col in cols_to_drop if col in df.columns]
    if existing_drops:
        print(f"[INFO] Dropping non-predictive columns: {existing_drops}")
        df = df.drop(columns=existing_drops)

    # 7. Convert machine_type to categorical
    df[config.CLEAN_CATEGORICAL_FEATURE] = df[config.CLEAN_CATEGORICAL_FEATURE].astype("category")
    
    # 8. Check target class distribution
    class_counts = df[config.CLEAN_TARGET].value_counts()
    class_pct = df[config.CLEAN_TARGET].value_counts(normalize=True) * 100
    print(f"[INFO] Target Class Distribution:")
    for cls, count in class_counts.items():
        print(f"   Class {cls}: {count} occurrences ({class_pct[cls]:.2f}%)")

    return df


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into stratified Train (70%), Validation (15%), and Test (15%) sets.
    Stratification ensures the rare failure target distribution is identical across splits.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The preprocessed and validated dataset.
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        train_df, val_df, test_df
    """
    target = config.CLEAN_TARGET
    
    # First split: Train (70%) and Temp (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size=(config.TEST_SIZE + config.VAL_SIZE),
        random_state=config.RANDOM_STATE,
        stratify=df[target]
    )
    
    # Second split: Split Temp into Validation (50% of Temp = 15%) and Test (50% of Temp = 15%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=config.RANDOM_STATE,
        stratify=temp_df[target]
    )
    
    # Verify split proportions and target ratios
    print("\n[INFO] Data Splitting Summary:")
    for name, dataset in [("Train Set", train_df), ("Val Set", val_df), ("Test Set", test_df)]:
        failures = dataset[target].sum()
        pct = (len(dataset) / len(df)) * 100
        fail_pct = (failures / len(dataset)) * 100
        print(f"   {name:10s}: {len(dataset):4d} rows ({pct:.1f}%) | Failures: {failures:3d} ({fail_pct:.2f}%)")
        
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Test script execution
    try:
        data = load_and_validate_data(config.DATA_PATH)
        train, val, test = split_data(data)
        print("[SUCCESS] Data loader module executed successfully!")
    except Exception as e:
        print(f"[ERROR] Data loader module verification failed: {e}")
