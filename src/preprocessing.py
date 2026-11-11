"""Data Preprocessing Module for EV Dataset."""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Removes duplicate rows from the DataFrame.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Tuple of (deduplicated DataFrame, count of dropped duplicates).
    """
    initial_len = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    dropped = initial_len - len(df_clean)
    return df_clean, dropped


def impute_missing_values(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Imputes missing numerical values using column medians (robust to skewness).
    
    Args:
        df: Input DataFrame.
        columns: List of columns to impute.
        
    Returns:
        Tuple of (imputed DataFrame, dictionary of imputation medians).
    """
    df_imputed = df.copy()
    medians = {}
    for col in columns:
        if col in df_imputed.columns and df_imputed[col].isnull().sum() > 0:
            median_val = float(df_imputed[col].median())
            df_imputed[col] = df_imputed[col].fillna(median_val)
            medians[col] = round(median_val, 2)
    return df_imputed, medians


def detect_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, Any]]:
    """Detects outliers using the Interquartile Range (IQR) rule:
    [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR].
    
    Args:
        df: Input DataFrame.
        columns: List of numerical columns to audit.
        
    Returns:
        Dictionary mapping column names to IQR bounds and outlier counts.
    """
    outlier_audit = {}
    for col in columns:
        if col in df.columns:
            series = df[col].dropna()
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_count = int(outlier_mask.sum())
            outlier_audit[col] = {
                "Q1": round(q25, 2),
                "Q3": round(q75, 2),
                "IQR": round(iqr, 2),
                "Lower_Bound": round(lower_bound, 2),
                "Upper_Bound": round(upper_bound, 2),
                "Outlier_Count": outlier_count
            }
    return outlier_audit


def preprocess_pipeline(df_raw: pd.DataFrame, impute_cols: List[str], outlier_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Executes the full end-to-end data preprocessing pipeline.
    
    Args:
        df_raw: Raw ingested DataFrame.
        impute_cols: Columns for median imputation.
        outlier_cols: Columns for IQR outlier auditing.
        
    Returns:
        Tuple of (cleaned DataFrame, metadata dictionary of audit results).
    """
    # 1. Deduplication
    df_dedup, dropped = remove_duplicates(df_raw)
    
    # 2. Imputation
    df_clean, medians = impute_missing_values(df_dedup, impute_cols)
    
    # 3. Outlier Audit
    outlier_audit = detect_outliers_iqr(df_clean, outlier_cols)
    
    meta = {
        "dropped_duplicates": dropped,
        "imputed_medians": medians,
        "outlier_audit": outlier_audit
    }
    return df_clean, meta
