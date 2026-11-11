"""Feature Engineering Module for EV Dataset."""

from typing import List
import numpy as np
import pandas as pd


def compute_range_efficiency(df: pd.DataFrame) -> pd.Series:
    """Computes Range Efficiency (Miles per kWh of battery capacity).
    
    Args:
        df: DataFrame containing Electric_Range_Miles and Battery_Capacity_kWh.
        
    Returns:
        Series of computed efficiency values.
    """
    return (df["Electric_Range_Miles"] / df["Battery_Capacity_kWh"]).round(2)


def compute_cost_per_range_mile(df: pd.DataFrame) -> pd.Series:
    """Computes Cost per Mile of Range (USD / Mile).
    
    Args:
        df: DataFrame containing Base_MSRP_USD and Electric_Range_Miles.
        
    Returns:
        Series of computed cost per mile values.
    """
    return (df["Base_MSRP_USD"] / df["Electric_Range_Miles"]).round(2)


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Appends engineered domain features to the EV DataFrame.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        DataFrame augmented with new engineered features.
    """
    df_feat = df.copy()
    df_feat["Range_Efficiency_Miles_per_kWh"] = compute_range_efficiency(df_feat)
    df_feat["Cost_per_Range_Mile_USD"] = compute_cost_per_range_mile(df_feat)
    return df_feat


def standardize_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Standardizes specified numerical features using Z-score normalization:
    Z = (X - mean) / std.
    
    Args:
        df: Input DataFrame.
        columns: List of columns to standardize.
        
    Returns:
        DataFrame with standardized columns.
    """
    df_scaled = df.copy()
    for col in columns:
        if col in df_scaled.columns:
            mean = df_scaled[col].mean()
            std = df_scaled[col].std(ddof=1)
            if std > 0:
                df_scaled[f"{col}_ZScore"] = ((df_scaled[col] - mean) / std).round(4)
    return df_scaled
