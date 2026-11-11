"""Statistical Analysis and Probability Distribution Module for EV Dataset."""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def compute_central_tendency(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Computes Mean, Median, and Mode for numerical columns.
    
    Args:
        df: Input DataFrame.
        columns: List of numerical columns.
        
    Returns:
        DataFrame summarizing central tendency measures.
    """
    rows = []
    for col in columns:
        if col in df.columns:
            s = df[col].dropna()
            mode_s = s.mode()
            mode_val = float(mode_s.iloc[0]) if not mode_s.empty else np.nan
            rows.append({
                "Feature": col,
                "Mean": round(float(s.mean()), 2),
                "Median": round(float(s.median()), 2),
                "Mode": round(mode_val, 2)
            })
    return pd.DataFrame(rows)


def compute_dispersion_measures(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Computes Variance, Standard Deviation, Range, and IQR.
    
    Args:
        df: Input DataFrame.
        columns: List of numerical columns.
        
    Returns:
        DataFrame summarizing dispersion metrics.
    """
    rows = []
    for col in columns:
        if col in df.columns:
            s = df[col].dropna()
            var_val = float(s.var(ddof=1))
            std_val = float(s.std(ddof=1))
            range_val = float(s.max() - s.min())
            q25 = float(s.quantile(0.25))
            q75 = float(s.quantile(0.75))
            iqr_val = q75 - q25
            rows.append({
                "Feature": col,
                "Variance": round(var_val, 2),
                "Std_Dev": round(std_val, 2),
                "Range": round(range_val, 2),
                "IQR": round(iqr_val, 2)
            })
    return pd.DataFrame(rows)


def compute_covariance_matrix(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Computes the sample covariance matrix for specified features.
    
    Args:
        df: Input DataFrame.
        columns: List of numerical columns.
        
    Returns:
        DataFrame representing the covariance matrix.
    """
    valid_cols = [c for c in columns if c in df.columns]
    return df[valid_cols].cov().round(2)


def compute_correlation_matrix(df: pd.DataFrame, columns: List[str], method: str = "pearson") -> pd.DataFrame:
    """Computes the pairwise correlation matrix (Pearson or Spearman).
    
    Args:
        df: Input DataFrame.
        columns: List of numerical columns.
        method: Correlation method ('pearson', 'spearman').
        
    Returns:
        DataFrame representing the correlation matrix.
    """
    valid_cols = [c for c in columns if c in df.columns]
    return df[valid_cols].corr(method=method).round(3)


def compute_probability_distributions(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Computes Skewness, Kurtosis, and Shapiro-Wilk Normality test statistics.
    
    Args:
        df: Input DataFrame.
        columns: List of numerical columns.
        
    Returns:
        DataFrame summarizing distribution shape and normality assessment.
    """
    rows = []
    for col in columns:
        if col in df.columns:
            s = df[col].dropna()
            skew_val = float(stats.skew(s))
            kurt_val = float(stats.kurtosis(s))
            # Test normality on sample (up to 500 records)
            shapiro_stat, shapiro_p = stats.shapiro(s[:500])
            is_normal = "Normal (p>0.05)" if shapiro_p > 0.05 else "Non-Normal (p<=0.05)"
            rows.append({
                "Feature": col,
                "Skewness": round(skew_val, 3),
                "Kurtosis": round(kurt_val, 3),
                "Shapiro_Stat": round(float(shapiro_stat), 4),
                "P_Value": round(float(shapiro_p), 5),
                "Normality": is_normal
            })
    return pd.DataFrame(rows)


def generate_statistical_report(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Consolidates all statistical measures into an integrated summary table.
    
    Args:
        df: Input DataFrame.
        columns: List of numerical columns.
        
    Returns:
        Unified DataFrame combining central tendency, dispersion, and shape.
    """
    ct = compute_central_tendency(df, columns)
    disp = compute_dispersion_measures(df, columns)
    dist = compute_probability_distributions(df, columns)
    
    merged = ct.merge(disp, on="Feature").merge(dist[["Feature", "Skewness", "Kurtosis", "Normality"]], on="Feature")
    return merged
