"""Data Loading and Initial Exploration Module for EV Dataset."""

from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Loads dataset from CSV file path.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        Loaded pandas DataFrame.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    return df


def inspect_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspects dataset structure, dimensions, and data types.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Dictionary containing shape, columns, and data types.
    """
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates missing values count and percentage per column.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Summary DataFrame of missing counts and percentages.
    """
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    summary = pd.DataFrame({
        "Missing_Count": null_counts,
        "Percentage": null_pct.round(2)
    })
    return summary[summary["Missing_Count"] > 0]


def check_duplicates(df: pd.DataFrame) -> int:
    """Counts duplicate rows in the DataFrame.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Number of duplicate rows.
    """
    return int(df.duplicated().sum())
