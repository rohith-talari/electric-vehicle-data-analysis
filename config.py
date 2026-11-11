"""Centralized Configuration for Electric Vehicle Data Analysis & ML Project.

Task 1: Group Discussion and Demonstration Method on Data Import, Exploration and Statistical Analysis
Revised Bloom's Taxonomy Level: K2 (Understand), K3 (Apply)
"""

from pathlib import Path

# Base Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"
SRC_DIR = PROJECT_ROOT / "src"

# Data File Paths
RAW_DATA_PATH = DATA_DIR / "ev_dataset.csv"
CLEANED_DATA_PATH = DATA_DIR / "ev_cleaned_dataset.csv"

# Artifact File Paths
STATISTICAL_SUMMARY_PATH = ARTIFACTS_DIR / "statistical_summary.csv"
COVARIANCE_MATRIX_PATH = ARTIFACTS_DIR / "covariance_matrix.csv"
CORRELATION_MATRIX_PATH = ARTIFACTS_DIR / "correlation_matrix.csv"

# Report File Paths
REPORT_PDF_PATH = REPORTS_DIR / "EV_Data_Analysis_Project_Report.pdf"

# Feature Column Groups
NUMERICAL_COLS = [
    "Battery_Capacity_kWh",
    "Electric_Range_Miles",
    "Base_MSRP_USD",
    "Charging_Time_Hours",
    "Energy_Consumption_Wh_km",
    "Acceleration_0_100_s",
    "Top_Speed_kmh",
]

CATEGORICAL_COLS = [
    "Make",
    "Model",
    "EV_Type",
    "CAFV_Eligibility",
]

ENGINEERED_COLS = [
    "Range_Efficiency_Miles_per_kWh",
    "Cost_per_Range_Mile_USD",
]

OUTLIER_TARGET_COLS = [
    "Battery_Capacity_kWh",
    "Electric_Range_Miles",
    "Base_MSRP_USD",
]

IMPUTE_MEDIAN_COLS = [
    "Battery_Capacity_kWh",
    "Base_MSRP_USD",
    "Charging_Time_Hours",
]

# Random Seed for Reproducibility
RANDOM_SEED = 42
