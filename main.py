"""ELECTRIC VEHICLE DATA ANALYSIS & MACHINE LEARNING
Unified Command-Line Interface (CLI) Entrypoint.

Task 1: Group Discussion and Demonstration Method on Data Import, Exploration and Statistical Analysis
Revised Bloom's Taxonomy Level: K2 (Understand), K3 (Apply)

Usage:
    python main.py --all          # Run complete end-to-end pipeline
    python main.py --explore      # Ingest dataset and audit schema
    python main.py --preprocess   # Run cleaning, imputation, and feature engineering
    python main.py --stats        # Compute statistical measures & export matrix artifacts
    python main.py --visualize    # Generate all 6 publication figures
    python main.py --demo         # Display demonstration walkthrough & GD guide
    python main.py --viva         # Display viva-voce questions & answers (K2 & K3)
    python main.py --test         # Execute unit test suite
    python main.py --status       # Verify project readiness & artifact status
    python main.py --app          # Launch Streamlit web application
"""

import argparse
import logging
from pathlib import Path
import subprocess
import sys
import unittest

from config import (
    CLEANED_DATA_PATH,
    CORRELATION_MATRIX_PATH,
    COVARIANCE_MATRIX_PATH,
    DATA_DIR,
    FIGURES_DIR,
    IMPUTE_MEDIAN_COLS,
    NUMERICAL_COLS,
    OUTLIER_TARGET_COLS,
    RAW_DATA_PATH,
    REPORT_PDF_PATH,
    REPORTS_DIR,
    STATISTICAL_SUMMARY_PATH,
)
from src.data_loader import check_duplicates, check_missing_values, inspect_schema, load_data
from src.demonstration import (
    display_group_discussion_guide,
    display_viva_questions_and_answers,
    run_demonstration_walkthrough,
)
from src.feature_engineering import apply_feature_engineering
from src.preprocessing import preprocess_pipeline
from src.statistical_analysis import (
    compute_correlation_matrix,
    compute_covariance_matrix,
    generate_statistical_report,
)
from src.visualization import (
    plot_bivariate_relationship,
    plot_categorical_benchmarks,
    plot_correlation_heatmap,
    plot_energy_vs_acceleration,
    plot_probability_distribution_qq,
    plot_univariate_distributions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_status():
    """Checks and displays availability of data, models, reports, and artifacts."""
    print("=" * 70)
    print("  ELECTRIC VEHICLE DATA ANALYSIS & ML - SYSTEM STATUS")
    print("=" * 70)
    
    artifacts = {
        "Raw Dataset (data/ev_dataset.csv)": RAW_DATA_PATH.exists(),
        "Cleaned Dataset (data/ev_cleaned_dataset.csv)": CLEANED_DATA_PATH.exists(),
        "Statistical Summary (artifacts/statistical_summary.csv)": STATISTICAL_SUMMARY_PATH.exists(),
        "Covariance Matrix (artifacts/covariance_matrix.csv)": COVARIANCE_MATRIX_PATH.exists(),
        "Correlation Matrix (artifacts/correlation_matrix.csv)": CORRELATION_MATRIX_PATH.exists(),
        "Project Report PDF (reports/EV_Data_Analysis_Project_Report.pdf)": REPORT_PDF_PATH.exists(),
    }
    
    for i in range(1, 7):
        fig_name = f"Figure {i} in reports/figures/"
        has_fig = any(FIGURES_DIR.glob(f"fig{i}_*.png"))
        artifacts[fig_name] = has_fig
        
    all_ready = True
    for item, exists in artifacts.items():
        status_tag = "[READY]  " if exists else "[MISSING]"
        if not exists:
            all_ready = False
        print(f"  {status_tag} {item}")
        
    print("-" * 70)
    if all_ready:
        print("  System Status: ALL PIPELINE ARTIFACTS VERIFIED & OPERATIONAL")
    else:
        print("  System Status: SOME ARTIFACTS MISSING. Run `python main.py --all`")
    print("=" * 70)


def run_explore():
    """Runs data loading and schema inspection."""
    logger.info("Executing Data Ingestion & Exploration...")
    df = load_data(RAW_DATA_PATH)
    schema = inspect_schema(df)
    print(f"\n[*] Dataset Shape: {schema['n_rows']} rows, {schema['n_cols']} columns")
    print("\n--- First 5 Records ---")
    print(df.head(5).to_string(index=False))
    
    missing = check_missing_values(df)
    print("\n--- Missing Values Audit ---")
    if not missing.empty:
        print(missing.to_string())
    else:
        print("Zero missing values detected.")
        
    dups = check_duplicates(df)
    print(f"\n[*] Duplicate Records: {dups}")
    return df


def run_preprocess(df=None):
    """Executes data preprocessing and feature engineering."""
    logger.info("Executing Preprocessing & Feature Engineering...")
    if df is None:
        df = load_data(RAW_DATA_PATH)
        
    df_clean, meta = preprocess_pipeline(df, IMPUTE_MEDIAN_COLS, OUTLIER_TARGET_COLS)
    df_clean = apply_feature_engineering(df_clean)
    
    # Save cleaned data
    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(CLEANED_DATA_PATH, index=False)
    
    print(f"\n[+] Dropped Duplicates: {meta['dropped_duplicates']}")
    print(f"[+] Imputed Medians: {meta['imputed_medians']}")
    print("\n--- Outlier Audit (IQR Method) ---")
    for col, data in meta['outlier_audit'].items():
        print(f"  {col}: Bounds [{data['Lower_Bound']}, {data['Upper_Bound']}] | Outliers: {data['Outlier_Count']}")
        
    print(f"\n[+] Cleaned dataset saved to: {CLEANED_DATA_PATH}")
    return df_clean


def run_stats(df=None):
    """Computes statistical measures and exports artifacts."""
    logger.info("Computing Statistical Measures & Matrices...")
    if df is None:
        if CLEANED_DATA_PATH.exists():
            df = load_data(CLEANED_DATA_PATH)
        else:
            df = run_preprocess()
            
    cols = NUMERICAL_COLS + ["Range_Efficiency_Miles_per_kWh"]
    summary_df = generate_statistical_report(df, cols)
    cov_matrix = compute_covariance_matrix(df, cols)
    corr_matrix = compute_correlation_matrix(df, cols)
    
    # Export artifacts
    STATISTICAL_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(STATISTICAL_SUMMARY_PATH, index=False)
    cov_matrix.to_csv(COVARIANCE_MATRIX_PATH)
    corr_matrix.to_csv(CORRELATION_MATRIX_PATH)
    
    print("\n--- Statistical Measures Summary Table ---")
    print(summary_df.to_string(index=False))
    
    print("\n--- Covariance Matrix ---")
    print(cov_matrix.to_string())
    
    print("\n--- Pearson Correlation Matrix ---")
    print(corr_matrix.to_string())
    
    print(f"\n[+] Exported statistical artifacts to: {STATISTICAL_SUMMARY_PATH.parent}")
    return summary_df, cov_matrix, corr_matrix


def run_visualize(df=None, corr_matrix=None):
    """Generates all 6 publication-grade figures."""
    logger.info("Generating Publication Visualizations...")
    if df is None:
        df = load_data(CLEANED_DATA_PATH if CLEANED_DATA_PATH.exists() else RAW_DATA_PATH)
    if corr_matrix is None:
        cols = NUMERICAL_COLS + ["Range_Efficiency_Miles_per_kWh"]
        corr_matrix = compute_correlation_matrix(df, cols)
        
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    plot_univariate_distributions(df, FIGURES_DIR / "fig1_univariate_distributions.png")
    plot_bivariate_relationship(df, FIGURES_DIR / "fig2_range_vs_battery_scatter.png")
    plot_correlation_heatmap(corr_matrix, FIGURES_DIR / "fig3_correlation_heatmap.png")
    plot_probability_distribution_qq(df, FIGURES_DIR / "fig4_probability_distribution_qq.png")
    plot_categorical_benchmarks(df, FIGURES_DIR / "fig5_manufacturer_benchmarks.png")
    plot_energy_vs_acceleration(df, FIGURES_DIR / "fig6_energy_vs_acceleration.png")
    
    print(f"\n[+] All 6 visual figures saved to: {FIGURES_DIR}")


def run_tests():
    """Discovers and runs unit tests."""
    logger.info("Running Unit Test Suite...")
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


def run_app():
    """Launches the Streamlit web dashboard."""
    logger.info("Starting Streamlit Dashboard...")
    app_path = Path("app/streamlit_app.py")
    if not app_path.exists():
        logger.error(f"Streamlit application file not found at: {app_path}")
        sys.exit(1)
        
    subprocess.run(["streamlit", "run", str(app_path)])


def main():
    parser = argparse.ArgumentParser(
        description="Electric Vehicle Data Analysis & Machine Learning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--all", action="store_true", help="Run entire end-to-end pipeline")
    parser.add_argument("--explore", action="store_true", help="Ingest and explore raw dataset")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess dataset and engineer features")
    parser.add_argument("--stats", action="store_true", help="Compute statistical metrics & export matrices")
    parser.add_argument("--visualize", action="store_true", help="Generate all 6 figures")
    parser.add_argument("--demo", action="store_true", help="Display demonstration protocol & GD themes")
    parser.add_argument("--viva", action="store_true", help="Display viva-voce Q&A (Bloom's K2 & K3)")
    parser.add_argument("--test", action="store_true", help="Execute unit test suite")
    parser.add_argument("--status", action="store_true", help="Check project artifact status")
    parser.add_argument("--app", action="store_true", help="Launch Streamlit web dashboard")
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1 or args.status:
        run_status()
        return

    if args.all:
        df_raw = run_explore()
        df_clean = run_preprocess(df_raw)
        _, _, corr = run_stats(df_clean)
        run_visualize(df_clean, corr)
        run_demonstration_walkthrough()
        display_group_discussion_guide()
        display_viva_questions_and_answers()
        run_status()
        return
        
    if args.explore:
        run_explore()
    if args.preprocess:
        run_preprocess()
    if args.stats:
        run_stats()
    if args.visualize:
        run_visualize()
    if args.demo:
        run_demonstration_walkthrough()
        display_group_discussion_guide()
    if args.viva:
        display_viva_questions_and_answers()
    if args.test:
        run_tests()
    if args.app:
        run_app()


if __name__ == "__main__":
    main()
