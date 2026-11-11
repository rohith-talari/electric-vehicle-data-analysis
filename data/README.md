# 📁 Data Directory

This directory contains raw and preprocessed datasets for the Electric Vehicle data analysis project:

- **`ev_dataset.csv`**: Raw ingested dataset containing 455 records and 13 variables across Battery Electric Vehicles (BEV) and Plug-in Hybrid Electric Vehicles (PHEV). Includes synthetic noise, duplicate records, and missing values to demonstrate data cleaning pipelines.
- **`ev_cleaned_dataset.csv`**: Preprocessed dataset with duplicates removed (450 rows), missing values imputed via median, and engineered features appended (`Range_Efficiency_Miles_per_kWh`, `Cost_per_Range_Mile_USD`).
