# ⚡ ELECTRIC VEHICLE DATA INTELLIGENCE & ANALYTICS
### Data Import, Exploration, Preprocessing, Statistical Analysis, and Visualization Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Bloom's Level](https://img.shields.io/badge/Bloom's%20Level-K2%20%7C%20K3-orange.svg)](#3-revised-blooms-taxonomy-mapping)
[![Tests](https://img.shields.io/badge/Tests-16%20Passed-brightgreen.svg)](tests/)

---
AUTHOR
TALARI ROHITH

## 1. Project Overview
**Electric Vehicle Data Intelligence & Analytics** is an end-to-end, reproducible data science and machine learning repository developed to satisfy the academic curriculum requirements of **Task 1: Group Discussion and Demonstration Method on Data Import, Exploration and Statistical Analysis using Python**.

This project provides:
1. **Self-Learning Synthesis:** Theoretical foundation on Python's core data science libraries (**NumPy, Pandas, Matplotlib, Seaborn, SciPy**).
2. **Automated Pipeline:** Ingestion of an authentic Electric Vehicle (EV) dataset, deduplication, missing value imputation, and Interquartile Range (IQR) outlier detection.
3. **Statistical Modeling:** Complete quantification of central tendency (Mean, Median, Mode), dispersion (Variance, Standard Deviation, Range, IQR), multivariate association (**Covariance & Pearson Correlation matrices**), and probability distribution analysis (**Skewness, Kurtosis, Shapiro-Wilk normality testing**).
4. **Visual Analytics:** Six publication-grade multi-panel visual figures uncovering vehicular range-battery scaling, bimodal powertrain segmentation, and performance trade-offs.
5. **Interactive UI & Deliverables:** Modular CLI tool (`main.py`), interactive dark-themed Streamlit dashboard (`app/streamlit_app.py`), executable Jupyter Notebook (`notebooks/`), automated test suite (`tests/`), and a 10-page formal PDF project report (`reports/EV_Data_Analysis_Project_Report.pdf`).

---

## 2. Academic Task Specifications
| Field | Academic Requirement Specification |
| :--- | :--- |
| **Task Name** | **Group Discussion and Demonstration Method on Data Import, Exploration and Statistical Analysis using Python** |
| **Task Level** | **Revised Bloom's Taxonomy Level: K2 (Understand), K3 (Apply)** |
| **Term Work / Self Learning** | **4 Hours (3 Hours Laboratory Implementation + 1 Hour Group Discussion & Demonstration)** |
| **Assessment / Due Date** | **31.08.2026 (Friday)** |
| **Domain Scope** | **Electric Vehicle (EV) Engineering & Machine Learning Dataset** |

---

## 3. Revised Bloom's Taxonomy Mapping
- **K2 (Understand):** 
  - Explain data structures in NumPy (`ndarray`) and Pandas (`DataFrame`, `Series`).
  - Interpret measures of central tendency vs dispersion under skewed conditions.
  - Understand the physical and mathematical significance of covariance and correlation signs.
  - Distinguish empirical probability density from theoretical Gaussian parametric models.
- **K3 (Apply):**
  - Implement Python programs to load CSV data and audit schema dynamically.
  - Apply median imputation to resist extreme value distortions.
  - Calculate IQR fences (`Q1 - 1.5 * IQR` and `Q3 + 1.5 * IQR`) to tag statistical outliers.
  - Formulate and compute full sample covariance and Pearson correlation matrices.
  - Construct multi-panel Matplotlib and Seaborn graphics for technical demonstration.

---

## 4. Repository Architecture
Aligned directly with production-grade ML repositories:

```
electric-vehicle-data-analysis-ml-main/
├── README.md                          # Comprehensive project documentation
├── requirements.txt                   # Dependency definitions
├── config.py                          # Centralized paths and hyperparameters
├── main.py                            # Unified CLI application entrypoint
├── LICENSE                            # MIT License
├── .gitignore                         # Python/IDE ignore rules
├── data/
│   ├── README.md                      # Data documentation & dictionary
│   ├── ev_dataset.csv                 # Raw dataset (455 rows, 13 features)
│   └── ev_cleaned_dataset.csv         # Cleaned & feature-engineered dataset
├── notebooks/
│   └── EV_Data_Import_Exploration_Statistical_Analysis.ipynb # End-to-end Jupyter Notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Ingestion & schema auditing
│   ├── preprocessing.py               # Deduplication, median imputation, IQR bounds
│   ├── feature_engineering.py         # Range efficiency, cost per mile, Z-score
│   ├── statistical_analysis.py        # Central tendency, dispersion, covariance, normality
│   ├── visualization.py               # 6 high-resolution Matplotlib/Seaborn figures
│   └── demonstration.py               # GD themes and demonstration protocol
├── reports/
│   ├── README.md
│   ├── EV_Data_Analysis_Project_Report.pdf  # 10-page publication-quality PDF report
│   └── figures/                       # Generated high-resolution plots
│       ├── fig1_univariate_distributions.png
│       ├── fig2_range_vs_battery_scatter.png
│       ├── fig3_correlation_heatmap.png
│       ├── fig4_probability_distribution_qq.png
│       ├── fig5_manufacturer_benchmarks.png
│       └── fig6_energy_vs_acceleration.png
├── artifacts/
│   ├── README.md
│   ├── statistical_summary.csv        # Consolidated statistical report
│   ├── covariance_matrix.csv          # Sample covariance matrix
│   └── correlation_matrix.csv         # Pearson correlation matrix
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py            # Ingestion unit tests
│   ├── test_preprocessing.py          # Preprocessing unit tests
│   ├── test_feature_engineering.py    # Feature computation tests
│   └── test_statistical_analysis.py   # Statistical measure verification tests
└── app/
    ├── __init__.py
    └── streamlit_app.py               # Interactive Streamlit analytics application
```

---

## 5. Dataset Specifications
The dataset captures 13 multidimensional variables across modern BEVs and PHEVs:

| Variable Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `VIN_Prefix` | String | - | Vehicle Identification Number prefix |
| `Make` & `Model` | Categorical | Text | Manufacturer and model identifier |
| `Model_Year` | Integer | Year | Model manufacturing year (2021 to 2026) |
| `EV_Type` | Categorical | Class | Powertrain classification (`BEV` or `PHEV`) |
| `CAFV_Eligibility` | Categorical | Status | Clean Alternative Fuel Vehicle eligibility status |
| `Battery_Capacity_kWh` | Numeric | kWh | Traction battery energy capacity (12 to 111.5 kWh) |
| `Electric_Range_Miles` | Numeric | Miles | Total rated pure electric driving range (15 to 405 mi) |
| `Base_MSRP_USD` | Numeric | USD ($) | Manufacturer Suggested Retail Price |
| `Charging_Time_Hours` | Numeric | Hours | Level 2 full charge duration |
| `Energy_Consumption_Wh_km` | Numeric | Wh/km | Electrical energy consumption per kilometer |
| `Acceleration_0_100_s` | Numeric | Sec | Elapsed time from 0 to 100 km/h |
| `Top_Speed_kmh` | Numeric | km/h | Governed top vehicular speed |

---

## 6. Summary of Statistical Findings
Quantified output across 450 validated vehicle records:

| Feature | Mean | Median | Mode | Variance | Std Dev | IQR | Skewness | Kurtosis | Normality |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Battery Capacity (kWh)** | 62.00 | 68.25 | 68.25 | 896.54 | 29.94 | 60.22 | -0.414 | -0.755 | Non-Normal (p<0.05) |
| **Electric Range (Miles)** | 206.25 | 244.00 | 15.00 | 12,114.15 | 110.06 | 222.75 | -0.659 | -0.834 | Non-Normal (p<0.05) |
| **Base MSRP (USD)** | $54,428.84 | $47,794.00 | $47,794.00 | 4.09e8 | $20,227.98 | $14,776.50 | +1.464 | +1.662 | Non-Normal (p<0.05) |
| **Charging Time (Hours)** | 5.64 | 6.10 | 1.20 | 7.65 | 2.77 | 5.28 | -0.284 | -0.705 | Non-Normal (p<0.05) |
| **Energy Consumption (Wh/km)**| 187.62 | 181.25 | 120.00 | 1,231.96 | 35.10 | 40.32 | +0.818 | +0.493 | Non-Normal (p<0.05) |
| **Acceleration (0-100 s)** | 5.55 | 5.50 | 5.50 | 1.60 | 1.26 | 1.90 | +0.095 | -0.735 | Non-Normal (p<0.05) |
| **Top Speed (km/h)** | 188.92 | 183.00 | 180.00 | 902.43 | 30.04 | 40.00 | +0.691 | -0.407 | Non-Normal (p<0.05) |
| **Range Efficiency (mi/kWh)** | 3.13 | 3.33 | 4.07 | 0.87 | 0.93 | 1.29 | -0.682 | +0.041 | Non-Normal (p<0.05) |

### Key Association Measures:
- **Covariance:** $\text{Cov}(\text{Battery}, \text{Range}) = +2962.45$
- **Correlation:** Pearson $r = +0.899$ (Strong direct linear dependency between battery size and driving range)
- **Economic Inversion:** Top speed and acceleration time exhibit negative correlation ($r = -0.698$), with faster models driving higher base prices ($r = -0.584$ with acceleration).

---

## 7. Quickstart & CLI Command Reference

### Installation
```bash
git clone <repo-url>
cd electric-vehicle-data-analysis-ml-main
pip install -r requirements.txt
```

### Unified CLI Operations
```bash
# 1. Run complete end-to-end analysis pipeline
python main.py --all

# 2. Ingest and explore dataset schema
python main.py --explore

# 3. Preprocess dataset, impute medians, and engineer features
python main.py --preprocess

# 4. Compute statistical measures and export CSV matrices
python main.py --stats

# 5. Generate all 6 high-resolution publication figures
python main.py --visualize

# 6. Display demonstration walkthrough & group discussion themes
python main.py --demo

# 7. Display viva-voce questions and model answers
python main.py --viva

# 8. Run unit test suite (16 tests)
python main.py --test

# 9. Launch interactive Streamlit web dashboard
python main.py --app
```

---

## 8. Group Discussion & Demonstration Guide

### Demonstration Method Protocol (Laboratory Evaluation)
1. **Step 1:** Ingest `ev_dataset.csv` and show initial dimensions ($455 \times 13$).
2. **Step 2:** Explain the rationale for median imputation on skewed financial and battery data.
3. **Step 3:** Highlight mean vs median divergence, skewness coefficients, and covariance values.
4. **Step 4:** Present figures in `reports/figures/` validating engineering relationships.

### Group Discussion Themes
- **Theme 1:** *Battery Capacity vs Mass & Efficiency Trade-offs.* (Larger packs boost range $r=0.899$, but increase mass and energy consumption Wh/km).
- **Theme 2:** *Bimodal Powertrain Segmentation.* (PHEVs address urban commuter range anxiety; BEVs target intercity highway transport).
- **Theme 3:** *Handling Financial Outliers in ML.* (High-end sports EVs are legitimate market signals and should not be discarded).

### Viva-Voce Questions (Bloom's K2 & K3)
1. **[K2] Why is the median preferred over the mean for Base MSRP?**  
   *Answer:* Base MSRP is positively skewed (+1.464) due to high-end luxury models. The median ($47,794) is robust against extreme values.
2. **[K2] What does negative skewness in Electric Range signify?**  
   *Answer:* It indicates a left-skewed density where most modern BEVs cluster around 250-320 miles, with low-range PHEVs forming the left tail.
3. **[K3] How does the IQR method detect outliers?**  
   *Answer:* Computes $\text{IQR} = Q_3 - Q_1$ and flags values outside $[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$.
4. **[K3] How do you interpret covariance between Battery Capacity and Range?**  
   *Answer:* Positive covariance (+2962.45) confirms direct co-movement: range expands proportionately with battery pack energy ($r=0.899$).

---

## 9. Quality Assurance & Automated Testing
The project includes a 16-test suite verifying pipeline integrity:
```bash
python -m unittest discover tests
```
- Schema validation & ingestion (`test_data_loader.py`)
- Deduplication & median imputation (`test_preprocessing.py`)
- Feature engineering formulas (`test_feature_engineering.py`)
- Central tendency, covariance, and correlation computations (`test_statistical_analysis.py`)

---

## 10. License
Distributed under the **MIT License**. See `LICENSE` for details.
