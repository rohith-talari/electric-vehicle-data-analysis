"""ELECTRIC VEHICLE DATA INTELLIGENCE & ANALYTICS DASHBOARD
Interactive Streamlit Application for EV Data Exploration, Statistical Analysis & Model Insights.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CLEANED_DATA_PATH,
    CORRELATION_MATRIX_PATH,
    COVARIANCE_MATRIX_PATH,
    FIGURES_DIR,
    RAW_DATA_PATH,
    STATISTICAL_SUMMARY_PATH,
)

# Page Configuration
st.set_page_config(
    page_title="EV Data Intelligence & Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_cached_data():
    raw_df = pd.read_csv(RAW_DATA_PATH) if RAW_DATA_PATH.exists() else None
    clean_df = pd.read_csv(CLEANED_DATA_PATH) if CLEANED_DATA_PATH.exists() else None
    stat_summary = pd.read_csv(STATISTICAL_SUMMARY_PATH) if STATISTICAL_SUMMARY_PATH.exists() else None
    cov_df = pd.read_csv(COVARIANCE_MATRIX_PATH, index_col=0) if COVARIANCE_MATRIX_PATH.exists() else None
    corr_df = pd.read_csv(CORRELATION_MATRIX_PATH, index_col=0) if CORRELATION_MATRIX_PATH.exists() else None
    return raw_df, clean_df, stat_summary, cov_df, corr_df


raw_df, clean_df, stat_summary, cov_df, corr_df = load_cached_data()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/electric-car.png", width=72)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "1. Overview & Exploration",
        "2. Preprocessing & Cleaning",
        "3. Statistical Measures & Matrices",
        "4. Probability Distributions",
        "5. Visual Analytics Studio",
        "6. Group Discussion & Viva Guide"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("**Course:** Python for Data Science / ML")
st.sidebar.caption("**Bloom's Level:** K2 (Understand), K3 (Apply)")
st.sidebar.caption("**Assessment Due:** 31.08.2026 (Friday)")

# Display Main Content
if page == "1. Overview & Exploration":
    st.markdown('<div class="main-header">⚡ Electric Vehicle Data Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Task 1: Data Import, Exploration, and Statistical Analysis using Python</div>', unsafe_allow_html=True)
    
    if raw_df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(raw_df))
        with col2:
            st.metric("Variables", raw_df.shape[1])
        with col3:
            st.metric("Manufacturers", raw_df["Make"].nunique())
        with col4:
            st.metric("Missing Values", raw_df.isnull().sum().sum())
            
        st.subheader("Raw Dataset Preview")
        st.dataframe(raw_df.head(20), use_container_width=True)
        
        st.subheader("Data Schema & Missing Values")
        null_counts = raw_df.isnull().sum()
        schema_df = pd.DataFrame({
            "Data Type": raw_df.dtypes.astype(str),
            "Missing Count": null_counts,
            "Missing Pct (%)": (null_counts / len(raw_df) * 100).round(2)
        })
        st.dataframe(schema_df, use_container_width=True)
    else:
        st.error("Raw dataset not found. Please run `python main.py --all` to generate data.")

elif page == "2. Preprocessing & Cleaning":
    st.markdown('<div class="main-header">🧹 Data Preprocessing & Feature Engineering</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Systematic deduplication, median imputation, IQR outlier auditing, and derived metrics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### 1. Deduplication\n- 5 duplicate records identified.\n- Removed to prevent artificial sample weighting.")
    with col2:
        st.success("### 2. Median Imputation\n- Applied to `Battery_Capacity_kWh`, `Base_MSRP_USD`, and `Charging_Time_Hours`.\n- Preserves central tendency against skewness.")
        
    st.subheader("Cleaned & Engineered Dataset")
    if clean_df is not None:
        st.dataframe(clean_df.head(20), use_container_width=True)
        
        st.subheader("Engineered Features")
        colA, colB = st.columns(2)
        with colA:
            st.write("**Range Efficiency (Miles / kWh):**")
            st.line_chart(clean_df["Range_Efficiency_Miles_per_kWh"].head(50))
        with colB:
            st.write("**Cost per Mile of Range ($ / Mile):**")
            st.line_chart(clean_df["Cost_per_Range_Mile_USD"].head(50))
    else:
        st.warning("Cleaned dataset not available.")

elif page == "3. Statistical Measures & Matrices":
    st.markdown('<div class="main-header">📊 Statistical Measures & Association Matrices</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Summary measures of central tendency, dispersion, covariance, and Pearson correlation</div>', unsafe_allow_html=True)
    
    if stat_summary is not None:
        st.subheader("Summary Table of Statistical Measures")
        st.dataframe(stat_summary, use_container_width=True)
        
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Covariance Matrix")
        if cov_df is not None:
            st.dataframe(cov_df, use_container_width=True)
    with col2:
        st.subheader("Pearson Correlation Matrix (r)")
        if corr_df is not None:
            st.dataframe(corr_df.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1), use_container_width=True)

elif page == "4. Probability Distributions":
    st.markdown('<div class="main-header">📈 Probability Distributions & Normality Testing</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Skewness, Kurtosis, and Shapiro-Wilk Hypothesis Testing</div>', unsafe_allow_html=True)
    
    if stat_summary is not None:
        st.dataframe(stat_summary[["Feature", "Skewness", "Kurtosis", "Normality"]], use_container_width=True)
        
    fig4_path = FIGURES_DIR / "fig4_probability_distribution_qq.png"
    if fig4_path.exists():
        st.image(str(fig4_path), caption="Figure 4: Empirical PDF vs Fitted Gaussian & Q-Q Plot", use_container_width=True)

elif page == "5. Visual Analytics Studio":
    st.markdown('<div class="main-header">🎨 Visual Analytics Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">High-resolution publication figures generated via Matplotlib and Seaborn</div>', unsafe_allow_html=True)
    
    fig_files = sorted(FIGURES_DIR.glob("fig*.png"))
    if fig_files:
        selected_fig = st.selectbox("Select Figure to Inspect", [f.name for f in fig_files])
        st.image(str(FIGURES_DIR / selected_fig), use_container_width=True)
    else:
        st.warning("Figures not found. Run `python main.py --visualize`.")

elif page == "6. Group Discussion & Viva Guide":
    st.markdown('<div class="main-header">🗣️ Group Discussion & Demonstration Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Preparation guide aligned with Revised Bloom\'s Taxonomy K2 (Understand) and K3 (Apply)</div>', unsafe_allow_html=True)
    
    with st.expander("📌 Demonstration Protocol (Laboratory Walkthrough)", expanded=True):
        st.markdown("""
        1. **Step 1:** Ingest dataset (`455 rows x 13 columns`) and audit initial schema with `df.info()`.
        2. **Step 2:** Justify median imputation for skewed MSRP and explain IQR outlier fences.
        3. **Step 3:** Highlight mean vs median divergence, skewness sign, and strong covariance (`+2962.45`).
        4. **Step 4:** Open figures in `reports/figures/` validating engineering relationships.
        """)
        
    with st.expander("💬 Group Discussion Themes & Talking Points", expanded=True):
        st.markdown("""
        - **Theme 1: Battery Capacity vs Mass/Efficiency Trade-offs:** Larger batteries expand range (`r = 0.899`), but add mass, driving up energy consumption (`Wh/km`).
        - **Theme 2: Bimodal Fleet Dynamics:** PHEVs (20-45 miles) serve urban commuters, whereas modern BEVs target 250+ mile highway mobility.
        - **Theme 3: Economic Outlier Treatment:** Luxury vehicle MSRPs ($85k+) represent genuine high-end market segments and must not be blindly clipped.
        """)
        
    with st.expander("🎓 Viva-Voce Questions & Model Answers", expanded=True):
        st.markdown("""
        - **[K2] Why is median chosen over mean for Base MSRP imputation?**  
          *Answer:* Base MSRP is positively skewed (`+1.464`) by luxury sports models. The median is resistant to extreme values.
        - **[K2] What does a negative skewness in Electric Range signify?**  
          *Answer:* It indicates a left-skewed distribution where the majority of modern BEVs cluster at higher ranges (250-320 miles).
        - **[K3] How is the IQR outlier detection rule formulated mathematically?**  
          *Answer:* `IQR = Q3 - Q1`. Outlier bounds are `[Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]`.
        - **[K3] How is covariance between Battery Capacity and Range interpreted?**  
          *Answer:* Positive covariance (`+2962.45`) indicates direct proportional scaling (`r = 0.899`).
        """)
