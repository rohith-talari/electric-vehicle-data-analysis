"""Data Visualization Module for EV Dataset."""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Configure styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150


def plot_univariate_distributions(df: pd.DataFrame, save_path: Optional[str | Path] = None) -> plt.Figure:
    """Figure 1: Univariate histograms with KDE curves and boxplots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Figure 1: Univariate Analysis - Histograms with KDE and Boxplots", fontsize=14, fontweight='bold', y=0.98)
    
    # Electric Range Histogram
    sns.histplot(df["Electric_Range_Miles"], kde=True, ax=axes[0, 0], color="#1e3a8a", bins=25)
    axes[0, 0].set_title("Electric Range Distribution (Miles)")
    axes[0, 0].set_xlabel("Electric Range (Miles)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].axvline(df["Electric_Range_Miles"].mean(), color='red', linestyle='--', label=f"Mean: {df['Electric_Range_Miles'].mean():.1f}")
    axes[0, 0].axvline(df["Electric_Range_Miles"].median(), color='green', linestyle=':', label=f"Median: {df['Electric_Range_Miles'].median():.1f}")
    axes[0, 0].legend()
    
    # Battery Capacity Histogram
    sns.histplot(df["Battery_Capacity_kWh"], kde=True, ax=axes[0, 1], color="#0d9488", bins=25)
    axes[0, 1].set_title("Battery Capacity Distribution (kWh)")
    axes[0, 1].set_xlabel("Battery Capacity (kWh)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].axvline(df["Battery_Capacity_kWh"].mean(), color='red', linestyle='--', label=f"Mean: {df['Battery_Capacity_kWh'].mean():.1f}")
    axes[0, 1].axvline(df["Battery_Capacity_kWh"].median(), color='green', linestyle=':', label=f"Median: {df['Battery_Capacity_kWh'].median():.1f}")
    axes[0, 1].legend()
    
    # Boxplot of Electric Range by EV Type
    sns.boxplot(data=df, x="EV_Type", y="Electric_Range_Miles", hue="EV_Type", ax=axes[1, 0], palette={"BEV": "#3b82f6", "PHEV": "#10b981"}, legend=False)
    axes[1, 0].set_title("Electric Range by Vehicle Powertrain Type")
    axes[1, 0].set_xlabel("EV Powertrain Type")
    axes[1, 0].set_ylabel("Electric Range (Miles)")
    
    # Boxplot of Base MSRP by EV Type
    sns.boxplot(data=df, x="EV_Type", y="Base_MSRP_USD", hue="EV_Type", ax=axes[1, 1], palette={"BEV": "#3b82f6", "PHEV": "#10b981"}, legend=False)
    axes[1, 1].set_title("Base MSRP ($) by Vehicle Powertrain Type")
    axes[1, 1].set_xlabel("EV Powertrain Type")
    axes[1, 1].set_ylabel("Base MSRP (USD)")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_bivariate_relationship(df: pd.DataFrame, save_path: Optional[str | Path] = None) -> plt.Figure:
    """Figure 2: Scatter plot of Range vs Battery Capacity with regression line."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="Battery_Capacity_kWh",
        y="Electric_Range_Miles",
        hue="EV_Type",
        style="EV_Type",
        size="Base_MSRP_USD",
        sizes=(40, 200),
        palette={"BEV": "#2563eb", "PHEV": "#059669"},
        alpha=0.8,
        ax=ax
    )
    sns.regplot(
        data=df,
        x="Battery_Capacity_kWh",
        y="Electric_Range_Miles",
        scatter=False,
        ax=ax,
        color="#dc2626",
        line_kws={"linestyle": "--", "linewidth": 2, "label": "Global Linear Fit"}
    )
    ax.set_title("Figure 2: Electric Range vs. Battery Capacity (Segmented by EV Type & MSRP)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Battery Capacity (kWh)")
    ax.set_ylabel("Electric Range (Miles)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, save_path: Optional[str | Path] = None) -> plt.Figure:
    """Figure 3: Pearson correlation matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"},
        ax=ax
    )
    ax.set_title("Figure 3: Pearson Correlation Matrix Heatmap of EV Metrics", fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_probability_distribution_qq(df: pd.DataFrame, save_path: Optional[str | Path] = None) -> plt.Figure:
    """Figure 4: Empirical distribution vs Gaussian fit and Q-Q normality plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 4: Probability Distribution Analysis (Electric Range)", fontsize=14, fontweight='bold')
    
    er_data = df["Electric_Range_Miles"].dropna()
    mu, std = stats.norm.fit(er_data)
    x_range = np.linspace(er_data.min() - 20, er_data.max() + 20, 300)
    pdf_normal = stats.norm.pdf(x_range, mu, std)
    
    sns.histplot(er_data, stat="density", kde=True, ax=axes[0], color="#6366f1", label="Empirical Density (KDE)", alpha=0.4)
    axes[0].plot(x_range, pdf_normal, 'r-', lw=2.5, label=f"Fitted Gaussian PDF\n($\mu$={mu:.1f}, $\sigma$={std:.1f})")
    axes[0].set_title("Empirical Density vs. Parametric Gaussian Fit")
    axes[0].set_xlabel("Electric Range (Miles)")
    axes[0].set_ylabel("Probability Density")
    axes[0].legend()
    
    stats.probplot(er_data, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_markerfacecolor('#4f46e5')
    axes[1].get_lines()[0].set_markeredgecolor('#312e81')
    axes[1].get_lines()[0].set_markersize(5.0)
    axes[1].get_lines()[1].set_color('#dc2626')
    axes[1].get_lines()[1].set_linewidth(2.0)
    axes[1].set_title("Normal Q-Q Plot (Quantile-Quantile)")
    axes[1].set_xlabel("Theoretical Quantiles")
    axes[1].set_ylabel("Ordered Sample Values")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_categorical_benchmarks(df: pd.DataFrame, save_path: Optional[str | Path] = None) -> plt.Figure:
    """Figure 5: Manufacturer range ranking and powertrain share."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Figure 5: Categorical Analysis - Manufacturer Benchmarks & Powertrain Share", fontsize=14, fontweight='bold')
    
    top_makes = df.groupby("Make")["Electric_Range_Miles"].mean().sort_values(ascending=False).head(8)
    sns.barplot(x=top_makes.values, y=top_makes.index, ax=axes[0], hue=top_makes.index, palette="viridis", legend=False)
    axes[0].set_title("Top 8 EV Manufacturers by Average Electric Range")
    axes[0].set_xlabel("Mean Electric Range (Miles)")
    axes[0].set_ylabel("Manufacturer")
    for i, v in enumerate(top_makes.values):
        axes[0].text(v + 3, i, f"{v:.1f} mi", va='center', fontweight='bold', fontsize=9)
        
    ev_counts = df["EV_Type"].value_counts()
    axes[1].pie(
        ev_counts,
        labels=[f"{k}\n({v} units)" for k, v in ev_counts.items()],
        autopct='%1.1f%%',
        startangle=140,
        colors=["#3b82f6", "#10b981"],
        explode=(0.05, 0),
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    axes[1].set_title("Fleet Composition: BEV vs. PHEV Share")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_energy_vs_acceleration(df: pd.DataFrame, save_path: Optional[str | Path] = None) -> plt.Figure:
    """Figure 6: Energy consumption vs acceleration colored by top speed."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    scatter = ax.scatter(
        df["Acceleration_0_100_s"],
        df["Energy_Consumption_Wh_km"],
        c=df["Top_Speed_kmh"],
        cmap="plasma",
        s=df["Battery_Capacity_kWh"] * 1.5,
        alpha=0.85,
        edgecolors='none'
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Top Speed (km/h)")
    ax.set_title("Figure 6: Energy Consumption (Wh/km) vs. Acceleration (0-100 km/h)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Acceleration 0-100 km/h (seconds - lower is faster)")
    ax.set_ylabel("Energy Consumption (Wh/km)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig
