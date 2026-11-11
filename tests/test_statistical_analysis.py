import unittest
import pandas as pd
import numpy as np
from src.statistical_analysis import (
    compute_central_tendency,
    compute_correlation_matrix,
    compute_covariance_matrix,
    compute_dispersion_measures,
    compute_probability_distributions,
    generate_statistical_report,
)


class TestStatisticalAnalysis(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame({
            "X": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "Y": [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
            "Z": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        })
        
    def test_central_tendency(self):
        ct = compute_central_tendency(self.df, ["X", "Y"])
        self.assertEqual(len(ct), 2)
        self.assertAlmostEqual(ct.loc[ct["Feature"] == "X", "Mean"].iloc[0], 55.0)
        self.assertAlmostEqual(ct.loc[ct["Feature"] == "X", "Median"].iloc[0], 55.0)
        
    def test_dispersion_measures(self):
        disp = compute_dispersion_measures(self.df, ["X"])
        self.assertAlmostEqual(disp["Range"].iloc[0], 90.0)
        self.assertGreater(disp["Variance"].iloc[0], 0)
        
    def test_covariance_and_correlation(self):
        cov = compute_covariance_matrix(self.df, ["X", "Y", "Z"])
        corr = compute_correlation_matrix(self.df, ["X", "Y", "Z"])
        # X and Y have perfect positive correlation
        self.assertAlmostEqual(corr.loc["X", "Y"], 1.0, places=2)
        # X and Z have perfect negative correlation
        self.assertAlmostEqual(corr.loc["X", "Z"], -1.0, places=2)
        
    def test_probability_distributions(self):
        dist = compute_probability_distributions(self.df, ["X"])
        self.assertIn("Skewness", dist.columns)
        self.assertIn("Kurtosis", dist.columns)
        self.assertIn("Normality", dist.columns)
        
    def test_generate_statistical_report(self):
        rep = generate_statistical_report(self.df, ["X", "Y"])
        self.assertEqual(len(rep), 2)
        self.assertIn("Mean", rep.columns)
        self.assertIn("Std_Dev", rep.columns)
        self.assertIn("Skewness", rep.columns)


if __name__ == "__main__":
    unittest.main()
