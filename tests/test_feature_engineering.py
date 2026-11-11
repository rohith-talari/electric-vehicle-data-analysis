import unittest
import pandas as pd
from src.feature_engineering import (
    apply_feature_engineering,
    compute_cost_per_range_mile,
    compute_range_efficiency,
    standardize_features,
)


class TestFeatureEngineering(unittest.TestCase):
    
    def setUp(self):
        self.sample_df = pd.DataFrame({
            "Battery_Capacity_kWh": [50.0, 100.0, 75.0],
            "Electric_Range_Miles": [200.0, 300.0, 300.0],
            "Base_MSRP_USD": [40000.0, 90000.0, 60000.0]
        })
        
    def test_compute_range_efficiency(self):
        eff = compute_range_efficiency(self.sample_df)
        self.assertAlmostEqual(eff.iloc[0], 4.00)
        self.assertAlmostEqual(eff.iloc[1], 3.00)
        
    def test_compute_cost_per_range_mile(self):
        cost = compute_cost_per_range_mile(self.sample_df)
        self.assertAlmostEqual(cost.iloc[0], 200.00)
        self.assertAlmostEqual(cost.iloc[1], 300.00)
        
    def test_apply_feature_engineering(self):
        df_feat = apply_feature_engineering(self.sample_df)
        self.assertIn("Range_Efficiency_Miles_per_kWh", df_feat.columns)
        self.assertIn("Cost_per_Range_Mile_USD", df_feat.columns)
        
    def test_standardize_features(self):
        df_std = standardize_features(self.sample_df, ["Battery_Capacity_kWh"])
        self.assertIn("Battery_Capacity_kWh_ZScore", df_std.columns)
        self.assertAlmostEqual(df_std["Battery_Capacity_kWh_ZScore"].mean(), 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
