import unittest
import pandas as pd
import numpy as np
from src.preprocessing import (
    detect_outliers_iqr,
    impute_missing_values,
    preprocess_pipeline,
    remove_duplicates,
)


class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        # Create a sample DataFrame with known duplicates and nulls
        self.sample_df = pd.DataFrame({
            "A": [10.0, 20.0, np.nan, 20.0, 100.0, 20.0],
            "B": [5.0, 10.0, 15.0, 10.0, 25.0, 10.0],
            "C": ["X", "Y", "Z", "Y", "W", "Y"]
        })
        
    def test_remove_duplicates(self):
        df_dedup, dropped = remove_duplicates(self.sample_df)
        self.assertEqual(dropped, 2)
        self.assertEqual(len(df_dedup), 4)
        
    def test_impute_missing_values(self):
        df_imp, medians = impute_missing_values(self.sample_df, ["A"])
        self.assertEqual(df_imp["A"].isnull().sum(), 0)
        self.assertIn("A", medians)
        self.assertEqual(medians["A"], 20.0)
        
    def test_detect_outliers_iqr(self):
        audit = detect_outliers_iqr(self.sample_df, ["A", "B"])
        self.assertIn("A", audit)
        self.assertIn("Q1", audit["A"])
        self.assertIn("Q3", audit["A"])
        self.assertIn("IQR", audit["A"])
        self.assertIn("Outlier_Count", audit["A"])


if __name__ == "__main__":
    unittest.main()
