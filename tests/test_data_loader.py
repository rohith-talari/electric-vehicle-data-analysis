import unittest
import pandas as pd
from config import RAW_DATA_PATH
from src.data_loader import check_duplicates, check_missing_values, inspect_schema, load_data


class TestDataLoader(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.df = load_data(RAW_DATA_PATH)
        
    def test_load_data_success(self):
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertGreater(len(self.df), 0)
        
    def test_schema_keys(self):
        schema = inspect_schema(self.df)
        self.assertIn("n_rows", schema)
        self.assertIn("n_cols", schema)
        self.assertIn("columns", schema)
        self.assertGreaterEqual(schema["n_cols"], 10)
        
    def test_check_missing_values(self):
        missing = check_missing_values(self.df)
        self.assertIsInstance(missing, pd.DataFrame)
        self.assertIn("Missing_Count", missing.columns)
        
    def test_check_duplicates(self):
        dups = check_duplicates(self.df)
        self.assertIsInstance(dups, int)
        self.assertGreaterEqual(dups, 0)


if __name__ == "__main__":
    unittest.main()
