import unittest
import pandas as pd
import numpy as np
from scipy import stats  # Make sure to import stats from scipy
from scripts.data_cleaning import clean_data  # Assuming the function is in a module named clean_data_module

class TestCleanData(unittest.TestCase):
    
    def setUp(self):
        # Create a sample dataframe with missing values and outliers
        data = {
            'numerical1': [1, 2, 3, 1000, np.nan],  # Outlier at 1000, missing value
            'numerical2': [10, 20, 30, 40, 50],  # No missing values or outliers
            'categorical': ['A', 'B', 'A', np.nan, 'B']  # Missing value
        }
        self.df = pd.DataFrame(data)
    
    def test_clean_data(self):
        cleaned_df = clean_data(self.df)

        # Test if missing numerical values are filled with median
        self.assertFalse(cleaned_df['numerical1'].isnull().any(), "Numerical column has missing values after cleaning.")
        median_num1 = np.nanmedian(self.df['numerical1'])
        self.assertEqual(cleaned_df['numerical1'].iloc[4], median_num1, "Numerical missing values not filled correctly.")
        
        # Test if missing categorical values are filled with the most frequent value
        self.assertFalse(cleaned_df['categorical'].isnull().any(), "Categorical column has missing values after cleaning.")
        most_frequent_cat = self.df['categorical'].mode()[0]
        self.assertEqual(cleaned_df['categorical'].iloc[3], most_frequent_cat, "Categorical missing values not filled correctly.")
        
        # Test if outliers are removed
        self.assertNotIn(1000, cleaned_df['numerical1'], "Outlier not removed from numerical1 column.")
    
    def test_outliers_removal(self):
        # Ensure outliers are removed correctly
        cleaned_df = clean_data(self.df)
        z_scores = np.abs(stats.zscore(self.df.select_dtypes(include=[np.number])))
        outliers_exist = (z_scores >= 3).any(axis=None)
        self.assertFalse(outliers_exist, "Outliers were not removed.")

if __name__ == '__main__':
    unittest.main()
