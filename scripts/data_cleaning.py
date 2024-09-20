import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy import stats

def clean_data(df):
    """
    Clean the data by handling missing values and outliers.

    Parameters:
    df (pd.DataFrame): The input dataframe to be cleaned.

    Returns:
    pd.DataFrame: The cleaned dataframe.
    """
    
    # Handle missing values
    # Define numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=[object]).columns.tolist()

    # Impute missing values for numerical columns with median
    num_imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    # Impute missing values for categorical columns with the most frequent value
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # Handle outliers
    # Define a function to remove outliers using Z-score method
    def remove_outliers_z(df, z_thresh=3):
        """
        Remove outliers from the dataframe using Z-score method.

        Parameters:
        df (pd.DataFrame): The dataframe to clean.
        z_thresh (float): The Z-score threshold to define outliers.

        Returns:
        pd.DataFrame: The dataframe with outliers removed.
        """
        z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
        df_no_outliers = df[(z_scores < z_thresh).all(axis=1)]
        return df_no_outliers

    # Remove outliers from numerical columns
    df = remove_outliers_z(df)
    
    return df

# Example usage:
# df = pd.read_csv('your_data.csv')
# cleaned_df = clean_data(df)
