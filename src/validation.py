# validation.py
import pandas as pd
from typing import List, Dict, Tuple
import numpy as np
import os
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, errors
from scipy import stats


class DataValidationError(Exception):
    """Custom exception to raise on validation failures."""
    pass

# Expected Schema & Types using Pandera
COLUMN_AND_TYPE_SCHEMA = DataFrameSchema(
    columns={
        # Correct Column Names & Data Types
        'age': Column(np.int64, nullable=False),
        'workclass': Column(object, nullable=True),
        'fnlwgt': Column(np.int64, nullable=False),
        'education': Column(object, nullable=False),
        'education-num': Column(np.int64, nullable=False),
        'marital-status': Column(object, nullable=False),
        'occupation': Column(object, nullable=True),
        'relationship': Column(object, nullable=False),
        'race': Column(object, nullable=False),
        'sex': Column(object, nullable=False),
        'capital-gain': Column(np.int64, nullable=False),
        'capital-loss': Column(np.int64, nullable=False),
        'hours-per-week': Column(np.int64, nullable=False),
        'native-country': Column(object, nullable=True),
        'income': Column(object, nullable=False)
    },
    strict=True # col name check
)

class DataValidator:
    """
    A class to perform data validation checks on the Adult Census dataset, 
    using Pandera for structural validation and custom methods for data quality.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.missing_threshold: float = 0.05 # 5% missingness threshold

    def validate_all(self):
        """Run all validation checks."""
        print("--- Starting Data Validation Checks ---")
        self.check_column_structure_and_types()
        self.check_for_empty_observations()
        self.check_missingness_threshold()
        self.check_for_duplicate_observations()
        print("--- All core data validation checks passed successfully! ---")

    ## 1 & 2. Correct column names and data types - Pandera
    def check_column_structure_and_types(self):
        
        try:
            # Validation based on the global schema
            COLUMN_AND_TYPE_SCHEMA.validate(self.df, lazy=True)
            print("Column names and critical data types are correct.")

        except errors.SchemaErrors as e:
            error_message = "Structural and Data Type validation failed (Pandera):\n"
            error_message += e.failure_cases.to_string()
            raise DataValidationError(error_message)
        
        except Exception as e:
            raise DataValidationError(f"An unexpected error occurred during Pandera validation: {e}")


    ## 3. No empty observations (row check)
    def check_for_empty_observations(self):
        
        empty_rows_count = self.df.isnull().all(axis=1).sum()
        
        if empty_rows_count > 0:
            error_message = f"{empty_rows_count} rows found with entirely empty observations."
            raise DataValidationError(error_message)
        print("No entirely empty observations found (i.e., no completely missing rows).")


    ## 4. Missingness not beyond expected threshold (column check - 5%)
    def check_missingness_threshold(self):
        
        missing_percent = self.df.isnull().sum() / len(self.df)
        
        exceeding_cols = missing_percent[missing_percent > self.missing_threshold]
        
        if not exceeding_cols.empty:
            error_details = "\n".join([
                f"Column '{col}': {perc:.2%}" for col, perc in exceeding_cols.items()])
            error_message = (
                f"Missingness exceeds the {self.missing_threshold:.0%} threshold in the following columns:\n{error_details}")
            raise DataValidationError(error_message)
        print(f"Missingness in all columns is within the {self.missing_threshold:.0%} threshold.")

    ## 5. Correct data file format and existence
    @staticmethod
    def check_file_format_and_existence(file_path: str):
    
        if not os.path.exists(file_path):
             raise DataValidationError(f"File not found: {file_path}. Expected a CSV file.")
        
        # check readability and format assumption
        try:
            pd.read_csv(file_path, nrows=5)
            print("Data file format (CSV) is confirmed and the file exists.")
        except Exception as e:
            raise DataValidationError(f"File format check failed. Error reading CSV: {e}")
        
    ## 6. No duplicate observations
    def check_for_duplicate_observations(self):
        
        duplicate_rows_count = self.df.duplicated().sum()

        if duplicate_rows_count > 0:
            error_message = f"{duplicate_rows_count} rows found with duplicate observations."
            raise DataValidationError(error_message)
        print("No duplicate observations found.")

