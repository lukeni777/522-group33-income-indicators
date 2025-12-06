# validation.py
import pandas as pd
from typing import List, Dict, Tuple
import numpy as np
import os
import click
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, errors
from sklearn.model_selection import train_test_split
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

# Expected Category levels
category_levels = {
    "workclass":[
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"
    ],
    "education":[
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"
    ],
    "marital-status":[
        "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ],
    "occupation":[
        "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
    ],
    "relationship":[
        "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
    ],
    "race":[
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
    ],
    "sex":[
        "Female", "Male"
    ],
    "native-country":[
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"
    ],
    "income":[
        ">50K", "<=50K"
    ]
}


numeric_cols = ['capital-gain','capital-loss']
# Target column
target_col = "income"

class DataValidator:
    """
    A class to perform data validation checks on the Adult Census dataset, 
    using Pandera for structural validation and custom methods for data quality.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.missing_threshold: float = 0.05 # 5% missingness threshold
        self.tolerance: float = 0.05 # 5% Target distribution tolerance
        self.threshold: float = 0.8 # 80% correlation threshold

    def validate_all(self, expected_dist):
        """Run all validation checks."""
        print("--- Starting Data Validation Checks ---")
        self.check_column_structure_and_types()
        self.check_for_empty_observations()
        self.check_missingness_threshold()
        self.check_for_duplicate_observations()
        self.check_for_outliers()
        self.check_category_levels()
        self.check_target_distribution(expected_dist)
        self.check_target_feature_correlation()
        self.check_feature_correlations()
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

    ## 7. No outlier or anomalous values
    def check_for_outliers(self):

        # numeric columns to check outliers from
        #numeric_cols = self.df.select_dtypes(include=['number']).columns
        columns_with_outliers = []

        # Using IQR method (Interquartile range) for outliers
        for col in numeric_cols:
            if self.df[col].nunique() <= 2:
                continue   # skip zero-inflated / categorical numeric columns
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = self.df[(self.df[col] < q1 - 1.5*iqr) | (self.df[col] > q3 + 1.5*iqr)] 
            if not outliers.empty:
                columns_with_outliers.append(col)
                print(outliers[[col]])

        if columns_with_outliers:
            error_message = f"{columns_with_outliers} column values have outliers."
            raise DataValidationError(error_message)
        print("No outliers found in numeric columns.")

    ## 8. Correct category levels (i.e., no string mismatches or single values)
    def check_category_levels(self):
        
        category_cols = self.df.select_dtypes(include=['object']).columns
        column_with_anomalies = []

        for col,allowed_levels in category_levels.items():
            if col in self.df.columns:
                # Strip leading/trailing spaces
                self.df[col] = self.df[col].astype(str).str.strip()   
                # Count values not in allowed levels
                mask = ~self.df[col].isin(allowed_levels)
                count_anomalies = mask.sum()
                
                if count_anomalies > 0:
                    column_with_anomalies.append({"column": col, "anomalous_count": count_anomalies}) 

        if column_with_anomalies:
            error_message = f"category columns with anomalies {column_with_anomalies}."
            raise DataValidationError(error_message)
        print("No anomalies found in categorical columns.")

    ## 9. Target/response variable follows expected distribution
    def check_target_distribution(self, expected_dist):
        
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        # Actual distribution (proportion)
        actual_dist = self.df[target_col].value_counts(normalize=True)

        # Compare with expected distribution
        anomalies = []
        for category, expected_prop in expected_dist.items():
            actual_prop = actual_dist.get(category, 0)
            if abs(actual_prop - expected_prop) > self.tolerance:
                anomalies.append(
                    f"Category '{category}' expected {expected_prop:.2f}, got {actual_prop:.2f}"
                )

        if anomalies:
            error_message = "Target distribution does NOT match expected distribution:" + "\n".join(anomalies)
            raise DataValidationError(error_message)

        print("Target distribution matches expected proportions.")

    ## 10. No anomalous correlations between target/response variable and features/explanatory variables
    def check_target_feature_correlation(self):
        
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()

        # Create temporary encoded column
        encoded_col = target_col + '_encoded'
        
        # Encode categorical target as numeric
        self.df[encoded_col] = self.df[target_col].map({'<=50K': 0, '>50K': 1})
        anomalies = []

        for col in numeric_cols:
            x = self.df[encoded_col]
            y = self.df[col]
            
            if x.nunique() <= 1 or y.nunique() <= 1:
                corr = 0  # skip constant columns
            else:
                corr = x.corr(y, method="pearson")
            if abs(corr) >= self.threshold:
                anomalies.append({"feature": col, "correlation": corr})

        if anomalies:
            error_message = "Anomalous correlations detected with target variable:\n"
            for item in anomalies:
                error_message += f"  - {item['feature']}: correlation = {item['correlation']:.2f}\n"
            raise DataValidationError(error_message)

        print("No anomalous correlations found between target and numeric features.")

    ## 11. No anomalous correlations between features/explanatory variables 
    def check_feature_correlations(self):
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        # Avoid correlating the encoded target with the numeric features twice and creating false anomalies.
        encoded_col = target_col + '_encoded'
        if encoded_col in numeric_cols:
            numeric_cols.remove(encoded_col)

        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            print("Not enough numeric features to check correlations.")
            return

        # Compute correlation matrix
        corr_matrix = self.df[numeric_cols].corr(method="pearson").abs()

        anomalies = []

        # Iterate over upper triangle of correlation matrix to avoid duplicate pairs
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1 = numeric_cols[i]
                col2 = numeric_cols[j]
                corr_value = corr_matrix.loc[col1, col2]

                if np.isnan(corr_value):
                    # Skip NaN correlations (happens if a column has zero variance)
                    continue

                if corr_value >= self.threshold:
                    anomalies.append({"feature_1": col1, "feature_2": col2, "correlation": corr_value})

        if anomalies:
            error_message = "Anomalous correlations detected between features:\n"
            for item in anomalies:
                error_message += f"  - {item['feature_1']} & {item['feature_2']}: correlation = {item['correlation']:.2f}\n"
            raise DataValidationError(error_message)

        print("No anomalous correlations found between numeric features.")


@click.command()
@click.option('--in_file', default="data/raw/adult_census_data.csv", help="Input raw file")
def main(in_file):
    # Read & Transform Data for Validation
    print("Loading data for validation...")
    adult_df = pd.read_csv(in_file)

    print("Basic Data Transformation in progress...")
    adult_df.income = adult_df.income.replace(to_replace=['<=50K.', '>50K.'], value=['<=50K','>50K'])
    adult_df = adult_df.drop_duplicates()

    # Remove outliers in capital-gain and capital-loss
    numeric_cols = ['capital-gain','capital-loss']
    for col in numeric_cols:
        if adult_df[col].nunique() <= 2:
                    continue   # skip zero-inflated / categorical numeric columns
        q1 = adult_df[col].quantile(0.25)
        q3 = adult_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
            
        adult_df = adult_df[(adult_df[col] >= lower) & (adult_df[col] <= upper)]

    # Remove anomalies in categorical columns (presence of '?')
    adult_df = adult_df.replace('?', np.nan)

    # Drop null values from the data
    adult_df = adult_df.dropna()
    expected_income_dist = {"<=50K": 0.80, ">50K": 0.20}

    # --- Validation ---
    print("Validation started...")
    try:
        # 1. Check file existence/format
        DataValidator.check_file_format_and_existence(in_file)
        
        # 2. Run other stucture & data quality checks
        validator = DataValidator(adult_df)
        validator.validate_all(expected_income_dist)
        print("\n\nSUCCESS: Data passed all validation checks and is ready for analysis!")

    except DataValidationError as e:
        print(f"\n=========================================================================")
        print(f"VALIDATION FAILED! Analysis Halted to Prevent Data Leakage/Errors.")
        print(f"Error Details: {e}")
        print(f"=========================================================================")
        adult_df = None
        
    except Exception as e:
        # Catch any other unexpected loading errors
        print(f"An unexpected error occurred during data loading: {e}")
        adult_df = None

    if adult_df is not None:
        print(f"\nProceeding with a validated DataFrame of shape: {adult_df.shape}")

    # Combine all married groups in marital status to one group. 
    adult_df['marital-status'] = adult_df['marital-status'].replace(to_replace=r'^Married\b.*', value='Married', regex=True)

    # Create a Test train split of the data
    adult_train, adult_test = train_test_split(adult_df, test_size=0.3, random_state=522)

    # Store Cleaned Data in processed data directory
    adult_train.to_csv(os.path.join("data", "processed", "adult_census_training_data.csv"), index=False)
    adult_test.to_csv(os.path.join("data", "processed", "adult_census_test_data.csv"), index=False)
    print("Cleaned training and test data saved to 'data/processed/' directory.")

import os
if __name__ == "__main__":
    main()