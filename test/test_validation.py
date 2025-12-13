import sys
import os
import pytest
import pandas as pd
import numpy as np
#from src.validation import DataValidator, DataValidationError

sys.path.append(os.path.join(os.path.dirname(__file__),".."))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.validation import DataValidator, DataValidationError

sample_data = pd.DataFrame({
    "age": [39, 58, 53],
    "workclass": ["State-gov","Self-emp-not-inc","Private"],
    "fnlwgt": [77516, 215646, 234721],
    "education": ["Bachelors","HS-grad","11th"],
    "education-num": [13, 9, 7],
    "marital-status": ["Never-married", "Divorced", "Married-civ-spouse"],
    "occupation": ["Adm-clerical", "Handlers-cleaners", "Handlers-cleaners"],
    "relationship": ["Not-in-family", "Not-in-family", "Husband"],
    "race": ["White","White","Black"],
    "sex": ['Male', 'Male', 'Female'],
    "capital-gain": [2174, 0, 14084],
    "capital-loss": [0, 2042, 0],
    "hours-per-week": [40, 13, 40],
    "native-country": ["United-States", "Cuba", "Jamaica"],
    "income": ["<=50K", "<=50K", ">50K"]
})

sample_data_weak_corr = pd.DataFrame({
    "age": [25, 45, 33, 52, 40],
    "workclass": ["State-gov","Self-emp-not-inc","Private","Private","Federal-gov"],
    "fnlwgt": [50000, 215000, 120000, 180000, 2000],
    "education": ["Bachelors","HS-grad","11th","Masters","Assoc-voc"],
    "education-num": [13, 9, 7, 14, 12],
    "marital-status": ["Never-married", "Divorced", "Married-civ-spouse", "Widowed", "Married-civ-spouse"],
    "occupation": ["Adm-clerical", "Handlers-cleaners", "Handlers-cleaners", "Exec-managerial", "Sales"],
    "relationship": ["Not-in-family", "Not-in-family", "Husband", "Unmarried", "Husband"],
    "race": ["White","White","Black","Asian-Pac-Islander","Other"],
    "sex": ['Male', 'Male', 'Female','Female','Male'],
    "capital-gain": [2174, 0, 14084, 500, 2000],
    "capital-loss": [0, 20, 0, 10, 15],
    "hours-per-week": [40, 13, 40, 37, 45],
    "native-country": ["United-States", "Cuba", "Jamaica", "Canada", "Germany"],
    "income": ["<=50K", "<=50K", ">50K", "<=50K", "<=50K"]
})

# 1 & 2. check_column_structure_and_types
def test_check_column_structure_and_types_valid():
    """Test validation passes on correct data."""
    validator = DataValidator(sample_data)
    # will not raise an error
    validator.check_column_structure_and_types()

def test_check_column_structure_and_types_invalid():
    """Test validation fails on invalid column and/or datatype."""
    df_invalid = sample_data.drop(columns=["age"])
    validator = DataValidator(df_invalid)    
    # will raise an error for missing column
    with pytest.raises(DataValidationError):
        validator.check_column_structure_and_types()
    
    df_invalid = sample_data.astype({'hours-per-week':'float'})
    validator = DataValidator(df_invalid)    
    # will raise an error for colum datatype mismatch
    with pytest.raises(DataValidationError):
        validator.check_column_structure_and_types()

# 3. check_for_empty_observations
def test_check_for_empty_observations_valid():
    """Test validation passes on correct data."""
    validator = DataValidator(sample_data)
    # will not raise an error
    validator.check_for_empty_observations()

def test_check_for_empty_observations_invalid():
    """Test validation fails when full empty row exist."""
    df_with_empty_row = sample_data.copy()
    # Append a fully empty row
    df_with_empty_row.loc[len(df_with_empty_row)] = np.nan

    validator = DataValidator(df_with_empty_row)

    with pytest.raises(DataValidationError) as excinfo:
        validator.check_for_empty_observations()

    assert "entirely empty observations" in str(excinfo.value)

# 4. check_missingness_threshold
def test_check_missingness_threshold_valid():
    """Test validation passes on correct data."""
    validator = DataValidator(sample_data)
    # will not raise an error
    validator.check_missingness_threshold()

def test_check_missingness_threshold_invalid():
    """Test validation fails when missingness exceeds 5%."""
    df_missing = sample_data.copy()
    # Introduce missing value (1 / 3 rows = 33.33%)
    df_missing.loc[0, "age"] = np.nan

    validator = DataValidator(df_missing)
    with pytest.raises(DataValidationError) as excinfo:
        validator.check_missingness_threshold()

    assert "Missingness exceeds " in str(excinfo.value)

# 5. check_file_format_and_existence
def test_check_file_format_and_existence_file_not_found(tmp_path):
    """Test fails when file does not exist."""
    tmp_path = "data/raw/adult_census_data.txt"
    non_existent_file = tmp_path
    
    with pytest.raises(DataValidationError) as excinfo:
        DataValidator.check_file_format_and_existence(str(non_existent_file))

    assert "File not found" in str(excinfo.value)

# 6. check_for_duplicate_observations
def test_check_for_duplicate_observations_valid():
    """Test passes when no duplicate rows exist."""
    validator = DataValidator(sample_data)
    validator.check_for_duplicate_observations()

def test_check_for_duplicate_observations_invalid():
    """Test fails when duplicate rows exist."""
    df_with_duplicates = sample_data.copy()

    # Append a duplicate of the first row
    df_with_duplicates = pd.concat(
        [df_with_duplicates, df_with_duplicates.iloc[[0]]],
        ignore_index=True
    )

    validator = DataValidator(df_with_duplicates)

    with pytest.raises(DataValidationError) as excinfo:
        validator.check_for_duplicate_observations()

    assert "duplicate observations" in str(excinfo.value)

# 7. check_for_outliers
def test_check_for_outliers_valid():
    """Test passes when no outliers exist."""
    validator = DataValidator(sample_data)
    validator.check_for_outliers()

def test_check_for_outliers_invalid():
    """Test fails when outliers exist in numeric columns."""
    df_outlier = sample_data.copy()

    # Inject a clear outlier
    df_outlier.loc[len(df_outlier)] = df_outlier.iloc[0]
    df_outlier.loc[len(df_outlier) - 1, "capital-gain"] = 10_000_000

    validator = DataValidator(df_outlier)

    with pytest.raises(DataValidationError) as excinfo:
        validator.check_for_outliers()

    error_msg = str(excinfo.value)
    assert "capital-gain" in error_msg

# 8. check_category_levels
def test_check_category_levels_valid():
    """Test passes when all category levels are valid."""
    validator = DataValidator(sample_data)
    validator.check_category_levels()
    
def test_check_category_levels_invalid_single_column():
    """Test fails when invalid category level exists."""
    df_invalid = sample_data.copy()

    # Inject an invalid category
    df_invalid.loc[0, "workclass"] = "Alien-gov"

    validator = DataValidator(df_invalid)

    with pytest.raises(DataValidationError) as excinfo:
        validator.check_category_levels()

    error_msg = str(excinfo.value)
    assert "workclass" in error_msg

def test_check_category_levels_multiple_columns():
    df_invalid = sample_data.copy()

    df_invalid.loc[0, "education"] = "UnknownDegree"
    df_invalid.loc[1, "race"] = "Martian"

    validator = DataValidator(df_invalid)

    with pytest.raises(DataValidationError) as excinfo:
        validator.check_category_levels()

    msg = str(excinfo.value)
    assert "education" in msg
    assert "race" in msg

# 9. check_target_distribution
expected_dist = {"<=50K": 0.80, ">50K": 0.20}
def test_check_target_distribution_valid():
    """Test passes when target distribution matches expected."""
    df_valid = sample_data.copy()

    # Force 80/20 distribution (4 <=50K, 1 >50K)
    df_valid = pd.concat(
        [
            df_valid.iloc[[0]].assign(income="<=50K"),
            df_valid.iloc[[1]].assign(income="<=50K"),
            df_valid.iloc[[2]].assign(income="<=50K"),
            df_valid.iloc[[0]].assign(income="<=50K"),
            df_valid.iloc[[1]].assign(income=">50K"),
        ],
        ignore_index=True,
    )

    validator = DataValidator(df_valid)
    # Should not raise
    validator.check_target_distribution(expected_dist)

def test_check_target_distribution_invalid():
    """Test fails when target distribution deviates beyond tolerance.""" 
    validator = DataValidator(sample_data)   
    with pytest.raises(DataValidationError) as excinfo:
        validator.check_target_distribution(expected_dist)

    msg = str(excinfo.value)
    assert "<=50K" in msg
    assert ">50K" in msg

# 10. check_target_feature_correlation
def test_check_target_feature_correlation_valid():
    """Test passes when target and feature correlation is below threshold."""
    df = pd.DataFrame({
        "age": [25, 55, 35],
        "hours-per-week": [40, 50, 38],
        "capital-gain": [0, 0, 0],
        "income": ["<=50K", ">50K", ">50K"]
    })
    validator = DataValidator(df)
    validator.threshold = 0.8
    validator.check_target_feature_correlation()

def test_check_target_feature_correlation_detects_anomaly():
    """Test passes for anomalous target and feature correlation."""
    df = pd.DataFrame({
        "age": [0, 1, 0, 1, 0],
        "hours-per-week": [0, 1, 0, 1, 0],  # perfectly correlated
        "income": ["<=50K", ">50K", "<=50K", ">50K", "<=50K"]
    })

    validator = DataValidator(df)
    validator.threshold = 0.8

    with pytest.raises(DataValidationError) as exc:
        validator.check_target_feature_correlation()

    assert "Anomalous correlations detected" in str(exc.value)

# 11. check_feature_correlations
def test_check_feature_correlations_valid():
    """Test passes when no feature correlations exceed threshold."""
    df = sample_data_weak_corr.copy()
    
    # Ensure weak correlation
    # df["capital-gain"] = [10, 200, 300]
    # df["capital-loss"] = [10, 50, 20]
    # df["hours-per-week"] = [40, 13, 40]

    validator = DataValidator(df)

    # Should not raise
    validator.check_feature_correlations()

def test_check_feature_correlations_invalid():
    """Test fails when numeric features are highly correlated."""
    df = sample_data.copy()

    # Create strong correlation
    df["capital-gain"] = [100, 200, 300]
    df["capital-loss"] = [200, 400, 600]  # perfectly correlated

    validator = DataValidator(df)

    with pytest.raises(DataValidationError) as excinfo:
        validator.check_feature_correlations()

    msg = str(excinfo.value)
    assert "capital-gain" in msg
    assert "capital-loss" in msg