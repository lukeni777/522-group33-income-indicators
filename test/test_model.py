import sys
import os
import pytest
import pandas as pd
import numpy as np
from click.testing import CliRunner

# Add the parent directory to sys.path so we can import the scripts
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import the modules to be tested
from src import preprocess_n_fit_model as prep
from src import evaluate_model as eval_mod

# ==========================================
# UNIT TESTS
# ==========================================

def test_binary_flag_positive():
    """
    Test that binary_flag returns 1 for values > 0.
    """
    input_series = pd.Series([100, 5000, 0.1])
    expected = pd.Series([1, 1, 1])
    
    result = prep.binary_flag(input_series)
    
    pd.testing.assert_series_equal(result, expected.astype(int), check_names=False)

def test_binary_flag_zero_negative():
    """
    Test that binary_flag returns 0 for values <= 0.
    """
    input_series = pd.Series([0, -100, -50])
    expected = pd.Series([0, 0, 0])
    
    result = prep.binary_flag(input_series)
    
    pd.testing.assert_series_equal(result, expected.astype(int), check_names=False)

def test_binary_flag_mixed():
    """
    Test binary_flag with a mix of positive, negative, and zero values.
    """
    input_series = pd.Series([100, 0, -5, 20])
    expected = pd.Series([1, 0, 0, 1])
    
    result = prep.binary_flag(input_series)
    
    pd.testing.assert_series_equal(result, expected.astype(int), check_names=False)

# ==========================================
# FIXTURES (Reproducible Test Data)
# ==========================================

@pytest.fixture
def raw_data():
    """
    Creates a small, reproducible dataframe matching the Adult Census schema.
    Includes enough rows (n=10) to support default 5-fold cross-validation.
    """
    data = pd.DataFrame({
        "age": [25, 40, 30, 22, 35, 60, 45, 28, 50, 32],
        "workclass": ["Private", "State-gov", "Private", "Private", "Self-emp", "Private", "Local-gov", "Private", "Federal-gov", "Private"],
        "fnlwgt": [1000, 2000, 1500, 1200, 3000, 5000, 2500, 1800, 4000, 2100],
        "education": ["Bachelors", "Masters", "HS-grad", "Some-college", "Doctorate", "HS-grad", "Masters", "Bachelors", "Prof-school", "Assoc"],
        "education-num": [13, 14, 9, 10, 16, 9, 14, 13, 15, 12],
        "marital-status": ["Never-married", "Married", "Divorced", "Never-married", "Married", "Widowed", "Separated", "Married", "Divorced", "Never-married"],
        "occupation": ["Tech", "Admin", "Sales", "Other", "Prof-specialty", "Craft", "Exec", "Sales", "Tech", "Other"],
        "relationship": ["Not-in-family", "Husband", "Unmarried", "Own-child", "Husband", "Not-in-family", "Wife", "Husband", "Unmarried", "Own-child"],
        "race": ["White", "Black", "Asian", "White", "White", "Black", "White", "Asian", "White", "Black"],
        "sex": ["Female", "Male", "Female", "Male", "Male", "Female", "Female", "Male", "Female", "Male"],
        "capital-gain": [0, 5000, 0, 0, 10000, 0, 0, 2000, 0, 0],
        "capital-loss": [0, 0, 100, 0, 0, 0, 0, 0, 50, 0],
        "hours-per-week": [40, 50, 35, 20, 60, 40, 45, 55, 38, 40],
        "native-country": ["USA", "USA", "Canada", "Mexico", "USA", "USA", "USA", "China", "Germany", "USA"],
        "income": ["<=50K", ">50K", "<=50K", "<=50K", ">50K", "<=50K", ">50K", ">50K", "<=50K", "<=50K"]
    })
    return data

@pytest.fixture
def setup_test_environment(tmp_path, raw_data):
    """
    Sets up a temporary directory environment with input CSVs 
    and output folders (results/figures, results/models, etc.).
    """
    # Define paths
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"
    
    # Save dummy data to csv
    raw_data.to_csv(train_file, index=False)
    raw_data.to_csv(test_file, index=False)
    
    # Create output directories structure inside the temp path
    results_dir = tmp_path / "results"
    os.makedirs(results_dir / "figures")
    os.makedirs(results_dir / "tables")
    os.makedirs(results_dir / "models")
    
    return train_file, test_file, results_dir

# ==========================================
# INTEGRATION TESTS
# ==========================================

def test_preprocess_and_fit_integration(setup_test_environment):
    """
    Test the preprocessing and model fitting script via CLI runner.
    Ensures input files are read and output pickle files are created.
    """
    train_path, test_path, out_dir = setup_test_environment
    runner = CliRunner()
    
    # Run the main function from preprocess_n_fit_model.py
    result = runner.invoke(prep.main, [
        '--in_train_file', str(train_path),
        '--in_test_file', str(test_path),
        '--out_dir', str(out_dir)
    ])
    
    # Check if script finished successfully with exit code 0
    assert result.exit_code == 0, f"Script failed with output: {result.output}"
    
    # Check if expected output files were created
    assert os.path.exists(out_dir / "tables" / "adult_df_head.csv")
    assert os.path.exists(out_dir / "models" / "income_preprocessor.pickle")
    assert os.path.exists(out_dir / "models" / "income_pipeline.pickle")

def test_evaluate_model_integration(setup_test_environment):
    """
    Test the evaluation script via CLI runner.
    This requires the model to be fitted first, so we run the 
    preprocess step immediately before the eval step within this test.
    """
    train_path, test_path, out_dir = setup_test_environment
    runner = CliRunner()
    
    # 1. Run Preprocessing first to generate the pickle file
    runner.invoke(prep.main, [
        '--in_train_file', str(train_path),
        '--in_test_file', str(test_path),
        '--out_dir', str(out_dir)
    ])
    
    # 2. Run Evaluation
    result = runner.invoke(eval_mod.main, [
        '--in_train_file', str(train_path),
        '--in_test_file', str(test_path),
        '--out_dir', str(out_dir)
    ])
    
    # Check if script finished successfully
    assert result.exit_code == 0, f"Evaluation script failed: {result.output}"
    
    # Check if expected evaluation artifacts were created
    assert os.path.exists(out_dir / "tables" / "income_indicator_score.csv")
    assert os.path.exists(out_dir / "figures" / "income_indicator_confusion_matrix.png")
    assert os.path.exists(out_dir / "tables" / "income_indicator_classification_report.csv")
