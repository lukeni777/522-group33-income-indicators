import sys
import os
import pytest
import pandas as pd
import numpy as np
from click.testing import CliRunner
from unittest.mock import patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from src.eda import (
    load_data,
    preprocess_data,
    create_visualizations,
    main,
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [39, np.nan, 53],
        "fnlwgt": [77516, 215646, 234721],
        "hours-per-week": [40, 13, 40],
        "workclass": ["State-gov", "Private", "Private"],
        "income": ["<=50K", "<=50K", ">50K"],
    })

@pytest.fixture
def sample_csv(sample_df, tmp_path):
    path = tmp_path/"test_adult.csv"
    sample_df.to_csv(path, index=False)
    return path 

def test_load_data(sample_csv):
    df = load_data(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_preprocess_data(sample_df):
    df_processed = preprocess_data(sample_df)

    # No missing values
    assert df_processed.isnull().sum().sum() == 0

    # Correct dtypes
    assert df_processed["age"].dtype == "int64"

def test_create_visualizations(sample_df, tmp_path):
    fig_dir = tmp_path/"figures"

    mock_chart = MagicMock()
    mock_chart.save.return_value = None
    mock_chart.show.return_value = None

    with patch("src.eda.aly.dist", return_value=mock_chart):
        create_visualizations(sample_df, str(fig_dir))

    assert fig_dir.exists()

def test_main_fails_when_input_missing():
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "--in_file", str("data/processed/adult_census_training.csv"),
            "--out_dir", str("results"),
        ],
    )

    assert result.exit_code != 0

def test_main_runs_successfully(sample_csv, tmp_path):
    runner = CliRunner()
    out_dir = tmp_path/"results"

    mock_chart = MagicMock()
    mock_chart.save.return_value = None
    mock_chart.show.return_value = None

    with patch("src.eda.aly.dist", return_value=mock_chart):
        result = runner.invoke(
            main,
            [
                "--in_file", str(sample_csv),
                "--out_dir", str(out_dir),
            ],
        )

    assert result.exit_code == 0
    assert "Loading data for EDA..." in result.output