import click
import pandas as pd
import altair as alt
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import altair_ally as aly
import numpy as np

# Enable saving larger charts if necessary
alt.data_transformers.disable_max_rows()

def load_data(in_file: str) -> pd.DataFrame:
    """
    Load the input CSV file for exploratory data analysis (EDA).

    Parameters
    ----------
    in_file : str
        Path to the input CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    print("Loading data for EDA...")
    return pd.read_csv(in_file)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic preprocessing on the dataset.

    This includes:
    - Simple imputation of missing values
    - Casting selected numerical features to integer types

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe.
    """
    simple_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    df_imp = pd.DataFrame(
        simple_imp.fit_transform(df),
        columns=df.columns,
        index=df.index,
    )

    # Recast numeric features
    df_imp = df_imp.astype({
        "age": "int64",
        "fnlwgt": "int64",
        "hours-per-week": "int64",
    })

    return df_imp

def create_visualizations(df: pd.DataFrame, fig_dir: str):
    #-----------------------EDA Visualisations-----------------------

    # Plot 1 - Univariate Distribution of Quantitative Features
    os.makedirs(fig_dir, exist_ok=True)

    # Plot 1 - Quantitative
    print("Saving Distribution of Quantitative Features plot......")
    chart1 = aly.dist(df, color="income").properties(
        title="Distribution of Quantitative Features"
    )
    chart1.save(os.path.join(fig_dir, "quantitative_distribution.png"))
    chart1.show()

    # Plot 2 - Categorical
    print("Saving Distribution of Categorical Features plot......")
    chart2 = aly.dist(
        df.select_dtypes(include="object").drop(
            columns=[
                "relationship", "sex", "education-num", "race",
                "native-country", "income_encoded", "capital-gain", "capital-loss"
            ],
            errors="ignore",  # safer
        ),
        dtype="object",
        color="income",
    ).properties(title="Distribution of Categorical Features")

    chart2.save(os.path.join(fig_dir, "categorical_distribution.png"))
    chart2.show()


@click.command()
@click.option('--in_file', default="data/processed/adult_census_training.csv", help="Input raw file")
@click.option('--out_dir', default="results", help="Base results directory")
def main(in_file, out_dir):
    """
    CLI entry point for running EDA.

    Loads the dataset, preprocesses it, and generates visualizations.

    Parameters
    ----------
    in_file : str
        Path to the input CSV file.
    out_dir : str
        Base output directory for results.

    Returns
    -------
    None
    """
    # Setup directories
    fig_dir = os.path.join(out_dir, "figures")

    df = load_data(in_file)
    df_processed = preprocess_data(df)
    create_visualizations(df_processed, fig_dir)



if __name__ == "__main__":
    main()