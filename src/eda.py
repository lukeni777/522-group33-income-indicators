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

@click.command()
@click.option('--in_file', default="data/processed/adult_census_training.csv", help="Input raw file")
@click.option('--out_dir', default="results", help="Base results directory")
def main(in_file, out_dir):
    # Setup directories
    fig_dir = os.path.join(out_dir, "figures")
    tab_dir = os.path.join(out_dir, "tables")

    print("Loading data for EDA...")
    adult_train = pd.read_csv(in_file)

    # Perform Simple Imputation
    simple_imp = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
    adult_train_imp = pd.DataFrame(simple_imp.fit_transform(adult_train), 
                                index=adult_train.index, 
                                columns=adult_train.columns)

    # Recast numerical featuers to int data types after Impute
    adult_train_imp = adult_train_imp.astype({'age':'int64',
                        'fnlwgt': 'int64',
                        'capital-gain': 'int64',
                        'capital-loss': 'int64',
                        'hours-per-week': 'int64'})

#-----------------------EDA Visualisations-----------------------

    # Plot 1 - Univariate Distribution of Quantitative Features
    print("Saving Distribution of Quantitative Features plot......")
    chart1 =  aly.dist(adult_train_imp, color='income').properties(title="Distribution of Quantitative Features")
    chart1.save(os.path.join(fig_dir, "quantitative_distribution.png"))
    chart1.show()
    
    # Plot 2 - Univariate Distribution of Categorical Features
    print("Saving Distribution of Categorical Features plot......")
    chart2 =  aly.dist(
          adult_train_imp.select_dtypes(include='object').drop(
                columns=['relationship', 'sex','education-num', 'race', 
                         'native-country']), 
         dtype='object', color='income'
         ).properties(title="Distribution of Categorical Features")
    chart2.save(os.path.join(fig_dir, "categorical_distribution.png"))
    chart1.show()


if __name__ == "__main__":
    main()