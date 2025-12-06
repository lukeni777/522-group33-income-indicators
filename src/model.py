import click
import pandas as pd
import numpy as np
import pickle
import os


@click.command()
@click.option('--in_train_file', default="data/processed/adult_census_training_data.csv")
@click.option('--in_test_file', default="data/processed/adult_census_test_data.csv")
@click.option('--out_dir', default="results", help="Base results directory")
def main(in_train_file, in_test_file, out_dir):
    # Setup directories
    adult_train = pd.read_csv(in_train_file)
    adult_test = pd.read_csv(in_test_file)

    target = "income"
    X_train = adult_train.drop(columns=target)
    X_test = adult_test.drop(columns=target)
    y_train = adult_train[target]
    y_test = adult_test[target]

    print(X_train.head())

if __name__ == "__main__":
    main()