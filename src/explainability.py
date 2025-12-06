import click
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

@click.command()
@click.option('--model_file', default="results/models/final_model.pickle", help="Path to model")
@click.option('--test_file', default="data/processed/test.csv", help="Path to test data")
@click.option('--out_dir', default="results", help="Base results directory")
def main(model_file, test_file, out_dir):
    # Setup directories
    fig_dir = os.path.join(out_dir, "figures")
    tab_dir = os.path.join(out_dir, "tables")

if __name__ == "__main__":
    main()