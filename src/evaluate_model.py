import click
import pandas as pd
import pickle
import os
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

@click.command()
@click.option('--in_train_file', default="data/processed/adult_census_training_data.csv")
@click.option('--in_test_file', default="data/processed/adult_census_test_data.csv")
@click.option('--out_dir', default="results", help="Base results directory")
def main(in_train_file, in_test_file, out_dir):
    # Setup directories
    fig_dir = os.path.join(out_dir, "figures")
    tab_dir = os.path.join(out_dir, "tables")
    mod_dir = os.path.join(out_dir, "models")

    # Read training and testing data
    adult_train = pd.read_csv(in_train_file)
    adult_test = pd.read_csv(in_test_file)

    target = "income"
    X_train = adult_train.drop(columns=target)
    X_test = adult_test.drop(columns=target)
    y_train = adult_train[target]
    y_test = adult_test[target]

    print("Evaluating model...")

    # read fitted model from pickle file
    with open(os.path.join(mod_dir, "income_pipeline.pickle"), 'rb') as f:
        model = pickle.load(f)

    # Accuracy, Precision, Recall, F1-Score
    classification_metrics = {
        "accuracy": "accuracy", 
        "precision": make_scorer(precision_score, pos_label=">50K"), 
        "recall": make_scorer(recall_score, pos_label=">50K"), 
        "f1": make_scorer(f1_score, pos_label=">50K")
    }

    cross_val_results = {}
    cross_val_results['model'] = pd.DataFrame(cross_validate(
        model,
        X_train,
        y_train,
        return_train_score=True,
        scoring=classification_metrics
    )).agg(['mean', 'std']).round(3).T

    # Show the train and validation scores
    print("Test Score: ", model.score(X_test, y_test))
    cross_val_results['model'].to_csv(os.path.join(tab_dir,"income_indicator_score.csv"), index=True)
    
    # Confusion matrix for the logistic regression
    confmat_logreg = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        normalize='all'
    )
    plt.savefig(os.path.join(fig_dir,"income_indicator_confusion_matrix.png"))
    plt.close()
    print("Confusion Matrix saved under results")

    # Classification Report
    pd.DataFrame(classification_report(
        y_test, 
        model.predict(X_test), 
        target_names=['<=50K', '>50K'],
        output_dict=True
    )).transpose().to_csv(os.path.join(tab_dir, "income_indicator_classification_report.csv"), index=True)    
    print("Classification Report saved under results")

# required for loading fitted model from pickle file
def binary_flag(x):
    # Binary conversion for captital features
    return (x > 0).astype(int)

if __name__ == "__main__":
    main()