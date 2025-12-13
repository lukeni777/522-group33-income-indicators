import click
import pandas as pd
import pickle
import os
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

def cross_val_model_results(estimator, X_train, y_train, scoring):
    """
    Perform cross-validation and return summary statistics of scoring metrics.
    
    This function conducts k-fold cross-validation on a given estimator and returns
    a summary DataFrame containing the mean and standard deviation of training and
    test scores for specified metrics across all folds.
    
    Parameters
    ----------
    estimator : estimator object
        A scikit-learn compatible estimator implementing 'fit' and 'predict' methods.
        Can be any classifier or regressor (e.g., LogisticRegression, RandomForestClassifier).
    X_train : pd.DataFrame
        Training data features with shape (n_samples, n_features).
    y_train : array-like
        Training data target values with shape (n_samples,).
    scoring : str or dict
        Scoring metric(s) to evaluate cross-validation performance.
        - If str: A single metric (e.g., 'accuracy', 'f1', 'roc_auc', 'neg_mean_squared_error')
        - If dict: Multiple metrics with custom names (e.g., {'acc': 'accuracy', 'f1': 'f1_weighted'})
        See scikit-learn's scoring parameter documentation for valid metric strings.
    
    Returns
    -------
    pd.DataFrame
        A transposed DataFrame with metrics as rows and statistics as columns ('mean', 'std').
        Rows include timing metrics ('fit_time', 'score_time') and score metrics 
        (e.g., 'test_score', 'train_score' for single metric, or 'test_<metric_name>', 
        'train_<metric_name>' for multiple metrics). All values are rounded to 3 decimal places.
    
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> 
    >>> # Single scoring metric
    >>> X, y = make_classification(n_samples=100, random_state=42)
    >>> model = RandomForestClassifier(random_state=42)
    >>> results = cross_val_model_results(model, X, y, scoring='accuracy')
    >>> print(results)
                  mean    std
    fit_time     0.045  0.003
    score_time   0.003  0.001
    test_score   0.920  0.025
    train_score  1.000  0.000
    
    >>> # Multiple scoring metrics
    >>> scoring = {'accuracy': 'accuracy', 'f1': 'f1_weighted', 'roc_auc': 'roc_auc'}
    >>> results = cross_val_model_results(model, X, y, scoring=scoring)
    >>> print(results)
                      mean    std
    fit_time         0.045  0.003
    score_time       0.005  0.001
    test_accuracy    0.920  0.025
    train_accuracy   1.000  0.000
    test_f1          0.918  0.027
    train_f1         1.000  0.000
    test_roc_auc     0.975  0.015
    train_roc_auc    1.000  0.000
    
    Notes
    -----
    - Uses scikit-learn's default 5-fold cross-validation
    - Training scores are included via return_train_score=True
    - Useful for comparing multiple models or hyperparameter configurations
    - For regression tasks, use scoring metrics like 'neg_mean_squared_error' or 'r2'
    
    See Also
    --------
    sklearn.model_selection.cross_validate : The underlying cross-validation function
    sklearn.metrics : Available scoring metrics
    """
    return pd.DataFrame(cross_validate(
            estimator=estimator,
            X=X_train,
            y=y_train,
            return_train_score=True,
            scoring=scoring
        )).agg(['mean', 'std']).round(3).T


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
    cross_val_results['model'] = cross_val_model_results(estimator=model, 
                                                         X_train=X_train, 
                                                         y_train=y_train, 
                                                         scoring=classification_metrics)

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