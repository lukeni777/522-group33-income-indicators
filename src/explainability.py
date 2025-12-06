import click
import pandas as pd
import pickle
import os
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

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

    # read fitted model from pickle file
    with open(os.path.join(mod_dir, "income_pipeline.pickle"), 'rb') as f:
        model = pickle.load(f)

    # A. Extract steps from the pipeline
    log_reg_model = model.named_steps['logisticregression']
    preprocessor_step = model.named_steps['columntransformer']

    # B. Transform Data for SHAP
    X_train_transformed = preprocessor_step.transform(X_train)
    X_test_transformed = preprocessor_step.transform(X_test)

    # C. Get Feature Names
    feature_names = preprocessor_step.get_feature_names_out()
    feature_names_map = {
        "pipeline__marital-status_Married":"Married",
        "pipeline__marital-status_Never-married":"Unmarried",
        "standardscaler__education-num":"Education level",
        "standardscaler__age":"Age",
        "standardscaler__hours-per-week":"Hours per week"
    }
    readable_feature_names = [feature_names_map.get(f, f) for f in feature_names]

    # D. Create Explainer
    X_train_summary = shap.sample(X_train_transformed, 100)
    explainer = shap.LinearExplainer(log_reg_model, X_train_summary)

    # E. Calculate SHAP Values
    shap_values = explainer.shap_values(X_test_transformed)

    # F. Handle Binary Classification Output - Class 1 (>50K)
    if isinstance(shap_values, list):
        vals_to_plot = shap_values[1]
    else:
        vals_to_plot = shap_values

    # G. Plot Summary with only Top 5 Features
    plt.figure(figsize=(10, 6))
    plt.title("Top 5 High Impact Features (SHAP)")

    shap.summary_plot(
        vals_to_plot, 
        X_test_transformed, 
        feature_names=readable_feature_names, 
        max_display=5, # Top 5 important features
        show=False
    )

    plt.xlabel("SHAP Value (Impact on Model Output)")
    plt.savefig(os.path.join(fig_dir,"income_indicator_explainability.png"))
    plt.close()

    print("Saved model explainability plot under results")

# required for loading fitted model from pickle file
def binary_flag(x):
    # Binary conversion for captital features
    return (x > 0).astype(int)

if __name__ == "__main__":
    main()