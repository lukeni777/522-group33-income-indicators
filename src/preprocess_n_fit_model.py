import click
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression


@click.command()
@click.option('--in_train_file', default="data/processed/adult_census_training_data.csv")
@click.option('--in_test_file', default="data/processed/adult_census_test_data.csv")
@click.option('--out_dir', default="results", help="Base results directory")
def main(in_train_file, in_test_file, out_dir):
    # Setup directories
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

    print("Extract from the training data.")
    print(X_train.head())
    X_train.head().to_csv(os.path.join(tab_dir, "adult_df_head.csv"), index=False)
    print("Saved extract to results folder")

    # -----------------------Preprocessing-----------------------
    # Features
    numeric_features = ["age","hours-per-week", "education-num"]
    categorical_features = ["workclass", "marital-status", "occupation", "native-country"]
    binary_features = ["sex"]
    drop_features = ["fnlwgt","education", "relationship", "race", "capital-gain", "capital-loss"]
    capital_features = ["capital-gain", "capital-loss"]    

    # Transformers
    numeric_transformer = StandardScaler()
    capital_transformer = FunctionTransformer(binary_flag, feature_names_out="one-to-one")
    binary_transformer = OneHotEncoder(drop="if_binary", dtype=int)
    categorical_transformer = make_pipeline(
                SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='Unknown'),
                OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            )

    # Preprocessor
    preprocessor = make_column_transformer(
        (numeric_transformer, numeric_features),
        (categorical_transformer, categorical_features),
        (binary_transformer, binary_features),
        (capital_transformer, capital_features),
        ("drop", drop_features)
    )
    preprocessor    
    # using pickle to save preprocessor    
    pickle.dump(preprocessor, open(os.path.join(mod_dir,"income_preprocessor.pickle"), "wb"))
    print("Preprocessor saved to pickle file")

    # tune model with preprocessing + LogisticRegression
    model = make_pipeline(
        preprocessor,
        LogisticRegression(class_weight='balanced', max_iter=1000, random_state=522, C=10)
    )

    # -----------------------Fitting-----------------------
    # fit model on the entire training set
    model_fit = model.fit(X_train, y_train)

    with open(os.path.join(mod_dir, "income_pipeline.pickle"), 'wb') as f:
        pickle.dump(model_fit, f)
    print("Fitted model saved to pickle file")

def binary_flag(x):
    # Binary conversion for captital features
    return (x > 0).astype(int)

if __name__ == "__main__":
    main()