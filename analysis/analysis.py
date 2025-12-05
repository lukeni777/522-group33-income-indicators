#imports
import numpy as np
import pandas as pd
import altair as alt
import altair_ally as aly
import shap
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import cross_validate

# Simplify working with large dataset in altair_ally
aly.alt.data_transformers.enable('vegafusion')

# Import the data from the UCI Repostitory. 

from ucimlrepo import fetch_ucirepo

DATA_PATH = '../data/raw/adult_census_data.csv'
  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as a pandas dataframe) 
adult_df = pd.concat([adult.data.features, adult.data.targets], axis=1)

# Store raw data in the project directory 
adult_df.to_csv(DATA_PATH)

# Rename target values
adult_df.income = adult_df.income.replace(to_replace=['<=50K.', '>50K.'], value=['<=50K','>50K'])

# Combine all married groups in marital status to one group. 
# adult_df['marital-status'] = adult_df['marital-status'].replace(to_replace=r'^Married\b.*', value='Married', regex=True)

# Remove duplicate rows
adult_df = adult_df.drop_duplicates()

# Remove outliers in capital-gain and capital-loss
numeric_cols = ['capital-gain','capital-loss']
for col in numeric_cols:
    if adult_df[col].nunique() <= 2:
                continue   # skip zero-inflated / categorical numeric columns
    q1 = adult_df[col].quantile(0.25)
    q3 = adult_df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
        
    adult_df = adult_df[(adult_df[col] >= lower) & (adult_df[col] <= upper)]

# Remove anomalies in categorical columns (presence of '?')
adult_df = adult_df.replace('?', np.nan)

# Drop null values from the data
adult_df = adult_df.dropna()

# Display First observations of the dataset
adult_df.head(5)

# --- Setup Block for Validation ---
import sys
from pathlib import Path

# Add the project root path to the system path
project_root = Path.cwd().parent 

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    print(f"Added project root ({project_root.name}) to sys.path.")

from src.validation import DataValidator, DataValidationError

expected_income_dist = {
    "<=50K": 0.80,
    ">50K": 0.20
}

# --- Validation ---

try:
    # 1. Check file existence/format
    DataValidator.check_file_format_and_existence(DATA_PATH)
    
    # 2. Run other stucture & data quality checks
    validator = DataValidator(adult_df)
    validator.validate_all(expected_income_dist)
    
    print("\n\nSUCCESS: Data passed all validation checks and is ready for analysis!")

except DataValidationError as e:
    print(f"\n=========================================================================")
    print(f"VALIDATION FAILED! Analysis Halted to Prevent Data Leakage/Errors.")
    print(f"Error Details: {e}")
    print(f"=========================================================================")
    adult_df = None
    
except Exception as e:
    # Catch any other unexpected loading errors
    print(f"An unexpected error occurred during data loading: {e}")
    adult_df = None

if adult_df is not None:
    print(f"\nProceeding with a validated DataFrame of shape: {adult_df.shape}")
    
# Combine all married groups in marital status to one group. 
adult_df['marital-status'] = adult_df['marital-status'].replace(to_replace=r'^Married\b.*', value='Married', regex=True)

# Create a Test train split of the data
adult_train, adult_test = train_test_split(adult_df, test_size=0.3, random_state=522)

# Investigate quality of the data
adult_train.info()

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

# Confirm all missing values have been imputed
adult_train_imp.info()

# Store Cleaned Data in processed data directory
adult_train.to_csv('../data/processed/adult_census_training_data.csv')
adult_test.to_csv('../data/processed/adult_census_test_data.csv')

# EDA 

# Univariate Distribution of the quantitative variables
aly.dist(adult_train_imp, color='income')

# Univariate Distribution of the categorical variables

aly.dist(adult_train_imp.select_dtypes(include='object').drop(columns=['relationship', 'sex',
                                                                       'education-num', 'race', 
                                                                       'native-country']), 
         dtype='object', color='income')

# Feature & Model Selection
# Pre-processing pipeline

def binary_flag(x):
    # Binary conversion for captital features
    return (x > 0).astype(int)

# Features
numeric_features = ["age","hours-per-week", "education-num"]
categorical_features = ["workclass", "marital-status", "occupation", "native-country"]
binary_features = ["sex"]
drop_features = ["fnlwgt","education", "relationship", "race", "capital-gain", "capital-loss"]
capital_features = ["capital-gain", "capital-loss"]
target = "income"

# Transformers
numeric_transformer = StandardScaler()
capital_transformer = FunctionTransformer(binary_flag, feature_names_out="one-to-one")
binary_transformer = OneHotEncoder(drop="if_binary", dtype=int)
categorical_transformer = make_pipeline(
            SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='Unknown'),
            # SimpleImputer(missing_values = np.nan, strategy='most_frequent'), 
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

# splitting features and target
X_train = adult_train.drop(columns=target)
X_test = adult_test.drop(columns=target)
y_train = adult_train[target]
y_test = adult_test[target]

# creating a pipeline with preprocessing + LogisticRegression
model = make_pipeline(
    preprocessor,
    LogisticRegression(class_weight='balanced', max_iter=1000, random_state=522, C=10)
)

# fit model on the entire training set
model.fit(X_train, y_train)

# Logistic regression
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
cross_val_results['model']

# Confusion matrix for the logistic regression
confmat_logreg = ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    normalize='all'
)

# Show the matrix
print(confmat_logreg)

# Classification Report

print("Classification Report: ")
pd.DataFrame(classification_report(
    y_test, 
    model.predict(X_test), 
    target_names=['<=50K', '>50K'],
    output_dict=True
)).transpose()

# A. Extract steps from the pipeline
log_reg_model = model.named_steps['logisticregression']
preprocessor_step = model.named_steps['columntransformer']

# B. Transform Data for SHAP
X_train_transformed = preprocessor_step.transform(X_train)
X_test_transformed = preprocessor_step.transform(X_test)

# C. Get Feature Names
feature_names = preprocessor_step.get_feature_names_out()

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
    feature_names=feature_names, 
    max_display=5 # Top 5 important features
)