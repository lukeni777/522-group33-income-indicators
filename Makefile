# Define the shell to ensure consistency
SHELL := /bin/bash

.PHONY: all step_read_data step_data_validation step_EDA step_modelling

all: step_read_data step_data_validation step_EDA step_modelling

# 1. Read Data
step_read_data: src/read_data.py
	python src/read_data.py --out_file="data/raw/adult_census_data.csv"

# 2. Validation (Depends on Read Data)
step_data_validation: src/validation.py step_read_data
	python src/validation.py --in_file="data/raw/adult_census_data.csv"

# 3. EDA (Depends on Validation)
step_EDA: src/eda.py step_data_validation
	python src/eda.py --in_file="data/processed/adult_census_training_data.csv" --out_dir="results"

# 4. Model (Depends on Read Data + Validation)
step_modelling: src/model.py step_data_validation
	python src/model.py --in_train_file="data/processed/adult_census_training_data.csv" --in_test_file="data/processed/adult_census_test_data.csv" --out_dir="results"

# 5. Explainability (Depends on Model)
# Needs to be added later

# 6. Report (Depends on ALL artifacts)
# Needs to be added later
