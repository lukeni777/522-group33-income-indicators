# Define the shell to ensure consistency
SHELL := /bin/bash

.PHONY: all cl clean scratch deploy

# Target: Run the whole analysis (builds the report)
all: report/income-predictor-report.html report/income-predictor-report.pdf

cl: 
	conda-lock lock \
		--file environment.yaml \
		-p linux-64 \
		-p osx-64 \
		-p osx-arm64 \
		-p win-64 \
		-p linux-aarch64 \
		--no-mamba

# 1. Read Data
data/raw/adult_census_data.csv: src/read_data.py
	python src/read_data.py --out_file="data/raw/adult_census_data.csv"

# 2. Validation & Splitting
data/processed/adult_census_training_data.csv data/processed/adult_census_test_data.csv: src/validation.py data/raw/adult_census_data.csv
	python src/validation.py --in_file="data/raw/adult_census_data.csv"

# 3. EDA
results/figures/categorical_distribution.png results/figures/quantitative_distribution.png: src/eda.py data/processed/adult_census_training_data.csv
	python src/eda.py --in_file="data/processed/adult_census_training_data.csv" --out_dir="results"

# 4. Preprocess and Fit Model
results/models/preprocessor.pickle results/models/income_pipeline.pickle: src/preprocess_n_fit_model.py data/processed/adult_census_training_data.csv data/processed/adult_census_test_data.csv
	python src/preprocess_n_fit_model.py --in_train_file="data/processed/adult_census_training_data.csv" --in_test_file="data/processed/adult_census_test_data.csv" --out_dir="results"

# 5. Evaluate Model
results/tables/income_indicator_classification_report.csv results/figures/income_indicator_confusion_matrix.png results/tables/income_indicator_score.csv: src/evaluate_model.py results/models/income_pipeline.pickle data/processed/adult_census_training_data.csv data/processed/adult_census_test_data.csv
	python src/evaluate_model.py --in_train_file="data/processed/adult_census_training_data.csv" --in_test_file="data/processed/adult_census_test_data.csv" --out_dir="results"

# 6. Explainability
results/figures/income_indicator_explainability.png: src/explainability.py results/models/income_pipeline.pickle data/processed/adult_census_training_data.csv data/processed/adult_census_test_data.csv
	python src/explainability.py --in_train_file="data/processed/adult_census_training_data.csv" --in_test_file="data/processed/adult_census_test_data.csv" --out_dir="results"

# 7a. Report
report/income-predictor-report.html: report/income-predictor-report.qmd results/tables/income_indicator_classification_report.csv results/figures/income_indicator_confusion_matrix.png results/tables/income_indicator_score.csv results/figures/categorical_distribution.png results/figures/quantitative_distribution.png results/figures/income_indicator_explainability.png
	quarto render report/income-predictor-report.qmd --to html

# 7b. Report (PDF via Typst)
report/income-predictor-report.pdf: report/income-predictor-report.qmd results/tables/income_indicator_classification_report.csv results/figures/income_indicator_confusion_matrix.png results/tables/income_indicator_score.csv results/figures/categorical_distribution.png results/figures/quantitative_distribution.png results/figures/income_indicator_explainability.png
	quarto render report/income-predictor-report.qmd --to typst

# 8. Deploy (GitHub Pages)
docs/index.html: report/income-predictor-report.html
	mkdir -p docs
	cp report/income-predictor-report.html docs/index.html
	
# Clean
clean:
	rm -f results/figures/*
	rm -f results/models/*
	rm -f results/tables/*
	rm -f report/income-predictor-report.html
	rm -f report/income-predictor-report.pdf
	rm -rf docs

# scratch
scratch:
	make clean
	python src/read_data.py --out_file="data/raw/adult_census_data.csv"
	python src/validation.py --in_file="data/raw/adult_census_data.csv"
	python src/eda.py --in_file="data/processed/adult_census_training_data.csv" --out_dir="results"
	python src/preprocess_n_fit_model.py --in_train_file="data/processed/adult_census_training_data.csv" --in_test_file="data/processed/adult_census_test_data.csv" --out_dir="results"
	python src/evaluate_model.py --in_train_file="data/processed/adult_census_training_data.csv" --in_test_file="data/processed/adult_census_test_data.csv" --out_dir="results"
	python src/explainability.py --in_train_file="data/processed/adult_census_training_data.csv" --in_test_file="data/processed/adult_census_test_data.csv" --out_dir="results"
	quarto render report/income-predictor-report.qmd --to html
	quarto render report/income-predictor-report.qmd --to typst
	make deploy
