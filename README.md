# 522-group33-income-indicators
****

A data analysis project undertaken as part of DSCI 522; a course in the Master of Data Science program at the University of British Columbia.

# Contributers/Authors
* Yuexiang Ni
* Michael Oyatsi
* Shruti Sasi
* Nishanth K.

# Project Summary
Our team set out to infer what socioeconomic factors contribute most to an individual's wealth. With our analysis and model, we envision this data being used by government and NGOs in determining what social investments can be made to improve people's lives. 

To accomplish this, we built a classification model to predict an individual's income group, split by whether they are high earners (> USD 50,000) or low earners (<= USD 50,000). Using a Logistic Regression classifier, our model accuracy was 78% on unseen test data with an associated F1 score of 0.72. To address the class imbalance in the data, we used a balanced weight approach while building our model. We also sought to understand what socioeconomic characteristics play a the biggest role in determining an individual's income group. Using SHAP analysis, our findings show that of the features in our model,  Marital Status, Age & Education are the biggest drivers of a High Income output. 

While the Logistic Regression classifier was chosen to easier identify the socioeconomic features that are drivers of high income, we see an opportunity to use an ensemble model such as Random Forest Classification to improve the model's prediction metrics. We also note the limitation of our findings that the strongest economic indicators for wealth are limited only to the features that were available to us in the dataset. There presents an opportunity to further explore what other indicators are stronger predictors through addition of more features, or feature engineering of the present features with a subject matter expert.


# Repository Structure 

- `data/` – raw and processed data files (created by `make`)
- `src/` – Python scripts for data processing, modelling, evaluation, and explainability
- `results/` – generated figures, models, and tables (created by `make`)
- `report/` – Quarto report (`.qmd`) and rendered outputs (`.html`)
- `environment.yaml` – conda environment specification
- `Makefile` – pipeline to run the full analysis and render the report
- `Dockerfile` – recipe for the Docker image

## How to run the data analysis

You can run the project in two ways:

1.  **Locally** using Conda and `environment.yaml`\
2.  **Virtually** using a pre-built Docker image from Docker Hub

Both options assume you start in the project root (the folder that
contains this README).

Ensure that you have Git installed on your machine. First, clone the repository to your local machine by opening your terminal and run the following commands:

``` bash
git clone https://github.com/lukeni777/522-group33-income-indicators.git
cd 522-group33-income-indicators
```

now choose one of the following options:
------------------------------------------------------------------------

## Option A – Run locally with Conda

### 1. Prerequisites

-   A working installation of **Conda** (Mambaforge, Miniconda, or
    Anaconda)
-   Basic command line tools (Terminal on macOS / Linux, Git Bash or
    similar on Windows)

### 2. Create and activate the environment

Create the Conda environment from `environment.yaml`:

``` bash
conda env create -f environment.yaml
```

Activate the environment (the name is defined in `environment.yml`; in
our case it is `dsci_522_project_env`):

``` bash
conda activate dsci_522_project_env
```

### 3. Run the full analysis and render the report

We provide a `Makefile` with a convenient `all` target that:

1.  Runs the data cleaning and analysis scripts in `src/`\
2.  Saves any processed data to `data/processed/`\
3.  Generates figures and tables in `results/`\
4.  Renders the Quarto report in `report/`

From the project root, run:

``` bash
make all
```

After this finishes, you should find the rendered report at `report/income-predictor-report.html`.

To remove all generated files and start fresh, you can run:

``` bash
make clean
```

We also provide `scratch` target that removes the old files before generating new documents and reports:

```bash
make scratch
```

## Option B – Run the project with Docker

Running with Docker allows you to use a pre-built image that already
contains all required dependencies (including JupyterLab). This is
useful if you do not want to manage Conda environments locally.

### 1. Prerequisites

-   A recent version of **Docker** installed and running\
-   Internet connection (to pull the image from Docker Hub)

### 2. Pull the pre-built image from Docker Hub

From the project root, open a terminal and run:

``` bash
# Pull the image from Docker Hub (this may take a few minutes)
docker pull lukeni777/income-indicators:7b04a79
```

### 3. Run the container and start JupyterLab

From the same project root, run the following commands.\
The script below automatically handles the difference between Windows
and macOS/Linux when mounting the project directory into the container.

``` bash
# Run the container and start JupyterLab

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  # Windows (Git Bash / MSYS) – avoid path mangling
  export MSYS_NO_PATHCONV=1
  docker run --rm -p 8888:8888 \
    -v "$(pwd)":/workplace \
    -w /workplace \
    lukeni777/income-indicators:latest
  unset MSYS_NO_PATHCONV
else
  # macOS / Linux
  docker run --rm -p 8888:8888 \
    -v "$PWD":/workplace \
    -w /workplace \
    lukeni777/income-indicators:latest
fi
```

This will start a JupyterLab server inside the container and print a URL

``` text
http://127.0.0.1:8888/lab
```

Copy this URL into your web browser to open JupyterLab.

Inside the JupyterLab file browser, you should see the project files
under the `/workplace` directory. 

Open a terminal on Jupyterlab and run:

``` bash
make all
```

After this finishes, you should find the rendered report at `report/income-predictor-report.html`.

To remove all generated files and start fresh, you can run:

``` bash
make clean
```
 OR

To remove the old files before generating new documents and reports, run:

```bash
make scratch
```
---

## Data analysis pipeline

This section describes the individual steps of the data analysis pipeline
and how to run them individually if desired. The individual steps are wired
together using the `Makefile` so that the full workflow can be run with the single `make all` command above.

At a high level, the steps are:

1. **Data loading and cleaning**  
   - Reads the raw dataset from `data/raw/`.  
   - Performs cleaning (handling missing values, recoding variables, selecting
     features).  
   - Outputs a cleaned dataset into `data/processed/`.

2. **Exploratory data analysis (EDA)**  
   - Generates summary tables and visualizations to understand the distributions
     of key variables and relationships between them.  
   - Saves intermediate figures and tables into the `results/` folder.

3. **Modelling and evaluation**  
   - Fits one or more predictive models using the processed dataset.  
   - Evaluates model performance using appropriate metrics and saves results in
     `results/`.

4. **Report rendering**  
   - Renders the Quarto report in `report/`, which pulls in the figures,
     tables, and metrics produced in earlier steps.  

You can inspect the `Makefile` for the exact targets and file dependencies if
you want to run individual stages.

### Detailed pipeline steps (Makefile targets)

Below is a more detailed description of each step in the Python pipeline and
how to run it individually from the command line.

1. **Read data – `step_read_data`**  
   Reads the raw Adult Census dataset from its source and saves it locally in
   the `data/raw/` folder.

   - With `make` (recommended):

     ```bash
     make step_read_data
     ```

   - Under the hood, this runs:

     ```bash
     python src/read_data.py --out_file="data/raw/adult_census_data.csv"
     ```

2. **Validate data – `step_data_validation`**  
   Performs checks on the downloaded data (e.g., row counts, column types,
   missing values) to ensure it is suitable for analysis.

   - With `make`:

     ```bash
     make step_data_validation
     ```

   - Under the hood, this runs:

     ```bash
     python src/validation.py --in_file="data/raw/adult_census_data.csv"
     ```

3. **Exploratory data analysis – `step_EDA`**  
   Creates summary tables and visualizations to explore the distributions of
   features and their relationships with the income indicator. Artifacts are
   saved in `results/` (figures and tables).

   - With `make`:

     ```bash
     make step_EDA
     ```

4. **Pre-processing – `step_preprocess`**  
   Splits the data into training and test sets and performs feature
   engineering/encoding. The processed datasets are saved to
   `data/processed/`.

   - With `make`:

     ```bash
     make step_preprocess
     ```

5. **Model training and evaluation – `step_evaluation`**  
   Trains the predictive model(s) on the processed training data and evaluates
   performance on the test data. Metrics and evaluation tables are written to
   `results/`.

   - With `make`:

     ```bash
     make step_evaluation
     ```

6. **Model explainability – `step_explainability`**  
   Produces explainability artifacts such as feature importance plots or SHAP
   value summaries to help interpret the model.

   - With `make`:

     ```bash
     make step_explainability
     ```

7. **Report rendering – `step_report`**  
   Renders the Quarto report, which pulls together all of the above artifacts
   (figures, tables, and metrics) into a single narrative document. 

   - With `make`:

     ```bash
     make step_report
     ```

   - Under the hood, this runs:

     ```bash
     quarto render report/income-predictor-report.qmd --to html
     ```

8. **Cleaning generated files – `clean`**  
   Removes downloaded data, processed datasets, figures, models, tables, and
   the rendered HTML report so you can test full reproducibility from scratch.

   - With `make`:

     ```bash
     make clean
     ```
---
# Testing
   To verify the analysis python packages in JupyterLab, open terminal and run:

```bash 
pytest test/
```
---
# Dependencies
  - python=3.12.12
  - pandas=2.2
  - jupyterlab=4.0
  - numpy=1.26
  - scikit-learn=1.7.2
  - matplotlib=3.10.8
  - pandera-pandas=0.27.0
  - scipy=1.16.3
  - shap=0.48.0  
  - tabulate=0.9.0
  - pip=25.3
  - altair=5.5
  - vl-convert-python=1.8.0 
  - conda-lock=3.0.4
  - pip:
      - ucimlrepo==0.0.7 
      - altair_ally==0.1.1 
      - vegafusion-python-embed==1.6.9 
      - vegafusion==1.6.9 
      - quarto-cli==1.8.26
      - pytest==9.0.1
      - pytest-mock==3.15.1
      - pytest-cov==7.0.0

# License Information
This project is licensed under the terms of the MIT License, offered under the [MIT open source license](https://opensource.org/license/MIT). See the [LICENSE.md](https://github.com/lukeni777/522-group33-income-indicators/blob/main/LICENSE) file for more information.

Report text and figures: Creative Commons Attribution 4.0 International (CC BY 4.0) (see `LICENSE.md`)
