# 522-group33-income-indicators
****

# Contributers/Authors
* Yuexiang Ni
* Michael Oyatsi
* Shruti Sasi
* Nishanth K.

# Project Summary
In this analysis, we use machine learning to predict whether an individuals income is above or below $50,000. As the government sets out massive investment in Canadian societies to improve the lives of citizens(Housing, Infrastructure and Communities Canada, 2025), we envision our analysis as a means of providing insights to the government as to what investments can drive the best chances of improving an individuals life. The persistent income and wealth inequeality increase presents a strong case for prudent investing to improve lives across all Canadians. (Yassin, Petit, & Abraham, 2024)

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

1.  **Locally** using Conda and `environment.yml`\
2.  **Virtually** using a pre-built Docker image from Docker Hub

Both options assume you start in the project root (the folder that
contains this README).

------------------------------------------------------------------------

## Option A – Run locally with Conda

### 1. Prerequisites

-   A working installation of **Conda** (Mambaforge, Miniconda, or
    Anaconda)
-   **Git** (if you want to clone the repository instead of downloading
    a ZIP)
-   Basic command line tools (Terminal on macOS / Linux, Git Bash or
    similar on Windows)

### 2. clone the repository
open your terminal and run the following commands:

``` bash
git clone https://github.com/lukeni777/522-group33-income-indicators.git
cd 522-group33-income-indicators
```

### 3. Create and activate the environment

Create the Conda environment from `environment.yml`:

``` bash
conda env create -f environment.yml
```

Activate the environment (the name is defined in `environment.yml`; in
our case it is `dsci_522_project_env`):

``` bash
conda activate dsci_522_project_env
```

### 4. Run the full analysis and render the report

We provide a `Makefile` with a convenient `all` target that:

1.  Runs the data cleaning and analysis scripts in `src/`\
2.  Saves any processed data to `data/processed/`\
3.  Generates figures and tables in `results/`\
4.  Renders the Quarto report in `report/`

From the project root, run:

``` bash
make all
```

After this finishes, you should find the rendered report at `report/report.html`.

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
docker pull lukeni777/income-indicators:latest
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
in the terminal that looks similar to:

``` text
http://127.0.0.1:8888/lab?token=...
```

Copy this URL into your web browser to open JupyterLab.

Inside the JupyterLab file browser, you should see the project files
under the `/workplace` directory. You can:
## Running the project with Docker

This project can be run in a reproducible environment using Docker.  
All commands below should be run from the project root directory

### 1. Prerequisites

- Docker installed and running on your machine
- Internet connection (if pulling the image from Docker Hub)

---

### 2. Pull the pre-built image from Docker Hub

From the project root, open a terminal and run:

```bash
# Pull the image from Docker Hub (this may take a few minutes)
docker pull lukeni777/income-indicators:latest
```
### 3. Run the Docker container
```bash
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

From the project root, open a terminal and run the following commands:

```bash
# Pull the image from Docker Hub (this may take a few minutes)
docker pull lukeni777/income-indicators:latest

# Run the container and start JupyterLab
docker run --rm -p 8888:8888 \
  -v "$PWD":/workplace \
  -w /workplace \
  lukeni777/income-indicators:latest
```
This will start a JupyterLab server inside the container and print a URL on the terminal like:

http://127.0.0.1:8888/lab

Copy this URL into your browser to open JupyterLab.

In the left file browser, you should see the project files under /workplace
(e.g., the analysis/ and data/ folders).
Open the notebook(s) in analysis/ to run the report.



# Dependencies
  - conda==23.11.0
  - python=3.9
  - pandas==2.2.1
  - jupyterlab==4.0.10
  - numpy==1.26.4
  - scikit-learn==1.4.0
  - matplotlib==3.8.2
  - shap==0.39.0
  - tabulate=0.9.0
  - pip==24.0
  - altair=5.3.0
  - ucimlrepo
  - vegafusion-python-embed=1.6.9
  - vegafusion=1.6.9 
  - vl-convert-python=1.7.0
  - altair_ally>=0.1.1 

# License Information
This project is licensed under the terms of the MIT Licence, offered under the [MIT open source license](https://opensource.org/license/MIT). See the [LICENSE.md](https://github.com/lukeni777/522-group33-income-indicators/blob/main/LICENSE) file for more information.