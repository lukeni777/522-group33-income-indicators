# 522-group33-income-indicators
****

# Contributers/Authors
* Yuexiang Ni
* Michael Oyatsi
* Shruti Sasi
* Nishanth K.

# Project Summary
In this analysis, we use machine learning to predict whether an individuals income is above or below $50,000. As the government sets out massive investment in Canadian societies to improve the lives of citizens(Housing, Infrastructure and Communities Canada, 2025), we envision our analysis as a means of providing insights to the government as to what investments can drive the best chances of improving an individuals life. The persistent income and wealth inequeality increase presents a strong case for prudent investing to improve lives across all Canadians. (Yassin, Petit, & Abraham, 2024)

# How to Run the Data Analysis
To replicate our analysis on your machine:
1. Clone this GitHub Repository on your local machine:
   * Click the green ``` Code <> ``` button and copy the URL.
   * On your local machine's terminal, navigate to the location where you would like this repository to reside in.
   * Run the command ``` git clone <URL> ``` in the terminal.
2. Creating the virtual environment
   * Navigate to the cloned repository on your machine. Ensure you are in the root of the repository.
   * Run the command ``` conda env create --file environment.yaml ``` in the terminal. Or use ``` conda-lock install --name DSCI_522_project_env conda-lock.yml ``` which might be faster if you have conda-lock installed on your machine.
   * Activate the virtual environment by running the following command in the terminal: ``` conda activate DSCI_522_project_env ```


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