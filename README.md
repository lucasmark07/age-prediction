# NHANES Age Prediction: Senior vs. Adult

This project aims to predict whether an individual from the National Health and Nutrition Examination Survey (NHANES) dataset is a 'Senior' (65 years or older) or an 'Adult' (under 65 years old) using machine learning techniques.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Architecture Flow](#architecture-flow)
- [How to Run the Notebook](#how-to-run-the-notebook)
- [Submission Summary](#submission-summary)

## Project Overview

The National Health and Nutrition Examination Survey (NHANES) is a comprehensive health study conducted by the CDC. This project utilizes a subset of the NHANES data, focusing on key features related to body statistics, lifestyle, and lab results to classify individuals into 'Senior' or 'Adult' age groups.

## Dataset

The dataset consists of `Train_Data.csv` (for training the model) and `Test_Data.csv` (for generating predictions). The target variable, `age_group`, is binary:
- **Adult**: 0 (individuals under 65 years old)
- **Senior**: 1 (individuals 65 years old and older)

**Features include:**
- `SEQN`: Sequence number (identifier)
- `RIDAGEYR`: Age in years
- `RIAGENDR`: Respondent's Gender (1=Male, 2=Female)
- `PAQ605`: Physical activity questionnaire response
- `BMXBMI`: Body Mass Index
- `LBXGLU`: Glucose level
- `DIQ010`: Diabetes questionnaire response
- `LBXGLT`: Glucose tolerance (Oral)
- `LBXIN`: Insulin level

Missing values (`NaN`) are handled through imputation.

## Tech Stack

- **Python**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Scikit-learn**: Machine learning library (for preprocessing, model selection, and evaluation)
- **XGBoost**: Gradient Boosting framework for classification
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: For interactive development and presentation

## Architecture Flow

The project follows a standard machine learning pipeline:

1.  **Data Loading**: Raw `Train_Data.csv` and `Test_Data.csv` are loaded.
2.  **Data Preprocessing**: 
    - Missing values in numerical features are imputed using the mean.
    - Missing values in categorical features are imputed using the most frequent value.
    - Categorical features are converted into numerical format using one-hot encoding.
    - The `is_senior` target variable is created from `RIDAGEYR`.
    - The `SEQN` identifier is handled to ensure correct submission mapping.
3.  **Model Training & Evaluation**: 
    - The preprocessed training data is split into training and validation sets.
    - An XGBoost Classifier is trained.
    - Hyperparameter tuning is performed using GridSearchCV to find optimal model parameters.
    - The model's performance is evaluated using metrics like Accuracy, Precision, Recall, F1-Score, and ROC AUC.
4.  **Prediction & Submission**: 
    - The trained model makes predictions on the preprocessed test data.
    - A `submission.csv` file is generated with `SEQN` and predicted `age_group`.

```mermaid
graph TD
    A[Raw Data: Train_Data.csv & Test_Data.csv] --> B{Data Preprocessing}
    B --> C[Imputation: Mean or Mode]
    C --> D[One-Hot Encoding]
    D --> E[Feature Scaling - Optional]
    E --> F[Processed Data]
    F --> G{Model Training & Evaluation}
    G --> H[Split Data: Train/Test]
    H --> I[XGBoost Classifier]
    I --> J[Hyperparameter Tuning: GridSearchCV]
    J --> K[Trained Model: xgboost_model.pkl]
    K --> L{Prediction & Submission}
    L --> M[Generate Predictions on Test Data]
    M --> N[Create submission.csv]
    G --> O[Model Evaluation Metrics]
    O --> P[Confusion Matrix]
    O --> Q[ROC Curve]
    N --> R[Submission to Hackathon]


## How to Run the Notebook

1.  **Download Files**: Ensure you have `Train_Data.csv`, `Test_Data.csv`, and `submission_notebook.ipynb`.
2.  **Open in Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com/), click `File` -> `Upload notebook`, and select `submission_notebook.ipynb`.
3.  **Upload CSVs to Colab Session**: In Colab, click the folder icon (Files tab) on the left sidebar, then click "Upload to session storage" to upload `Train_Data.csv` and `Test_Data.csv`.
4.  **Run All Cells**: Go to `Runtime` -> `Run all` to execute the entire notebook.

This will train the model, evaluate it, and generate `submission.csv` in your Colab session.

## Submission Summary

This submission presents a machine learning model to predict whether an individual is a 'Senior' (65+) or 'Adult' (under 65) based on the National Health and Nutrition Examination Survey (NHANES) dataset. The process involved robust data preprocessing, including imputation of missing values and one-hot encoding of categorical features. An XGBoost Classifier was trained with hyperparameter tuning, demonstrating strong internal validation metrics (Accuracy, Precision, Recall, F1-Score, and ROC AUC all at 1.0000). The `submission.csv` file contains the model's predictions for the test set, formatted as required for the hackathon.
