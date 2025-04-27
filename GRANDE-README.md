# Spaceship Titanic Prediction with GRANDE
## Overview

This repository demonstrates how to apply the GRANDE model to the Kaggle “Spaceship Titanic” competition. We preprocess the data, tune hyperparameters with Optuna, train and validate the model, and generate a submission CSV.

**Data**

- train_processed.csv – preprocessed training set
- test_processed.csv – preprocessed test set

Note: All missing values, categorical encodings, and feature engineering steps should be applied before generating these CSVs.

## Preprocessing

- Load the processed CSV files into pandas DataFrames.
- Verify data types (df.dtypes) to ensure numeric and categorical features are correctly preserved.
- Separate features (X) and target (y), dropping PassengerId and Transported from X.

## Hyperparameter Tuning

We use Optuna with pruning to search the space:

- Define an Optuna objective that:
  - Samples hyperparameters (e.g. learning rate, depth).
  - Fits on a train split and evaluates on a validation split.
  - Reports intermediate metrics for pruning.
- Extract study.best_params.

## Training & Validation
- Split the dataset into train/validation (e.g. 80/20 stratified).
- Train the final GRANDE model on the full training set with best hyperparameters.
- Predict on the validation set and compute accuracy.

## Submission
- Reload test_processed.csv.
- Drop PassengerId, predict probabilities, threshold at 0.5 on the positive class.
- Build a DataFrame with PassengerId and Transported (bool).
- Save to submission_grande.csv.

## Results
- Validation Accuracy: ~0.84
