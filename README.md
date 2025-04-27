# INF367A Project: Spaceship Titanic

This is our project repository for our INF367A project: **Spaceship Titanic**.

Here aim to use given tabular data on passengers to predict which of the passengers on the Spaceship Titanic were transported by a space anomoly. To achieve this, we implement several standard models suited to the problem, as well as two novel state-of-the-art models: GRANDE and TABM.


This project is made for the ML Kaggle competition: [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/).


## How to view and reproduce:

1. Clone this repository.
2. Install the required libraries (`requirements.txt`).
3. View **1-EDA-and-preprocessing.ipynb** notebook to see the EDA and preprocess the data.
4. View **2-models.ipynb** notebook to see the training, tuning and evaluation of our models, excepting TABM and GRANDE.
5. View **3-TABM.ipynb** and **4-GRANDE.ipynb** to see our individual implementations of a novel method.


## Project data

The Kaggle dataset includes 14 features that describe each passenger:

- **PassengerId**: Unique ID (`group_passenger`) identifying the passenger and group.
- **HomePlanet**: The planet the passenger departed from.
- **CryoSleep**: Whether the passenger opted for suspended animation during the voyage.
- **Cabin**: Cabin assignment (formatted as `deck/number/side`).
- **Destination**: The planet the passenger will be debarking to.
- **Age**: Passenger's age.
- **VIP**: If the passenger paid for VIP services.
- **RoomService**, **FoodCourt**, **ShoppingMall**, **Spa**, **VRDeck**: Expenditures at various amenities.
- **Name**: Passenger's full name.
- **Transported**: **Target variable** — Whether the passenger was transported by the anomaly, our target.

All data files are located in the `data/` directory.


## Evaluation
Evaluation is simply the accuracy of the generated predicitons: correct ÷ total predictions.


## Project Structure

- **data/**: Contains the Kaggle dataset files and the processed datasets.
- **submissions/**: Contains all submissions delivered for the Kaggle Competition.
- **1-EDA-and-preprocessing**: Code for data cleaning and feature engineering.
- **2-models**: Training some standard models.
- **3-TABM**: Elias TABM implementation
- **4-GRANDE**: Johannes GRANDE implementation


## Goal
- Thorough data analysis.
- Preprocess the dataset to handle missing values, feature engineering and encoding.
- Train and optimize multiple standard machine learning models to predict the target variable `Transported`.
- Implement our own novel method to the project: GRANDE and TABM.
- Evaluate models and submit to Kaggle!



# TABM Description:

This project includes Elias' implementation of **TABM** which is a simple but powerful method for tabular deep learning.

TABM makes a single MLP behave like an ensemble of many MLPs by sharing most weights and producing multiple predictions per input. It's inspired by BatchEnsemble but tuned specifically for tabular tasks.
Thanks to the weight sharing, TABM gets better performance, faster training, and smaller models compared to traditional deep ensembles or transformer-style models.

#### Key ideas:
- Multiple predictions per sample, trained together, we choose the amount by defining **k**.

- Heavy weight sharing to keep it efficient and faster.

- You get a strong generalization from the ensemble structure.

### My Implementation:

For my implementation, I build on the authors' own PyTorch codebase.
I load and split the preprocessed data (see 1-EDA-and-preprocessing.ipynb) into DataLoaders, define a setup function that handles various model parameters, and implement the training loop.
The model is trained for 30 trials with a random assortment of parameters in each trial.
The best set of parameters (based on validation accuracy) is then chosen to train on the full training set.
Finally, I evaluate the final TABM model and compare it to scores achieved by other models.

#### Info:
You can see the full implementation in the **3-TABM.ipynb** notebook.

This work builds on code from the authors' official repository: [TabM GitHub Repo](https://github.com/yandex-research/tabm).

You can read the paper on the novel model here: [TabM Paper](https://arxiv.org/abs/2410.24210).


# GRANDE: Gradient-Based Decision Tree Ensembles for Tabular Data

## Spaceship Titanic – GRANDE Implementation

This repository demonstrates the application of the GRANDE (Gradient-Based Decision Tree Ensembles) paper on Kaggle’s Spaceship Titanic dataset in a fully reproducible Jupyter notebook (`4-GRANDE.ipynb`).

## Overview

**GRANDE** is a novel, end-to-end gradient-based method for learning hard, axis-aligned decision tree ensembles on tabular data. It combines:

- **Axis-aligned splits** for strong inductive bias on tabular features  
- **Dense, differentiable tree representation** with a straight-through estimator  
- **Instance-wise estimator weighting** to encourage both simple and complex local rules  
- **Regularization** via feature- and data-subsetting, plus dropout on trees  

## Notebook Structure

- `4-GRANDE.ipynb` — end-to-end notebook  
  1. **Data loading & preprocessing** — same pipeline as our baseline models (see `1-EDA-and-preprocessing.ipynb`)
  2. **Optuna HPO** — 20-trial tuning of GRANDE’s built-in search space  
  3. **Final training** — retrain with best hyperparameters  
  4. **Evaluation** — compute held-out validation accuracy  
  5. **Submission** — generate Space Titanic predictions  

## Results

On a 20-trial Optuna study, we achieved a best CV accuracy of 81.66% with:
```
{
  "depth": 7,
  "n_estimators": 1289,
  "learning_rate_weights": 0.0123,
  "learning_rate_index": 0.1572,
  "learning_rate_values": 0.0454,
  "learning_rate_leaf": 0.1181,
  "cosine_decay_steps": 0,
  "dropout": 0,
  "selected_variables": 0.75,
  "data_subset_fraction": 1.0,
  "focal_loss": False,
  "temperature": 0.25
}
```

We generated predictions on the processed test set using our final GRANDE model. Uploading the predictions to Kaggle yielded an accuracy of 0.80009 (~80.01%). Putting us in a competitive position in the leaderboard.

## Links
- [GRANDE paper](https://arxiv.org/abs/2309.17130)
- [GRANDE GitHub](https://github.com/s-marton/GRANDE)
