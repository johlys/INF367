# INF367A Project: Spaceship Titanic
Contributors: Johannes Krispinus Lysne and Elias Ruud Aronsen

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
