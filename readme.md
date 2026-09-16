# Crop Yield Prediction Using Machine Learning

## Overview

This project uses **Multiple Linear Regression** to predict crop yield based on:

- Temperature
- Rainfall
- Fertilizer Usage
- Pesticide Usage

The project includes data exploration, visualization, model training, evaluation, and prediction.

## Dataset

The dataset contains **1,000 records and 9 columns**.

The model uses:

Temperature
Rainfall
Fertilizer_Usage
Pesticide_Usage

Target:

Yield

## Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pickle

## Model

The project uses **Multiple Linear Regression**.

The data was split into:

- 80% training data
- 20% testing data

## Results

Metric Result
Mean Squared Error (MSE): 0.3346
Root Mean Squared Error (RMSE): 0.5785
R-squared Score (R²): 0.9964

## Prediction Example

For the following conditions:

Temperature: 25
Rainfall: 150
Fertilizer Usage: 200
Pesticide Usage: 30

The model predicted:

8.17 tonnes per hectare

## Project Structure

Crop Yield Prediction Model/
│
├── src/
│ └── crop_yield_prediction.py
│
├── models/
│ └── crop_yield_model.pkl
│
├── outputs/
│ ├── yield_distribution.png
│ ├── temperature-vs-yield.png
│ ├── rainfall-vs-yield.png
│ ├── fertilizer-usage-vs-yield.png
│ ├── pesticide-usage-vs-yield.png
│ ├── correlation_heatmap.png
│ ├── actual_vs_predicted.png
│ └── residuals.png
│
├── README.md
├── requirements.txt
└── .gitignore

## How to Run

Install the required libraries:
pip install -r requirements.txt

Run the project:
python src/crop_yield_prediction.py

## What I Learned

- Data cleaning and inspection
- Exploratory Data Analysis
- Data visualization
- Feature selection
- Train-test splitting
- Multiple Linear Regression
- Model evaluation
- Model saving and loading
- Making predictions with a trained model

## Limitations

The dataset contains some unusual values, including negative rainfall and yield observations. Therefore, the results should be viewed as part of a Machine Learning learning project rather than real-world agricultural prediction.

## Author

**Frankline Oyoo**

Data Science | Analytics | Machine Learning

GitHub: `github.com/Oyoo-Frankline`
