# Crop Yield Prediction Using Machine Learning

## About the Project

I built this project to explore how machine learning can be used to predict crop yield from environmental and agricultural factors.

I used **Multiple Linear Regression** to predict yield based on:

* Temperature
* Rainfall
* Fertilizer usage
* Pesticide usage

The project covers data exploration, visualization, model training, evaluation, and making predictions with the trained model.

## Dataset

The dataset contains **1,000 records and 9 columns**.

The model uses four features:

* `Temperature`
* `Rainfall`
* `Fertilizer_Usage`
* `Pesticide_Usage`

**Target:** `Yield`

## Tools

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Pickle

## Model

I used **Multiple Linear Regression** and split the data into:

* **80% training**
* **20% testing**

### Results

| Metric                         | Result |
| ------------------------------ | -----: |
| Mean Squared Error (MSE)       | 0.3346 |
| Root Mean Squared Error (RMSE) | 0.5785 |
| R² Score                       | 0.9964 |

### Example Prediction

For:

* Temperature: **25**
* Rainfall: **150**
* Fertilizer Usage: **200**
* Pesticide Usage: **30**

The model predicted a yield of approximately **8.17 tonnes per hectare**.

## Project Structure

```text
Crop-Yield-Prediction-Model/
│
├── src/
│   └── crop_yield_prediction.py
│
├── models/
│   └── crop_yield_model.pkl
│
├── outputs/
│   ├── yield_distribution.png
│   ├── temperature-vs-yield.png
│   ├── rainfall-vs-yield.png
│   ├── fertilizer-usage-vs-yield.png
│   ├── pesticide-usage-vs-yield.png
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   └── residuals.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Run the Project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the prediction script:

```bash
python src/crop_yield_prediction.py
```

## What I Learned

Through this project, I practiced:

* Data cleaning and inspection
* Exploratory Data Analysis
* Data visualization
* Feature selection
* Train-test splitting
* Multiple Linear Regression
* Model evaluation
* Saving and loading trained models
* Making predictions with a trained model

## Limitations

The dataset contains some unusual values, including negative rainfall and yield observations. Because of this, I consider this project mainly a **machine learning learning project**, rather than a model ready for real-world agricultural decision-making.

## Author

**Frankline Oyoo**
