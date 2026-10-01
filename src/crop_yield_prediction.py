import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# Loading dataset
data_url = (
    "https://raw.githubusercontent.com/Explore-AI/Public-Data/"
    "master/Data/Python/Crop_yield.csv"
)

df = pd.read_csv(data_url)

print("Dataset loaded successfully.")
print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")


# Data inspection
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nSummary statistics:")
print(df.describe())


# Data quality checks
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# Exploratory data analysis
plt.figure(figsize=(8, 5))

plt.hist(df["Yield"], bins=30)

plt.title("Distribution of Crop Yield")
plt.xlabel("Yield")
plt.ylabel("Frequency")

plt.savefig(
    OUTPUT_DIR / "yield_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


features = [
    "Temperature",
    "Rainfall",
    "Fertilizer_Usage",
    "Pesticide_Usage"
]

for feature in features:

    plt.figure(figsize=(8, 5))

    plt.scatter(df[feature], df["Yield"])

    plt.title(f"{feature} vs Crop Yield")
    plt.xlabel(feature)
    plt.ylabel("Yield")

    filename = feature.lower().replace("_", "-") + "-vs-yield.png"

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# Correlation analysis
correlation_columns = [
    "Temperature",
    "Rainfall",
    "Fertilizer_Usage",
    "Pesticide_Usage",
    "Yield"
]

correlation_matrix = df[correlation_columns].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# features and target
X = df[
    [
        "Temperature",
        "Rainfall",
        "Fertilizer_Usage",
        "Pesticide_Usage"
    ]
]

y = df["Yield"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")

print("\nTesting data:")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")


# Training model
lm = LinearRegression()

lm.fit(X_train, y_train)

print("\nModel training completed.")


# Generating predictions
y_pred = lm.predict(X_test)

print("\nFirst 10 predictions:")
print(y_pred[:10])


# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared Score (R²): {r2:.4f}")


# Model coefficients
coefficients = pd.DataFrame(
    {
        "Feature": X.columns,
        "Coefficient": lm.coef_
    }
)

print("\nModel Coefficients:")
print(coefficients)

print(f"\nModel Intercept: {lm.intercept_:.4f}")


# Actual vs predicted
results = pd.DataFrame(
    {
        "Actual Yield": y_test.values,
        "Predicted Yield": y_pred
    }
)

print("\nActual vs Predicted Yield:")
print(results.head(10))

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title("Actual vs Predicted Crop Yield")
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Residual analysis
residuals = y_test - y_pred

plt.figure(figsize=(8, 5))

plt.scatter(y_pred, residuals)

plt.axhline(y=0, linestyle="--")

plt.title("Residuals vs Predicted Yield")
plt.xlabel("Predicted Yield")
plt.ylabel("Residuals")

plt.savefig(
    OUTPUT_DIR / "residuals.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Saving trained model
model_save_path = MODEL_DIR / "crop_yield_model.pkl"

with open(model_save_path, "wb") as file:
    pickle.dump(lm, file)

print(f"\nModel saved to: {model_save_path}")


# Loading  saved model
model_load_path = MODEL_DIR / "crop_yield_model.pkl"

with open(model_load_path, "rb") as file:
    load_model = pickle.load(file)

print("Saved model loaded successfully.")


# Predicting new crop conditions
new_conditions = {
    "Temperature": [25],
    "Rainfall": [150],
    "Fertilizer_Usage": [200],
    "Pesticide_Usage": [30]
}

new_conditions_df = pd.DataFrame(new_conditions)

print("\nNew conditions:")
print(new_conditions_df)

predicted_yield = load_model.predict(new_conditions_df)

print(
    f"\nPredicted crop yield: "
    f"{predicted_yield[0]:.2f} tonnes per hectare"
)


# Reusable prediction function
def predict_crop_yield(
    temperature,
    rainfall,
    fertilizer_usage,
    pesticide_usage
):
    new_data = pd.DataFrame(
        {
            "Temperature": [temperature],
            "Rainfall": [rainfall],
            "Fertilizer_Usage": [fertilizer_usage],
            "Pesticide_Usage": [pesticide_usage]
        }
    )

    prediction = load_model.predict(new_data)

    return prediction[0]


prediction = predict_crop_yield(
    temperature=25,
    rainfall=150,
    fertilizer_usage=200,
    pesticide_usage=30
)

print(
    f"\nPrediction using reusable function: "
    f"{prediction:.2f} tonnes per hectare"
)

print("\nCrop Yield Prediction project completed successfully.")
