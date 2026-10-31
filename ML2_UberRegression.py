# AIM: Predict Uber ride fare prices using Linear Regression,
# Ridge Regression and Lasso Regression.

# Installation commands:
# Windows:
# python -m pip install pandas numpy matplotlib seaborn scikit-learn
# python practical_regression.py
#
# Ubuntu/Linux:
# pip install pandas numpy matplotlib seaborn scikit-learn
# python3 -m pip install pandas numpy matplotlib seaborn scikit-learn
# python3 practical_regression.py


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# Load dataset

df = pd.read_csv("uber.csv")

print("Original Dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

# Remove unnecessary columns

df = df.drop(columns=["Unnamed: 0", "key"], errors="ignore")

# Remove missing values in important columns

df = df.dropna(subset=[
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count"
])

# Convert columns into numeric values

numeric_columns = [
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=numeric_columns)

# Remove invalid records

df = df[
    (df["fare_amount"] > 0) &
    (df["passenger_count"] > 0) &
    (df["passenger_count"] <= 6) &
    (df["pickup_longitude"].between(-180, 180)) &
    (df["dropoff_longitude"].between(-180, 180)) &
    (df["pickup_latitude"].between(-90, 90)) &
    (df["dropoff_latitude"].between(-90, 90))
].copy()

# Convert pickup datetime

df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"],
    errors="coerce",
    utc=True
)

df = df.dropna(subset=["pickup_datetime"])

# Extract date and time features

df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.day
df["month"] = df["pickup_datetime"].dt.month
df["year"] = df["pickup_datetime"].dt.year
df["day_of_week"] = df["pickup_datetime"].dt.dayofweek

# Calculate distance using Haversine formula

def calculate_distance(lat1, lon1, lat2, lon2):

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2 +
        np.cos(lat1) * np.cos(lat2) *
        np.sin(dlon / 2) ** 2
    )

    distance = 6371 * 2 * np.arcsin(np.sqrt(a))

    return distance


df["distance"] = calculate_distance(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"]
)

# Remove zero-distance and unrealistic rides

df = df[
    (df["distance"] > 0) &
    (df["distance"] < 200)
].copy()

print("\nDataset After Preprocessing:")
print(df.head())

# Identify outliers using IQR method

Q1 = df["fare_amount"].quantile(0.25)
Q3 = df["fare_amount"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df["fare_amount"] < lower_limit) |
    (df["fare_amount"] > upper_limit)
]

print("\nNumber of Fare Outliers:", len(outliers))

print("\nLower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

# Remove fare outliers

df = df[
    (df["fare_amount"] >= lower_limit) &
    (df["fare_amount"] <= upper_limit)
].copy()

# Check correlation

features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "hour",
    "day",
    "month",
    "year",
    "day_of_week",
    "distance",
    "fare_amount"
]

correlation = df[features].corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()

# Select input and output variables

X = df[
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "hour",
        "day",
        "month",
        "year",
        "day_of_week",
        "distance"
    ]
]

y = df["fare_amount"]

# Split dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create regression models

models = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=1.0),

    "Lasso Regression": Lasso(alpha=0.01, max_iter=10000)
}

results = []

# Train and evaluate all models

for name, model in models.items():

    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        model
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    r2 = r2_score(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    mae = mean_absolute_error(y_test, predictions)

    results.append({
        "Model": name,
        "R2 Score": r2,
        "RMSE": rmse,
        "MAE": mae
    })

    print("\n", name)

    print("R2 Score:", round(r2, 4))

    print("RMSE:", round(rmse, 4))

    print("MAE:", round(mae, 4))


# Compare all regression models

results_df = pd.DataFrame(results)

print("\nRegression Model Comparison:")

print(results_df)

# Display the best model based on R2 score

best_model = results_df.loc[
    results_df["R2 Score"].idxmax()
]

print("\nBest Performing Model:")

print(best_model)

# Plot model comparison

plt.figure(figsize=(8, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="R2 Score"
)

plt.title("Comparison of Regression Models")

plt.ylabel("R2 Score")

plt.show()


# ============================================================
# EXPLANATION LINE BY LINE
# ============================================================

# import pandas as pd
# Imports Pandas to load and process the dataset.

# import numpy as np
# Imports NumPy for mathematical calculations.

# import matplotlib.pyplot as plt
# Imports Matplotlib to display graphs.

# import seaborn as sns
# Imports Seaborn to create the correlation heatmap.

# train_test_split
# Divides the dataset into training and testing data.

# StandardScaler
# Standardizes the input features.

# SimpleImputer
# Handles missing values in the input features.

# make_pipeline
# Combines preprocessing steps and regression models.

# LinearRegression, Ridge, Lasso
# Imports the three regression algorithms.

# r2_score, mean_squared_error, mean_absolute_error
# Imports evaluation metrics for regression models.

# df = pd.read_csv("uber.csv")
# Loads the Uber dataset from the local CSV file.

# print(df.head())
# Displays the first five rows of the dataset.

# df.info()
# Displays column names, data types and non-null counts.

# df.drop(...)
# Removes unnecessary columns such as key and Unnamed: 0.

# df.dropna(...)
# Removes records with missing values in important columns.

# pd.to_numeric(...)
# Converts the selected columns into numeric values.

# df.dropna(...)
# Removes records containing invalid numeric values.

# df = df[(...)]
# Removes invalid fares, passenger counts and geographical coordinates.

# pd.to_datetime(...)
# Converts pickup_datetime into a date and time format.

# df["hour"]
# Extracts the pickup hour.

# df["day"]
# Extracts the day of the month.

# df["month"]
# Extracts the month.

# df["year"]
# Extracts the year.

# df["day_of_week"]
# Extracts the day of the week.

# def calculate_distance(...)
# Defines a function to calculate geographical distance.

# np.radians(...)
# Converts latitude and longitude from degrees into radians.

# dlat and dlon
# Calculate the differences in latitude and longitude.

# a = (...)
# Calculates the intermediate value used in the Haversine formula.

# distance = 6371 * ...
# Calculates the distance between pickup and drop-off points in kilometres.
# 6371 represents the approximate Earth's radius in kilometres.

# return distance
# Returns the calculated geographical distance.

# df["distance"] = calculate_distance(...)
# Adds the calculated ride distance to the dataset.

# df = df[(df["distance"] > 0) & ...]
# Removes zero-distance and unrealistic rides.

# Q1
# Calculates the first quartile of fare amounts.

# Q3
# Calculates the third quartile of fare amounts.

# IQR = Q3 - Q1
# Calculates the interquartile range.

# lower_limit
# Calculates the lower boundary for identifying fare outliers.

# upper_limit
# Calculates the upper boundary for identifying fare outliers.

# outliers = df[...]
# Selects fare records outside the IQR boundaries.

# print(len(outliers))
# Displays the number of identified outliers.

# df = df[(...)]
# Removes the identified fare outliers.

# features = [...]
# Stores the numerical columns used for correlation analysis.

# df[features].corr()
# Calculates the correlation between the selected variables.

# sns.heatmap(...)
# Displays the correlation matrix as a heatmap.

# X = df[[...]]
# Selects pickup details, drop-off details, passenger count,
# time information and distance as input features.

# y = df["fare_amount"]
# Selects fare_amount as the target variable to predict.

# train_test_split(...)
# Divides data into 80% training and 20% testing data.

# models = {...}
# Creates Linear Regression, Ridge Regression and Lasso Regression models.

# for name, model in models.items():
# Iterates through all three regression models.

# make_pipeline(...)
# Creates a pipeline that fills missing values, scales features
# and then applies the selected regression model.

# pipeline.fit(...)
# Trains the regression model using training data.

# pipeline.predict(...)
# Predicts Uber fare prices for testing data.

# r2_score(...)
# Calculates R-squared, which measures how well the model
# explains variations in fare prices.

# np.sqrt(mean_squared_error(...))
# Calculates RMSE, the square root of the mean squared prediction error.

# mean_absolute_error(...)
# Calculates MAE, the average absolute prediction error.

# results.append(...)
# Stores the evaluation scores of each model.

# pd.DataFrame(results)
# Creates a comparison table containing all model scores.

# idxmax()
# Finds the model with the highest R2 score.

# sns.barplot(...)
# Creates a bar graph comparing the R2 scores of all models.

# plt.show()
# Displays the final comparison graph.