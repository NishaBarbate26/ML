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
# AIM:
# Predict Uber ride fare prices using:
# 1. Linear Regression.
# 2. Ridge Regression.
# 3. Lasso Regression.
#
# Dataset: uber.csv
#
# ============================================================
# 1. INSTALLATION AND EXECUTION COMMANDS
# ============================================================

# WINDOWS:
# python -m pip install pandas numpy matplotlib seaborn scikit-learn
# python practical_regression.py

# UBUNTU / LINUX:
# pip install pandas numpy matplotlib seaborn scikit-learn
# python3 -m pip install pandas numpy matplotlib seaborn scikit-learn
# python3 practical_regression.py

# DATASET:
# uber.csv

# Keep uber.csv in the same folder as the Python program.

# ============================================================
# 2. IMPORTING REQUIRED LIBRARIES
# ============================================================

# import pandas as pd

# Pandas is a Python library used for data manipulation
# and data analysis.

# It provides a DataFrame structure to store data
# in rows and columns.

# pd is the alias (short name) for pandas.

# Uses:
# 1. Loading CSV datasets.
# 2. Displaying dataset records.
# 3. Removing missing values.
# 4. Selecting columns.
# 5. Creating new columns.
# 6. Performing data analysis.


# import numpy as np

# NumPy stands for Numerical Python.

# It is used for mathematical and numerical operations.

# np is the alias for NumPy.

# Uses in this practical:
# 1. Converting degrees into radians.
# 2. Performing trigonometric calculations.
# 3. Calculating geographical distance.
# 4. Calculating square roots.
# 5. Performing numerical operations.


# import matplotlib.pyplot as plt

# Matplotlib is a Python library used for data visualization.

# pyplot is a module used to create graphs.

# plt is the alias for matplotlib.pyplot.

# Uses:
# 1. Creating graphs.
# 2. Setting graph titles.
# 3. Adding axis labels.
# 4. Displaying visualizations.


# import seaborn as sns

# Seaborn is a Python data visualization library
# built on top of Matplotlib.

# sns is the alias for seaborn.

# Uses:
# 1. Creating correlation heatmaps.
# 2. Creating statistical graphs.
# 3. Creating bar plots.
# 4. Visualizing relationships between variables.


# from sklearn.model_selection import train_test_split

# Imports the train_test_split() function.

# It divides the dataset into:
# 1. Training dataset.
# 2. Testing dataset.

# Training data is used to train the model.

# Testing data is used to evaluate the model
# on unseen observations.


# from sklearn.preprocessing import StandardScaler

# StandardScaler is used for feature scaling.

# It standardizes numerical features.

# After standardization:
# Mean = 0.
# Standard Deviation = 1.

# Formula:
# Z = (X - Mean) / Standard Deviation

# It brings features with different numerical ranges
# onto a common scale.


# from sklearn.impute import SimpleImputer

# SimpleImputer is used to handle missing values.

# Missing values are also called null values or NaN values.

# It replaces missing values with a selected statistical value.

# In this practical, the median strategy is used.

# Median is the middle value of sorted observations.

# It is less affected by extreme values than the mean.


# from sklearn.pipeline import make_pipeline

# make_pipeline() creates a machine learning pipeline.

# A pipeline combines multiple preprocessing steps
# and a machine learning model into one sequence.

# In this practical, the pipeline performs:
# 1. Missing value handling.
# 2. Feature scaling.
# 3. Regression model training.

# It ensures that the same preprocessing operations
# are applied during training and prediction.


# from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Imports three regression algorithms.

# 1. Linear Regression:
# Predicts a continuous numerical value using a linear relationship.

# 2. Ridge Regression:
# Linear regression with L2 regularization.
# It reduces the effect of large coefficients.

# 3. Lasso Regression:
# Linear regression with L1 regularization.
# It can reduce some feature coefficients to exactly zero.


# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Imports evaluation metrics used to measure
# the performance of regression models.

# r2_score():
# Calculates the R-squared score.

# mean_squared_error():
# Calculates the Mean Squared Error.

# mean_absolute_error():
# Calculates the Mean Absolute Error.


# ============================================================
# 3. LOADING THE UBER DATASET
# ============================================================

# df = pd.read_csv("uber.csv")

# pd.read_csv() reads a CSV file.

# uber.csv is the dataset filename.

# df is the DataFrame variable used to store the dataset.

# The Uber dataset contains information about Uber rides.

# IMPORTANT DATASET COLUMNS:

# 1. fare_amount:
#    Actual fare charged for the ride.
#    This is the target variable to be predicted.

# 2. pickup_datetime:
#    Date and time when the passenger was picked up.

# 3. pickup_longitude:
#    Longitude coordinate of the pickup location.

# 4. pickup_latitude:
#    Latitude coordinate of the pickup location.

# 5. dropoff_longitude:
#    Longitude coordinate of the drop-off location.

# 6. dropoff_latitude:
#    Latitude coordinate of the drop-off location.

# 7. passenger_count:
#    Number of passengers travelling in the Uber.

# 8. key:
#    Unique identifier associated with a ride.

# 9. Unnamed: 0:
#    An unnecessary index column that may be present
#    in the dataset.

# print("Original Dataset:")
# print(df.head())

# print() displays information on the screen.

# df.head() displays the first five records of the dataset.

# PURPOSE:
# To check whether the dataset has loaded successfully
# and understand its structure.


# print("\nDataset Information:")
# print(df.info())

# info() displays the structural information of the dataset.

# It provides:
# 1. Column names.
# 2. Number of non-null values.
# 3. Data types.
# 4. Memory usage.

# WHY USE info()?

# It helps identify missing values and incorrect data types.

# For example:
# pickup_datetime should represent date and time.
# fare_amount should be numerical.


# ============================================================
# 4. REMOVING UNNECESSARY COLUMNS
# ============================================================

# df = df.drop(columns=["Unnamed: 0", "key"], errors="ignore")

# drop() removes unnecessary columns from the DataFrame.

# columns specifies which columns should be removed.

# Unnamed: 0:
# Usually represents an unnecessary saved index column.

# key:
# Represents a ride identifier.
# It does not directly describe the ride fare.

# errors="ignore":
# Prevents an error if any specified column does not exist.

# WHY REMOVE THESE COLUMNS?

# These columns are not useful for predicting the fare.

# Removing unnecessary columns helps keep
# the dataset focused on relevant information.


# ============================================================
# 5. REMOVING MISSING VALUES IN IMPORTANT COLUMNS
# ============================================================

# df = df.dropna(subset=[
#     "fare_amount",
#     "pickup_datetime",
#     "pickup_longitude",
#     "pickup_latitude",
#     "dropoff_longitude",
#     "dropoff_latitude",
#     "passenger_count"
# ])

# dropna() removes rows containing missing values.

# subset specifies the columns in which missing values
# must be checked.

# The program removes records where any of these important
# columns contains a missing value.

# WHY REMOVE THESE MISSING VALUES?

# These columns are necessary for fare prediction.

# Without a fare amount, the actual target value is unknown.

# Without pickup and drop-off coordinates,
# the geographical distance cannot be calculated.

# Without pickup_datetime, time-based features cannot be extracted.

# Without passenger_count, the passenger information is incomplete.


# ============================================================
# 6. CONVERTING COLUMNS INTO NUMERIC VALUES
# ============================================================

# numeric_columns = [
#     "fare_amount",
#     "pickup_longitude",
#     "pickup_latitude",
#     "dropoff_longitude",
#     "dropoff_latitude",
#     "passenger_count"
# ]

# Creates a list containing the columns that should
# contain numerical values.

# These columns are required for mathematical calculations
# and regression model training.


# for column in numeric_columns:

# The for loop processes each column in numeric_columns
# one by one.

# This avoids writing the same conversion operation
# separately for every column.


# df[column] = pd.to_numeric(df[column], errors="coerce")

# pd.to_numeric() converts column values into numerical data.

# errors="coerce":
# Converts values that cannot be converted into numbers
# into NaN (Not a Number).

# Example:
# "10.5" becomes 10.5.
# "Invalid" becomes NaN.

# WHY CONVERT INTO NUMERIC VALUES?

# Machine learning regression algorithms require
# numerical input features.

# Text values cannot be directly used for these calculations.


# df = df.dropna(subset=numeric_columns)

# Removes rows containing NaN values in the selected
# numerical columns.

# This ensures that the important numerical values
# are available for further processing.


# ============================================================
# 7. REMOVING INVALID RECORDS
# ============================================================

# df = df[
#     (df["fare_amount"] > 0) &
#     (df["passenger_count"] > 0) &
#     (df["passenger_count"] <= 6) &
#     (df["pickup_longitude"].between(-180, 180)) &
#     (df["dropoff_longitude"].between(-180, 180)) &
#     (df["pickup_latitude"].between(-90, 90)) &
#     (df["dropoff_latitude"].between(-90, 90))
# ].copy()

# This step removes invalid or unrealistic records.

# & represents the logical AND operator.

# All the specified conditions must be satisfied
# for a record to remain in the dataset.


# CONDITION 1:

# fare_amount > 0

# Removes rides having zero or negative fare amounts.

# A valid ride fare should be positive.


# CONDITION 2:

# passenger_count > 0

# Removes rides having zero or negative passenger counts.

# A ride must have at least one passenger.


# CONDITION 3:

# passenger_count <= 6

# Keeps rides having a maximum of six passengers.

# This practical uses six as the selected upper limit
# for reasonable passenger counts.


# CONDITION 4:

# pickup_longitude between -180 and 180.

# Longitude represents the east-west geographical position.

# Valid longitude values range from -180 to +180 degrees.


# CONDITION 5:

# dropoff_longitude between -180 and 180.

# Checks whether the drop-off longitude is within
# the valid geographical range.


# CONDITION 6:

# pickup_latitude between -90 and 90.

# Latitude represents the north-south geographical position.

# Valid latitude values range from -90 to +90 degrees.


# CONDITION 7:

# dropoff_latitude between -90 and 90.

# Checks whether the drop-off latitude is within
# the valid geographical range.


# .copy()

# Creates an independent copy of the filtered DataFrame.

# It helps avoid chained assignment warnings
# when modifying the filtered dataset later.


# WHY REMOVE INVALID RECORDS?

# Incorrect values can affect distance calculations
# and reduce the reliability of the regression model.


# ============================================================
# 8. CONVERTING PICKUP DATETIME
# ============================================================

# df["pickup_datetime"] = pd.to_datetime(
#     df["pickup_datetime"],
#     errors="coerce",
#     utc=True
# )

# pd.to_datetime() converts the pickup_datetime column
# into a proper date and time format.

# errors="coerce":
# Converts invalid date and time values into NaT.

# NaT means Not a Time.

# utc=True:
# Converts or interprets the timestamps as UTC timestamps.

# WHY CONVERT DATETIME?

# Date and time information can be used to extract
# useful features such as hour, day, month and year.

# These features may help identify patterns in ride fares.


# df = df.dropna(subset=["pickup_datetime"])

# Removes records containing invalid datetime values.

# This ensures that valid date and time information
# is available for feature extraction.


# ============================================================
# 9. EXTRACTING DATE AND TIME FEATURES
# ============================================================

# df["hour"] = df["pickup_datetime"].dt.hour

# Extracts the hour from the pickup datetime.

# Possible values = 0 to 23.

# Example:
# 08:30 gives hour = 8.
# 18:45 gives hour = 18.

# PURPOSE:
# Ride fares may vary depending on the time of the day.


# df["day"] = df["pickup_datetime"].dt.day

# Extracts the day of the month.

# Possible values = 1 to 31.

# PURPOSE:
# Helps identify patterns in fares on different dates
# of the month.


# df["month"] = df["pickup_datetime"].dt.month

# Extracts the month from the pickup datetime.

# Possible values = 1 to 12.

# PURPOSE:
# Helps identify monthly or seasonal fare patterns.


# df["year"] = df["pickup_datetime"].dt.year

# Extracts the year from the pickup datetime.

# PURPOSE:
# Helps identify changes in fare patterns across years.


# df["day_of_week"] = df["pickup_datetime"].dt.dayofweek

# Extracts the day of the week.

# Possible values:
# Monday = 0.
# Tuesday = 1.
# Wednesday = 2.
# Thursday = 3.
# Friday = 4.
# Saturday = 5.
# Sunday = 6.

# PURPOSE:
# Helps identify differences between weekday
# and weekend ride patterns.


# WHY FEATURE EXTRACTION?

# Feature extraction converts existing information
# into additional useful input variables.

# Instead of using only the original datetime column,
# we extract separate numerical features.

# These features can help the model learn
# possible time-based fare patterns.


# ============================================================
# 10. CALCULATING DISTANCE USING HAVERSINE FORMULA
# ============================================================

# def calculate_distance(lat1, lon1, lat2, lon2):

# def defines a user-defined Python function.

# calculate_distance is the function name.

# Parameters:

# lat1 = Pickup latitude.
# lon1 = Pickup longitude.
# lat2 = Drop-off latitude.
# lon2 = Drop-off longitude.

# PURPOSE:

# Calculates the geographical distance between
# the pickup and drop-off locations.

# The function uses the Haversine formula.


# lat1 = np.radians(lat1)

# Converts pickup latitude from degrees into radians.

# np.radians() performs degree-to-radian conversion.


# lon1 = np.radians(lon1)

# Converts pickup longitude from degrees into radians.


# lat2 = np.radians(lat2)

# Converts drop-off latitude from degrees into radians.


# lon2 = np.radians(lon2)

# Converts drop-off longitude from degrees into radians.


# WHY CONVERT DEGREES INTO RADIANS?

# Trigonometric functions in NumPy use radians.

# Therefore, geographical coordinates must be converted
# into radians before applying the Haversine formula.


# dlat = lat2 - lat1

# Calculates the difference between pickup and drop-off latitudes.


# dlon = lon2 - lon1

# Calculates the difference between pickup and drop-off longitudes.


# a = (
#     np.sin(dlat / 2) ** 2 +
#     np.cos(lat1) * np.cos(lat2) *
#     np.sin(dlon / 2) ** 2
# )

# Calculates the intermediate value used in the Haversine formula.

# np.sin():
# Calculates the sine of an angle.

# np.cos():
# Calculates the cosine of an angle.

# ** 2:
# Squares the calculated value.

# Haversine formula is used to calculate the shortest
# distance between two points on the surface of a sphere.


# distance = 6371 * 2 * np.arcsin(np.sqrt(a))

# Calculates the geographical distance in kilometres.

# np.sqrt(a):
# Calculates the square root of a.

# np.arcsin():
# Calculates the inverse sine.

# 6371:
# Represents the approximate radius of the Earth in kilometres.

# Formula:

# Distance = 2 * R * arcsin(sqrt(a))

# Where:
# R = Earth's radius.
# a = Intermediate value calculated using latitude and longitude.

# OUTPUT:
# Distance between pickup and drop-off locations in kilometres.


# return distance

# return sends the calculated distance back
# to the location from which the function was called.


# ============================================================
# 11. ADDING DISTANCE TO THE DATASET
# ============================================================

# df["distance"] = calculate_distance(
#     df["pickup_latitude"],
#     df["pickup_longitude"],
#     df["dropoff_latitude"],
#     df["dropoff_longitude"]
# )

# Calls the calculate_distance() function.

# Passes the pickup and drop-off coordinates
# for every ride in the dataset.

# Stores the calculated distance in a new column called distance.

# The distance column represents the approximate
# geographical distance of each ride in kilometres.

# WHY CALCULATE DISTANCE?

# Distance is an important feature for fare prediction.

# Generally, longer rides may have higher fares.

# The distance feature provides useful information
# to the regression algorithms.


# ============================================================
# 12. REMOVING ZERO-DISTANCE AND UNREALISTIC RIDES
# ============================================================

# df = df[
#     (df["distance"] > 0) &
#     (df["distance"] < 200)
# ].copy()

# Removes rides having zero or negative distance.

# Also removes rides with distances of 200 kilometres
# or more.

# This practical uses 200 kilometres as the selected
# upper limit for filtering unrealistic rides.

# WHY REMOVE ZERO DISTANCE?

# Zero-distance rides may indicate incorrect coordinates
# or invalid ride records.

# WHY REMOVE VERY LARGE DISTANCES?

# Extremely large distances may represent incorrect
# geographical coordinates or unusual records.

# .copy() creates an independent copy of the filtered data.


# ============================================================
# 13. DISPLAYING THE PREPROCESSED DATASET
# ============================================================

# print("\nDataset After Preprocessing:")
# print(df.head())

# Displays the first five records after preprocessing.

# At this stage, the dataset contains:
# 1. Valid fare amounts.
# 2. Valid pickup and drop-off coordinates.
# 3. Valid passenger counts.
# 4. Proper datetime values.
# 5. Extracted time-based features.
# 6. Calculated geographical distances.

# This helps us verify that preprocessing was performed.


# ============================================================
# 14. IDENTIFYING OUTLIERS USING IQR METHOD
# ============================================================

# OUTLIER:

# An outlier is a data observation that is unusually
# different from most other observations.

# Example:
# If most ride fares are between 5 and 50,
# a fare of 500 may be an unusual observation.

# Outliers can negatively affect regression models,
# especially Linear Regression.

# IQR METHOD:

# IQR stands for Interquartile Range.

# It is a statistical method used to identify outliers.


# Q1 = df["fare_amount"].quantile(0.25)

# Calculates the first quartile.

# Q1 represents the 25th percentile.

# Approximately 25% of the observations have values
# less than or equal to Q1.


# Q3 = df["fare_amount"].quantile(0.75)

# Calculates the third quartile.

# Q3 represents the 75th percentile.

# Approximately 75% of the observations have values
# less than or equal to Q3.


# IQR = Q3 - Q1

# Calculates the Interquartile Range.

# FORMULA:

# IQR = Q3 - Q1

# IQR represents the spread of the middle 50%
# of the observations.


# lower_limit = Q1 - 1.5 * IQR

# Calculates the lower boundary for identifying outliers.

# Any fare below this value is considered a lower outlier.


# upper_limit = Q3 + 1.5 * IQR

# Calculates the upper boundary for identifying outliers.

# Any fare above this value is considered an upper outlier.


# outliers = df[
#     (df["fare_amount"] < lower_limit) |
#     (df["fare_amount"] > upper_limit)
# ]

# Selects all fare records outside the calculated boundaries.

# | represents the logical OR operator.

# A record is selected if:
# 1. Fare is below the lower limit.
# OR
# 2. Fare is above the upper limit.

# These selected records are stored in outliers.


# print("\nNumber of Fare Outliers:", len(outliers))

# len() returns the total number of records in outliers.

# Displays the number of identified fare outliers.


# print("\nLower Limit:", lower_limit)

# Displays the calculated lower outlier boundary.


# print("Upper Limit:", upper_limit)

# Displays the calculated upper outlier boundary.


# ============================================================
# 15. REMOVING FARE OUTLIERS
# ============================================================

# df = df[
#     (df["fare_amount"] >= lower_limit) &
#     (df["fare_amount"] <= upper_limit)
# ].copy()

# Keeps only the records whose fare amounts
# are within the calculated IQR boundaries.

# Records below the lower limit are removed.

# Records above the upper limit are removed.

# WHY REMOVE OUTLIERS?

# Extremely high or low fare values can influence
# the regression model and affect its predictions.

# Removing selected outliers helps reduce the effect
# of extreme fare observations.

# NOTE:
# IQR is a statistical filtering method.
# Not every outlier is necessarily an incorrect observation.


# ============================================================
# 16. CORRELATION ANALYSIS
# ============================================================

# features = [
#     "pickup_longitude",
#     "pickup_latitude",
#     "dropoff_longitude",
#     "dropoff_latitude",
#     "passenger_count",
#     "hour",
#     "day",
#     "month",
#     "year",
#     "day_of_week",
#     "distance",
#     "fare_amount"
# ]

# Creates a list of numerical features for correlation analysis.

# The list contains pickup information,
# drop-off information, passenger count,
# time features, distance and fare amount.

# Correlation analysis helps understand the relationship
# between numerical variables.


# correlation = df[features].corr()

# .corr() calculates the correlation matrix.

# Correlation measures the strength and direction
# of the linear relationship between two numerical variables.

# CORRELATION COEFFICIENT RANGE:

# +1 = Perfect positive correlation.

#  0 = No linear correlation.

# -1 = Perfect negative correlation.


# POSITIVE CORRELATION:

# When one variable increases, the other variable
# tends to increase.

# Example:
# Distance and fare may show positive correlation.


# NEGATIVE CORRELATION:

# When one variable increases, the other variable
# tends to decrease.


# ZERO OR LOW CORRELATION:

# Indicates a weak or no linear relationship
# between the two variables.

# IMPORTANT:
# Correlation does not prove causation.


# print("\nCorrelation Matrix:")
# print(correlation)

# Displays the calculated correlation matrix.

# Each cell contains the correlation coefficient
# between two selected numerical variables.


# ============================================================
# 17. CREATING THE CORRELATION HEATMAP
# ============================================================

# plt.figure(figsize=(10, 8))

# Creates a new figure for the heatmap.

# Width = 10 inches.
# Height = 8 inches.

# A larger figure makes the correlation values easier to read.


# sns.heatmap(
#     correlation,
#     annot=True,
#     cmap="coolwarm",
#     fmt=".2f"
# )

# sns.heatmap() creates a visual representation
# of the correlation matrix.

# correlation:
# Supplies the correlation matrix to the heatmap.

# annot=True:
# Displays the numerical correlation values
# inside the heatmap cells.

# cmap="coolwarm":
# Applies a colour scheme to represent positive
# and negative correlation values.

# fmt=".2f":
# Displays correlation values up to two decimal places.


# plt.title("Correlation Heatmap")

# Sets the title of the heatmap.


# plt.show()

# Displays the correlation heatmap.


# ============================================================
# 18. HEATMAP INTERPRETATION
# ============================================================

# GRAPH TYPE:
# Correlation Heatmap.

# PURPOSE:
# To visualize the relationships between numerical variables.

# EACH CELL:
# Represents the correlation coefficient between
# two variables.

# DIAGONAL VALUES:
# Every variable has a correlation of 1 with itself.

# POSITIVE VALUES:
# Indicate positive relationships.

# NEGATIVE VALUES:
# Indicate negative relationships.

# VALUES CLOSE TO ZERO:
# Indicate weak linear relationships.

# COLOURS:
# The coolwarm colour map uses different colours
# to distinguish positive and negative correlations.

# WHY IS THE HEATMAP USEFUL?

# 1. Helps identify relationships between variables.
# 2. Helps understand which features are associated with fare.
# 3. Helps identify highly correlated input features.
# 4. Makes numerical relationships easier to visualize.

# IMPORTANT:
# A strong correlation does not always mean
# that one variable causes another variable to change.


# ============================================================
# 19. SELECTING INPUT AND OUTPUT VARIABLES
# ============================================================

# X = df[
#     [
#         "pickup_longitude",
#         "pickup_latitude",
#         "dropoff_longitude",
#         "dropoff_latitude",
#         "passenger_count",
#         "hour",
#         "day",
#         "month",
#         "year",
#         "day_of_week",
#         "distance"
#     ]
# ]

# X represents the independent variables or input features.

# These features are provided to the regression models
# to predict the Uber fare amount.

# INPUT FEATURES:

# 1. pickup_longitude:
#    Longitude of the pickup location.

# 2. pickup_latitude:
#    Latitude of the pickup location.

# 3. dropoff_longitude:
#    Longitude of the drop-off location.

# 4. dropoff_latitude:
#    Latitude of the drop-off location.

# 5. passenger_count:
#    Number of passengers.

# 6. hour:
#    Pickup hour.

# 7. day:
#    Pickup day of the month.

# 8. month:
#    Pickup month.

# 9. year:
#    Pickup year.

# 10. day_of_week:
#     Day of the week.

# 11. distance:
#     Calculated geographical distance in kilometres.


# y = df["fare_amount"]

# y represents the dependent variable or target variable.

# It contains the actual Uber fare amount.

# The regression models learn the relationship between
# X and y and predict the fare amount.

# IMPORTANT:

# X = Independent variables.

# y = Dependent variable.

# This is a REGRESSION problem because the output
# is a continuous numerical value (fare amount).

# CLASSIFICATION predicts categories.

# REGRESSION predicts numerical values.


# ============================================================
# 20. SPLITTING DATA INTO TRAINING AND TESTING DATA
# ============================================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42
# )

# train_test_split() divides the dataset into
# training and testing datasets.

# X:
# Input features.

# y:
# Target fare values.

# test_size=0.2:
# Allocates 20% of the dataset for testing.

# Remaining 80% is used for training.

# random_state=42:
# Ensures that the same random split is generated
# every time the program runs.

# OUTPUT VARIABLES:

# X_train:
# Input features used for training.

# X_test:
# Input features used for testing.

# y_train:
# Actual fare values used for training.

# y_test:
# Actual fare values used for evaluation.

# WHY SPLIT DATA?

# To evaluate how well the regression models
# predict fare values for unseen observations.


# ============================================================
# 21. CREATING REGRESSION MODELS
# ============================================================

# models = {

#     "Linear Regression": LinearRegression(),

#     "Ridge Regression": Ridge(alpha=1.0),

#     "Lasso Regression": Lasso(alpha=0.01, max_iter=10000)

# }

# Creates a dictionary containing three regression models.

# A dictionary stores data in key-value pairs.

# Model name = Key.

# Regression algorithm = Value.

# All three algorithms are used to predict
# the same target variable: fare_amount.


# ---------------- LINEAR REGRESSION ----------------

# LinearRegression()

# Creates a Linear Regression model.

# Linear Regression is a supervised machine learning
# algorithm used to predict continuous numerical values.

# It establishes a linear relationship between
# independent variables and a dependent variable.

# GENERAL EQUATION:

# Y = b0 + b1X1 + b2X2 + ... + bnXn

# Where:

# Y = Predicted output (fare amount).

# b0 = Intercept.

# b1, b2, ..., bn = Regression coefficients.

# X1, X2, ..., Xn = Input features.

# OBJECTIVE:

# Finds coefficients that minimize the sum
# of squared prediction errors.

# PURPOSE IN THIS PRACTICAL:

# Predicts Uber fare prices using ride distance,
# location, passenger count and time-based features.


# ---------------- RIDGE REGRESSION ----------------

# Ridge(alpha=1.0)

# Creates a Ridge Regression model.

# Ridge Regression is Linear Regression with L2 regularization.

# Regularization is a technique used to control
# the size of model coefficients.

# It helps reduce overfitting and improves model stability.

# L2 REGULARIZATION:

# Adds a penalty based on the squared values
# of the regression coefficients.

# GENERAL OBJECTIVE:

# Sum of Squared Errors + alpha * Sum of Squared Coefficients

# alpha=1.0:

# Specifies the regularization strength.

# Higher alpha:
# Applies stronger regularization.

# Lower alpha:
# Applies weaker regularization.

# Ridge generally reduces coefficient values
# but does not normally make them exactly zero.

# PURPOSE IN THIS PRACTICAL:

# Predicts Uber fares while controlling large
# regression coefficients.


# ---------------- LASSO REGRESSION ----------------

# Lasso(alpha=0.01, max_iter=10000)

# Creates a Lasso Regression model.

# Lasso stands for Least Absolute Shrinkage
# and Selection Operator.

# It is Linear Regression with L1 regularization.

# L1 REGULARIZATION:

# Adds a penalty based on the absolute values
# of the regression coefficients.

# GENERAL OBJECTIVE:

# Sum of Squared Errors + alpha * Sum of Absolute Coefficients

# alpha=0.01:

# Specifies the regularization strength.

# Higher alpha:
# Applies stronger regularization.

# Lower alpha:
# Applies weaker regularization.

# IMPORTANT FEATURE:

# Lasso can reduce some regression coefficients
# exactly to zero.

# Therefore, it can perform automatic feature selection.

# max_iter=10000:

# Specifies the maximum number of iterations
# allowed for the optimization algorithm.

# It provides more iterations to help the algorithm converge.

# PURPOSE IN THIS PRACTICAL:

# Predicts Uber fares while controlling model complexity
# and potentially removing less useful features.


# ============================================================
# 22. CREATING A RESULTS LIST
# ============================================================

# results = []

# Creates an empty Python list.

# The list stores the evaluation results
# of all three regression models.

# Each model's results contain:
# 1. Model name.
# 2. R2 Score.
# 3. RMSE.
# 4. MAE.


# ============================================================
# 23. TRAINING AND EVALUATING ALL MODELS
# ============================================================

# for name, model in models.items():

# A for loop processes all three regression models.

# models.items() returns the key-value pairs
# from the models dictionary.

# name:
# Stores the name of the current regression model.

# model:
# Stores the corresponding regression algorithm.

# The loop executes the training and evaluation process
# separately for Linear, Ridge and Lasso Regression.


# pipeline = make_pipeline(
#     SimpleImputer(strategy="median"),
#     StandardScaler(),
#     model
# )

# make_pipeline() creates a machine learning pipeline.

# The pipeline contains three steps.


# STEP 1: SimpleImputer(strategy="median")

# Handles missing values in input features.

# strategy="median":
# Replaces missing numerical values with
# the median of the corresponding feature.

# Example:
# If the values are 10, 20, 30, 40, 50,
# the median is 30.

# If a value is missing, it is replaced with 30.

# WHY USE MEDIAN?

# Median is less affected by extreme values
# than the mean.

# This is useful when numerical features contain outliers.


# STEP 2: StandardScaler()

# Standardizes the input features.

# It transforms features so that their mean
# is approximately zero and standard deviation is one.

# WHY USE SCALING?

# Features have different numerical ranges.

# Scaling makes their values comparable.

# It is especially important for Ridge and Lasso
# because regularization is affected by feature scales.

# It also keeps preprocessing consistent across models.


# STEP 3: model

# Applies the selected regression algorithm.

# During the first loop:
# Linear Regression is applied.

# During the second loop:
# Ridge Regression is applied.

# During the third loop:
# Lasso Regression is applied.


# pipeline.fit(X_train, y_train)

# fit() trains the complete pipeline.

# First, missing values are handled.

# Next, the input features are standardized.

# Finally, the selected regression algorithm
# learns the relationship between input features
# and actual fare amounts.

# The model learns from training data only.


# predictions = pipeline.predict(X_test)

# predict() uses the trained pipeline to predict
# fare amounts for testing observations.

# The testing input features pass through:
# 1. Missing value handling.
# 2. Feature scaling.
# 3. Trained regression model.

# predictions stores the predicted fare values.

# These predictions are compared with y_test
# to evaluate the model.


# ============================================================
# 24. CALCULATING R2 SCORE
# ============================================================

# r2 = r2_score(y_test, predictions)

# r2_score() calculates the coefficient of determination.

# R2 measures how well the model explains the variation
# in actual fare values.

# FORMULA:

# R2 = 1 - (Sum of Squared Residuals /
#           Total Sum of Squares)

# WHERE:

# Residual = Actual Value - Predicted Value.

# GENERAL INTERPRETATION:

# R2 = 1:
# Perfect predictions.

# R2 close to 1:
# The model explains a large proportion
# of the variation in fare values.

# R2 = 0:
# The model performs approximately like
# predicting the mean fare for every observation.

# R2 below 0:
# The model performs worse than the mean prediction baseline.

# IMPORTANT:
# A higher R2 score is generally preferred.

# R2 is not necessarily a percentage accuracy.


# ============================================================
# 25. CALCULATING RMSE
# ============================================================

# rmse = np.sqrt(mean_squared_error(y_test, predictions))

# RMSE stands for Root Mean Squared Error.

# It measures the average prediction error
# while giving greater importance to larger errors.

# STEP 1:

# mean_squared_error() calculates the Mean Squared Error.

# MSE = Average of Squared Differences
#       Between Actual and Predicted Values.

# STEP 2:

# np.sqrt() calculates the square root of MSE.

# FORMULA:

# RMSE = sqrt((1/n) * Sum((Actual - Predicted)^2))

# WHERE:

# n = Total number of testing observations.

# Actual = Actual fare value.

# Predicted = Predicted fare value.

# INTERPRETATION:

# Lower RMSE indicates better prediction performance.

# RMSE is expressed in the same unit as the target variable.

# If fare_amount is measured in dollars,
# RMSE is also measured in dollars.

# RMSE gives more penalty to large prediction errors
# because the errors are squared.


# ============================================================
# 26. CALCULATING MAE
# ============================================================

# mae = mean_absolute_error(y_test, predictions)

# MAE stands for Mean Absolute Error.

# It calculates the average absolute difference
# between actual and predicted fare values.

# FORMULA:

# MAE = (1/n) * Sum(|Actual - Predicted|)

# WHERE:

# n = Total number of testing observations.

# | | = Absolute value.

# INTERPRETATION:

# Lower MAE indicates better prediction performance.

# MAE is easy to understand because it represents
# the average magnitude of prediction errors.

# Example:

# Actual fare = 20.

# Predicted fare = 18.

# Absolute error = |20 - 18| = 2.

# If MAE = 2, the average absolute prediction error
# is 2 fare units.


# ============================================================
# 27. STORING MODEL RESULTS
# ============================================================

# results.append({

#     "Model": name,

#     "R2 Score": r2,

#     "RMSE": rmse,

#     "MAE": mae

# })

# append() adds a new item to the results list.

# Each model's name and evaluation metrics
# are stored as a dictionary.

# The results are stored for later comparison.

# The same process is repeated for all three models.


# print("\n", name)

# Displays the name of the current regression model.


# print("R2 Score:", round(r2, 4))

# Displays the R2 score rounded to four decimal places.

# round() controls the number of decimal places
# displayed in the output.


# print("RMSE:", round(rmse, 4))

# Displays the RMSE rounded to four decimal places.


# print("MAE:", round(mae, 4))

# Displays the MAE rounded to four decimal places.


# ============================================================
# 28. COMPARING ALL REGRESSION MODELS
# ============================================================

# results_df = pd.DataFrame(results)

# Converts the results list into a pandas DataFrame.

# The comparison DataFrame contains four columns:

# Model:
# Name of the regression algorithm.

# R2 Score:
# Measures how well the model explains variation in fare values.

# RMSE:
# Measures the square root of the average squared prediction error.

# MAE:
# Measures the average absolute prediction error.

# Each row represents one regression model.


# print("\nRegression Model Comparison:")

# Displays the heading for the comparison results.


# print(results_df)

# Displays the comparison table containing
# Linear, Ridge and Lasso Regression performance.

# PURPOSE:

# To compare the performance of all three algorithms
# using the same testing dataset.


# ============================================================
# 29. IDENTIFYING THE BEST PERFORMING MODEL
# ============================================================

# best_model = results_df.loc[
#     results_df["R2 Score"].idxmax()
# ]

# idxmax() returns the index of the maximum value
# in the R2 Score column.

# loc[] selects the complete row corresponding
# to that index.

# The selected row contains:
# 1. Model name.
# 2. R2 Score.
# 3. RMSE.
# 4. MAE.

# The model with the highest R2 score
# is selected as the best performing model
# according to the R2 criterion.

# IMPORTANT:

# A higher R2 score is generally better.

# However, RMSE and MAE should also be checked.

# For RMSE:
# Lower is better.

# For MAE:
# Lower is better.

# A good model should ideally have:
# Higher R2.
# Lower RMSE.
# Lower MAE.

# Selecting the highest R2 model alone does not
# guarantee that it also has the lowest RMSE and MAE.


# print("\nBest Performing Model:")

# Displays the heading for the best model.


# print(best_model)

# Displays the selected model name
# and its evaluation scores.


# ============================================================
# 30. VISUALIZATION OF MODEL COMPARISON
# ============================================================

# plt.figure(figsize=(8, 5))

# Creates a new figure for model comparison.

# Width = 8 inches.
# Height = 5 inches.


# sns.barplot(
#     data=results_df,
#     x="Model",
#     y="R2 Score"
# )

# sns.barplot() creates a bar graph.

# data=results_df:
# Supplies the model comparison DataFrame.

# x="Model":
# Displays the regression model names on the horizontal axis.

# y="R2 Score":
# Displays the R2 scores on the vertical axis.

# Each bar represents the R2 score
# of one regression algorithm.

# THREE BARS:

# 1. Linear Regression.
# 2. Ridge Regression.
# 3. Lasso Regression.

# PURPOSE:

# To visually compare the R2 scores
# of the three regression models.


# plt.title("Comparison of Regression Models")

# Sets the title of the bar graph.


# plt.ylabel("R2 Score")

# Sets the vertical axis label.


# plt.show()

# Displays the final comparison graph.


# ============================================================
# 31. DETAILED BAR GRAPH INTERPRETATION
# ============================================================

# GRAPH TYPE:
# Bar Graph.

# GRAPH TITLE:
# Comparison of Regression Models.

# X-AXIS:
# Displays the names of the three regression models.

# Y-AXIS:
# Displays the corresponding R2 scores.

# EACH BAR:
# Represents the R2 score of one regression algorithm.

# TALLER BAR:
# Represents a higher R2 score.

# SHORTER BAR:
# Represents a lower R2 score.

# INTERPRETATION:

# 1. Compare the heights of all three bars.

# 2. The tallest bar represents the highest R2 score.

# 3. The model with the highest R2 score is selected
#    as the best model according to this comparison.

# 4. The graph helps us compare the explanatory
#    performance of the three regression algorithms.

# IMPORTANT:
# A higher R2 score does not always mean
# the model has the lowest prediction error.

# Therefore, RMSE and MAE should also be considered.


# ============================================================
# 32. IMPORTANT REGRESSION CONCEPTS
# ============================================================

# WHAT IS MACHINE LEARNING?

# Machine Learning is a branch of Artificial Intelligence
# that allows computers to learn patterns from data
# and make predictions without being explicitly programmed
# for every individual case.


# WHAT IS REGRESSION?

# Regression is a supervised machine learning technique
# used to predict continuous numerical values.

# Examples:
# 1. Predicting house prices.
# 2. Predicting Uber ride fares.
# 3. Predicting temperature.
# 4. Predicting salary.


# WHAT IS LINEAR REGRESSION?

# Linear Regression is a supervised learning algorithm
# that establishes a linear relationship between
# independent and dependent variables.

# It predicts numerical values.


# WHAT IS REGULARIZATION?

# Regularization is a technique used to reduce
# model complexity by applying a penalty to large coefficients.

# It helps control overfitting.


# DIFFERENCE BETWEEN LINEAR, RIDGE AND LASSO:

# LINEAR REGRESSION:
# Does not apply an L1 or L2 regularization penalty.

# RIDGE REGRESSION:
# Uses L2 regularization.
# Shrinks coefficients towards zero.
# Usually does not make coefficients exactly zero.

# LASSO REGRESSION:
# Uses L1 regularization.
# Can make some coefficients exactly zero.
# Can perform feature selection.


# WHAT IS OVERFITTING?

# Overfitting occurs when a model learns the training data
# too closely, including noise and unnecessary patterns.

# Such a model may perform poorly on unseen testing data.


# WHAT IS UNDERFITTING?

# Underfitting occurs when a model is too simple
# to learn the important patterns in the dataset.

# It may perform poorly on both training and testing data.


# WHAT IS A FEATURE?

# A feature is an input variable used by a machine learning model.

# Examples in this practical:
# Distance.
# Passenger count.
# Pickup coordinates.
# Drop-off coordinates.
# Hour.
# Day.
# Month.


# WHAT IS A TARGET VARIABLE?

# A target variable is the output that a machine learning
# model tries to predict.

# In this practical:
# Target variable = fare_amount.


# WHAT IS FEATURE SCALING?

# Feature scaling transforms numerical features
# to a common scale.

# StandardScaler uses:
# Mean = 0.
# Standard Deviation = 1.


# WHAT IS DATA PREPROCESSING?

# Data preprocessing is the process of preparing raw data
# before applying a machine learning algorithm.

# It includes:
# 1. Removing unnecessary columns.
# 2. Handling missing values.
# 3. Converting data types.
# 4. Removing invalid records.
# 5. Extracting useful features.
# 6. Calculating distance.
# 7. Identifying and handling outliers.
# 8. Scaling input features.


# WHAT IS THE HAVERSINE FORMULA?

# The Haversine formula calculates the great-circle distance
# between two geographical points on a spherical Earth.

# It uses:
# Latitude and longitude coordinates.

# In this practical, it calculates the approximate
# distance between Uber pickup and drop-off locations.


# WHAT IS IQR?

# IQR stands for Interquartile Range.

# It measures the spread of the middle 50% of the data.

# Formula:
# IQR = Q3 - Q1.

# It is used to identify possible outliers.


# WHAT IS A PIPELINE?

# A pipeline combines preprocessing steps
# and a machine learning model into a single workflow.

# In this practical:

# Step 1: SimpleImputer.
# Step 2: StandardScaler.
# Step 3: Regression algorithm.

# It helps maintain a consistent processing sequence
# during model training and prediction.


# ============================================================
# 33. IMPORTANT VIVA QUESTIONS AND ANSWERS
# ============================================================

# Q1. What is the aim of this practical?

# Ans:
# To predict Uber ride fare prices using Linear Regression,
# Ridge Regression and Lasso Regression and compare
# their performance.


# Q2. Which dataset is used?

# Ans:
# Uber fare dataset (uber.csv).


# Q3. What type of machine learning problem is this?

# Ans:
# Supervised regression problem.


# Q4. Why is it called a regression problem?

# Ans:
# Because the output is a continuous numerical value,
# which is the Uber fare amount.


# Q5. What is the target variable?

# Ans:
# fare_amount.


# Q6. What are independent variables?

# Ans:
# Pickup coordinates, drop-off coordinates,
# passenger count, time features and calculated distance.


# Q7. What is the difference between regression and classification?

# Ans:
# Regression predicts continuous numerical values.

# Classification predicts discrete categories or classes.


# Q8. What is Linear Regression?

# Ans:
# Linear Regression is a supervised machine learning
# algorithm used to predict continuous values
# by establishing a linear relationship between variables.


# Q9. What is Ridge Regression?

# Ans:
# Ridge Regression is Linear Regression with L2 regularization.
# It reduces the effect of large coefficients.


# Q10. What is Lasso Regression?

# Ans:
# Lasso Regression is Linear Regression with L1 regularization.
# It can reduce some coefficients exactly to zero.


# Q11. What is the difference between Ridge and Lasso?

# Ans:
# Ridge uses L2 regularization and shrinks coefficients
# towards zero.

# Lasso uses L1 regularization and can make some
# coefficients exactly zero.


# Q12. What is alpha in Ridge and Lasso?

# Ans:
# Alpha controls the strength of regularization.

# Higher alpha means stronger regularization.

# Lower alpha means weaker regularization.


# Q13. Why is StandardScaler used?

# Ans:
# It standardizes features to a common scale
# with mean approximately zero and standard deviation one.


# Q14. Why is SimpleImputer used?

# Ans:
# It replaces missing values with a selected statistical value.

# In this practical, the median strategy is used.


# Q15. What is the median?

# Ans:
# Median is the middle value of a dataset
# after arranging the values in ascending order.


# Q16. What is the Haversine formula?

# Ans:
# It calculates the geographical distance between
# two points on a spherical Earth using latitude
# and longitude coordinates.


# Q17. Why is distance calculated?

# Ans:
# Distance is an important feature for fare prediction
# because ride distance may influence the fare amount.


# Q18. Why convert degrees into radians?

# Ans:
# Because NumPy trigonometric functions use radians
# instead of degrees.


# Q19. What is IQR?

# Ans:
# IQR stands for Interquartile Range.
# It is the difference between the third quartile
# and the first quartile.


# Q20. Why are outliers removed?

# Ans:
# Extreme values can affect regression model training
# and increase prediction errors.


# Q21. What is correlation?

# Ans:
# Correlation measures the strength and direction
# of the relationship between two numerical variables.


# Q22. What is a correlation heatmap?

# Ans:
# It is a graphical representation of a correlation matrix
# using colours to show positive and negative relationships.


# Q23. What is R2 score?

# Ans:
# R2 score is the coefficient of determination.
# It measures how well the model explains the variation
# in the target variable.


# Q24. What is RMSE?

# Ans:
# RMSE stands for Root Mean Squared Error.
# It is the square root of the average squared
# prediction errors.


# Q25. What is MAE?

# Ans:
# MAE stands for Mean Absolute Error.
# It measures the average absolute difference
# between actual and predicted values.


# Q26. Which is better: Higher or lower R2?

# Ans:
# A higher R2 score is generally better.


# Q27. Which is better: Higher or lower RMSE?

# Ans:
# Lower RMSE is better.


# Q28. Which is better: Higher or lower MAE?

# Ans:
# Lower MAE is better.


# Q29. Why is the dataset divided into training and testing data?

# Ans:
# Training data is used to train the model.
# Testing data is used to evaluate its performance
# on unseen observations.


# Q30. What is random_state=42?

# Ans:
# It fixes the random seed to produce a reproducible
# training and testing split.


# Q31. What is test_size=0.2?

# Ans:
# It means 20% of the dataset is used for testing
# and 80% is used for training.


# Q32. What is the purpose of make_pipeline()?

# Ans:
# It combines missing value handling, feature scaling
# and the regression model into one workflow.


# Q33. What is the purpose of max_iter in Lasso?

# Ans:
# It specifies the maximum number of iterations
# allowed for the optimization algorithm to converge.


# Q34. Can Lasso perform feature selection?

# Ans:
# Yes. Lasso can reduce some feature coefficients
# exactly to zero, effectively removing those features
# from the model.


# Q35. What is the best regression model in this practical?

# Ans:
# The program selects the model with the highest R2 score.

# RMSE and MAE should also be checked before
# making a final performance comparison.


# ============================================================
# 34. FINAL CONCLUSION
# ============================================================

# The Uber fare prediction practical was successfully performed
# using Linear Regression, Ridge Regression and Lasso Regression.

# First, the Uber dataset was loaded and inspected.

# Unnecessary columns were removed.

# Missing values, invalid records and incorrect numerical values
# were handled during preprocessing.

# Pickup datetime was converted into a proper date and time format.

# Hour, day, month, year and day of the week
# were extracted as additional features.

# The Haversine formula was used to calculate
# the geographical distance between pickup and drop-off locations.

# The IQR method was applied to identify fare outliers.

# A correlation matrix and heatmap were created
# to understand relationships between numerical variables.

# The dataset was divided into training and testing data.

# Three regression algorithms were trained using
# a machine learning pipeline.

# R2 score, RMSE and MAE were calculated
# to evaluate the performance of each model.

# A comparison table and bar graph were created
# to compare the three regression algorithms.

# Finally, the model with the highest R2 score
# was selected as the best performing model
# according to the R2 evaluation criterion.

# RESULT:

# Uber ride fare prediction was successfully implemented
# using Linear Regression, Ridge Regression and Lasso Regression.

# The performance of all three regression models
# was evaluated and compared using R2 score, RMSE and MAE.

# ============================================================
# END OF PRACTICAL
# ============================================================
