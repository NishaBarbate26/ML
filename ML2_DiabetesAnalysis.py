# AIM: Perform Univariate, Bivariate and Multiple Regression Analysis
# Using UCI Diabetes and Pima Indians Diabetes Datasets.
#
# Windows:
# python -m pip install pandas numpy matplotlib seaborn scikit-learn scipy
# python practical_regression_B.py
#
# Ubuntu/Linux:
# pip install pandas numpy matplotlib seaborn scikit-learn scipy
# python3 -m pip install pandas numpy matplotlib seaborn scikit-learn scipy
# python3 practical_regression_B.py


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew, kurtosis, mode

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    classification_report,
    confusion_matrix
)


# Load both datasets

uci = pd.read_csv("diabetes.csv")
pima = pd.read_csv("diabetes_pima.csv")

datasets = {
    "UCI Diabetes Dataset": uci,
    "Pima Indians Diabetes Dataset": pima
}


# Function for univariate analysis

def univariate_analysis(df, name):

    print("\n===================================")
    print("UNIVARIATE ANALYSIS:", name)
    print("===================================")

    numeric_data = df.select_dtypes(include=np.number)

    print("\nFrequency:")
    for column in numeric_data.columns:
        print("\n", column)
        print(numeric_data[column].value_counts().head(10))

    print("\nMean:")
    print(numeric_data.mean())

    print("\nMedian:")
    print(numeric_data.median())

    print("\nMode:")
    print(numeric_data.mode().iloc[0])

    print("\nVariance:")
    print(numeric_data.var())

    print("\nStandard Deviation:")
    print(numeric_data.std())

    print("\nSkewness:")
    print(numeric_data.apply(skew, nan_policy="omit"))

    print("\nKurtosis:")
    print(numeric_data.apply(kurtosis, nan_policy="omit"))

    numeric_data.hist(figsize=(12, 8))

    plt.suptitle("Univariate Analysis - " + name)

    plt.tight_layout()

    plt.show()


# Function for bivariate and multiple regression analysis

def regression_analysis(df, name):

    print("\n===================================")
    print("REGRESSION ANALYSIS:", name)
    print("===================================")

    target = "Outcome"

    if target not in df.columns:
        print("Outcome column not found in", name)
        return None

    df = df.copy()

    df = df.select_dtypes(include=np.number)

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=[target])

    df = df[df[target].isin([0, 1])]

    X = df.drop(columns=[target])
    y = df[target]

    if X.shape[1] == 0:
        print("No numeric input features available.")
        return None

    # Replace missing values

    imputer = SimpleImputer(strategy="median")

    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    # Bivariate analysis

    print("\nBivariate Correlation with Outcome:")

    print(X_imputed.corrwith(y))

    # Linear regression using the first feature

    first_feature = X.columns[0]

    X_single = X_imputed[[first_feature]]

    X_train, X_test, y_train, y_test = train_test_split(
        X_single,
        y,
        test_size=0.2,
        random_state=42
    )

    linear_model = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )

    linear_model.fit(X_train, y_train)

    linear_prediction = linear_model.predict(X_test)

    print("\nLinear Regression using:", first_feature)

    print("R2 Score:", r2_score(y_test, linear_prediction))

    print(
        "RMSE:",
        np.sqrt(mean_squared_error(y_test, linear_prediction))
    )

    # Logistic regression

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logistic_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    logistic_model.fit(X_train, y_train)

    logistic_prediction = logistic_model.predict(X_test)

    accuracy = accuracy_score(y_test, logistic_prediction)

    print("\nLogistic Regression Results:")

    print("Accuracy:", accuracy * 100, "%")

    print("\nConfusion Matrix:")

    print(confusion_matrix(y_test, logistic_prediction))

    print("\nClassification Report:")

    print(classification_report(y_test, logistic_prediction))

    # Multiple regression analysis

    multiple_model = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )

    multiple_model.fit(X_train, y_train)

    multiple_prediction = multiple_model.predict(X_test)

    multiple_r2 = r2_score(y_test, multiple_prediction)

    multiple_rmse = np.sqrt(
        mean_squared_error(y_test, multiple_prediction)
    )

    multiple_mae = mean_absolute_error(
        y_test,
        multiple_prediction
    )

    print("\nMultiple Regression Results:")

    print("R2 Score:", multiple_r2)

    print("RMSE:", multiple_rmse)

    print("MAE:", multiple_mae)

    return {
        "Dataset": name,
        "Linear Regression R2": r2_score(y_test, multiple_prediction),
        "Logistic Regression Accuracy": accuracy,
        "Multiple Regression R2": multiple_r2,
        "Multiple Regression RMSE": multiple_rmse,
        "Multiple Regression MAE": multiple_mae
    }


# Perform analysis on both datasets

results = []

for name, df in datasets.items():

    print("\n\nDATASET:", name)

    print("\nFirst Five Records:")

    print(df.head())

    univariate_analysis(df, name)

    result = regression_analysis(df, name)

    if result is not None:
        results.append(result)


# Compare both datasets

print("\n===================================")
print("COMPARISON OF BOTH DATASETS")
print("===================================")

comparison = pd.DataFrame(results)

print(comparison.to_string(index=False))

# Plot comparison of logistic regression accuracy

if not comparison.empty:

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=comparison,
        x="Dataset",
        y="Logistic Regression Accuracy"
    )

    plt.title("Diabetes Dataset Accuracy Comparison")

    plt.ylabel("Accuracy")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.show()


# ============================================================
# EXPLANATION LINE BY LINE
# ============================================================

# import pandas as pd
# Imports Pandas to read and process both datasets.

# import numpy as np
# Imports NumPy for mathematical calculations.

# import matplotlib.pyplot as plt
# Imports Matplotlib to display graphs.

# import seaborn as sns
# Imports Seaborn to create comparison graphs.

# from scipy.stats import skew, kurtosis, mode
# Imports statistical functions for skewness and kurtosis.

# train_test_split
# Divides data into training and testing sets.

# LinearRegression
# Imports the Linear Regression algorithm.

# LogisticRegression
# Imports the Logistic Regression classification algorithm.

# SimpleImputer
# Replaces missing numeric values.

# StandardScaler
# Standardizes numeric features.

# make_pipeline
# Combines preprocessing and machine learning models.

# accuracy_score
# Calculates classification accuracy.

# r2_score
# Calculates the R-squared regression score.

# mean_squared_error
# Calculates the mean squared error.

# mean_absolute_error
# Calculates the mean absolute error.

# classification_report
# Displays precision, recall and F1-score.

# confusion_matrix
# Displays actual and predicted classification results.

# pd.read_csv("diabetes.csv")
# Loads the UCI Diabetes dataset from the local CSV file.

# pd.read_csv("diabetes_pima.csv")
# Loads the Pima Indians Diabetes dataset.

# datasets = {...}
# Stores both datasets in a dictionary for repeated analysis.

# def univariate_analysis(df, name):
# Defines a function to perform univariate analysis.

# df.select_dtypes(include=np.number)
# Selects only numeric columns.

# value_counts()
# Displays the frequency of values in each column.

# numeric_data.mean()
# Calculates the mean of each numeric column.

# numeric_data.median()
# Calculates the median of each numeric column.

# numeric_data.mode()
# Calculates the mode of each numeric column.

# numeric_data.var()
# Calculates the variance of each numeric column.

# numeric_data.std()
# Calculates the standard deviation of each numeric column.

# numeric_data.apply(skew, nan_policy="omit")
# Calculates skewness for each numeric column.

# numeric_data.apply(kurtosis, nan_policy="omit")
# Calculates kurtosis for each numeric column.

# numeric_data.hist()
# Displays histograms for the numeric columns.

# def regression_analysis(df, name):
# Defines a function for bivariate and multiple regression analysis.

# target = "Outcome"
# Selects Outcome as the diabetes target column.

# df.select_dtypes(include=np.number)
# Selects numeric columns for regression analysis.

# df.replace(...)
# Converts infinite values into missing values.

# df.dropna(subset=[target])
# Removes records where the target value is missing.

# df[df[target].isin([0, 1])]
# Keeps records with binary diabetes labels.

# X = df.drop(columns=[target])
# Selects all numeric input features.

# y = df[target]
# Stores the diabetes outcome.

# SimpleImputer(strategy="median")
# Creates an object to replace missing values with column medians.

# imputer.fit_transform(X)
# Learns median values and fills missing feature values.

# X_imputed
# Stores the cleaned input features.

# X_imputed.corrwith(y)
# Calculates correlation between individual features and the outcome.

# first_feature = X.columns[0]
# Selects the first input feature for simple linear regression.

# train_test_split(...)
# Splits the selected feature and target into training and testing sets.

# make_pipeline(StandardScaler(), LinearRegression())
# Creates a pipeline to scale data and apply linear regression.

# linear_model.fit(...)
# Trains the linear regression model.

# linear_model.predict(...)
# Predicts numerical values for the test data.

# r2_score(...)
# Measures how well the linear model explains variations in the target.

# mean_squared_error(...)
# Calculates the mean squared prediction error.

# LogisticRegression(max_iter=1000)
# Creates a logistic regression classification model.

# stratify=y
# Maintains the same proportion of diabetes classes in training
# and testing datasets.

# logistic_model.fit(...)
# Trains the logistic regression model.

# logistic_model.predict(...)
# Predicts whether a patient has diabetes or not.

# accuracy_score(...)
# Calculates the percentage of correct predictions.

# confusion_matrix(...)
# Compares actual and predicted diabetes classes.

# classification_report(...)
# Displays precision, recall and F1-score.

# make_pipeline(StandardScaler(), LinearRegression())
# Creates the multiple linear regression model.

# multiple_model.fit(...)
# Trains the model using all available input features.

# multiple_model.predict(...)
# Predicts the target values using multiple features.

# multiple_r2
# Stores the R-squared score.

# multiple_rmse
# Stores the root mean squared error.

# multiple_mae
# Stores the mean absolute error.

# return {...}
# Returns the evaluation scores for the current dataset.

# for name, df in datasets.items():
# Iterates through both diabetes datasets.

# univariate_analysis(df, name)
# Performs univariate statistical analysis.

# regression_analysis(df, name)
# Performs regression and classification analysis.

# pd.DataFrame(results)
# Creates a comparison table for both datasets.

# sns.barplot(...)
# Creates a bar graph comparing logistic regression accuracy.

# plt.show()
# Displays the final comparison graph.