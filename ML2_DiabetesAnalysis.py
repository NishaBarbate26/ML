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
# AIM:
# Perform Univariate, Bivariate and Multiple Regression Analysis
# Using UCI Diabetes and Pima Indians Diabetes Datasets.
# ============================================================


# ============================================================
# 1. INSTALLATION AND EXECUTION COMMANDS
# ============================================================

# Windows:
# python -m pip install pandas numpy matplotlib seaborn scipy scikit-learn
# python practical_diabetes.py

# Ubuntu/Linux:
# pip install pandas numpy matplotlib seaborn scipy scikit-learn
# python3 -m pip install pandas numpy matplotlib seaborn scipy scikit-learn
# python3 practical_diabetes.py


# ============================================================
# 2. IMPORTING LIBRARIES
# ============================================================

# import pandas as pd
# Imports Pandas for loading, storing, and processing datasets.
# pd is the short name used for Pandas.

# import numpy as np
# Imports NumPy for numerical calculations.
# np is the short name used for NumPy.

# import matplotlib.pyplot as plt
# Imports Matplotlib for creating graphs and visualizations.
# plt is the short name used for pyplot.

# import seaborn as sns
# Imports Seaborn for creating statistical graphs.
# sns is the short name used for Seaborn.

# from scipy.stats import skew, kurtosis, mode
# Imports statistical functions from SciPy.
# skew() calculates the skewness of data.
# kurtosis() calculates the kurtosis of data.
# mode() calculates the most frequently occurring value.
# In this program, mode is imported but not directly used.
# Pandas mode() is used instead.

# from sklearn.model_selection import train_test_split
# Imports train_test_split.
# It divides the dataset into training and testing data.

# from sklearn.linear_model import LinearRegression, LogisticRegression
# Imports two regression algorithms.
# LinearRegression predicts continuous numerical values.
# LogisticRegression predicts categories such as 0 and 1.

# from sklearn.impute import SimpleImputer
# Imports SimpleImputer.
# It replaces missing values in the dataset.

# from sklearn.preprocessing import StandardScaler
# Imports StandardScaler.
# It standardizes numerical features to a similar scale.

# from sklearn.pipeline import make_pipeline
# Imports make_pipeline.
# It combines multiple machine learning steps into one pipeline.

# from sklearn.metrics import (
#     accuracy_score,
#     r2_score,
#     mean_squared_error,
#     mean_absolute_error,
#     classification_report,
#     confusion_matrix
# )

# accuracy_score() calculates classification accuracy.
# r2_score() measures how well a regression model fits the data.
# mean_squared_error() calculates the average squared prediction error.
# mean_absolute_error() calculates the average absolute prediction error.
# classification_report() displays precision, recall, and F1-score.
# confusion_matrix() displays correct and incorrect classifications.


# ============================================================
# 3. LOADING BOTH DATASETS
# ============================================================

# uci = pd.read_csv("diabetes.csv")
# Loads the UCI Diabetes dataset from a local CSV file.
# The dataset is stored in the variable uci.

# pima = pd.read_csv("diabetes_pima.csv")
# Loads the Pima Indians Diabetes dataset.
# The dataset is stored in the variable pima.

# datasets = {
#     "UCI Diabetes Dataset": uci,
#     "Pima Indians Diabetes Dataset": pima
# }

# Creates a dictionary containing both datasets.
# A dictionary stores data using key-value pairs.
# The dataset names are the keys.
# The loaded DataFrames are the values.
# This makes it possible to perform the same analysis on both datasets.


# ============================================================
# 4. CREATING THE UNIVARIATE ANALYSIS FUNCTION
# ============================================================

# def univariate_analysis(df, name):
# Defines a function named univariate_analysis.
# df represents the dataset passed to the function.
# name represents the name of the dataset.

# Univariate analysis means analyzing one variable at a time.
# It helps us understand the distribution and characteristics
# of individual numerical columns.


# ============================================================
# 5. DISPLAYING THE UNIVARIATE ANALYSIS HEADING
# ============================================================

# print("\n===================================")
# Prints a separator line.
# \n moves the output to a new line.

# print("UNIVARIATE ANALYSIS:", name)
# Displays the name of the dataset being analyzed.

# print("===================================")
# Prints another separator line.


# ============================================================
# 6. SELECTING NUMERICAL DATA
# ============================================================

# numeric_data = df.select_dtypes(include=np.number)
# Selects only numerical columns from the dataset.
# Numerical columns contain integer or decimal values.
# Statistical calculations are performed on these columns.


# ============================================================
# 7. CALCULATING FREQUENCY
# ============================================================

# print("\nFrequency:")
# Displays the frequency heading.

# for column in numeric_data.columns:
# Starts a loop that processes each numerical column.
# column stores the current column name.

#     print("\n", column)
# Displays the name of the current column.

#     print(numeric_data[column].value_counts().head(10))
# value_counts() counts how many times each value occurs.
# head(10) displays only the first 10 frequency results.
# Frequency tells us how often a particular value appears.


# ============================================================
# 8. CALCULATING MEAN
# ============================================================

# print("\nMean:")
# Displays the mean heading.

# print(numeric_data.mean())
# Calculates the mean of every numerical column.
# Mean is the average of all values.
# Formula:
# Mean = Sum of all values / Number of values.


# ============================================================
# 9. CALCULATING MEDIAN
# ============================================================

# print("\nMedian:")
# Displays the median heading.

# print(numeric_data.median())
# Calculates the median of every numerical column.
# Median is the middle value after arranging the data in order.
# If there are two middle values, their average is taken.


# ============================================================
# 10. CALCULATING MODE
# ============================================================

# print("\nMode:")
# Displays the mode heading.

# print(numeric_data.mode().iloc[0])
# mode() finds the most frequently occurring value in each column.
# A dataset can have more than one mode.
# iloc[0] selects the first row of the mode result.
# It displays one mode for each numerical column.


# ============================================================
# 11. CALCULATING VARIANCE
# ============================================================

# print("\nVariance:")
# Displays the variance heading.

# print(numeric_data.var())
# Calculates the variance of every numerical column.
# Variance measures how far data values are spread out
# from their mean.
# Higher variance means greater spread in the data.


# ============================================================
# 12. CALCULATING STANDARD DEVIATION
# ============================================================

# print("\nStandard Deviation:")
# Displays the standard deviation heading.

# print(numeric_data.std())
# Calculates the standard deviation of every numerical column.
# Standard deviation measures the spread of data values.
# It is the square root of variance.
# A higher value indicates greater variation in the data.


# ============================================================
# 13. CALCULATING SKEWNESS
# ============================================================

# print("\nSkewness:")
# Displays the skewness heading.

# print(numeric_data.apply(skew, nan_policy="omit"))
# apply() applies the skew() function to every numerical column.
# skew() measures the asymmetry of a data distribution.
# nan_policy="omit" ignores missing values during calculation.

# Positive skewness:
# The distribution has a longer tail on the right side.

# Negative skewness:
# The distribution has a longer tail on the left side.

# Zero skewness:
# The distribution is approximately symmetrical.


# ============================================================
# 14. CALCULATING KURTOSIS
# ============================================================

# print("\nKurtosis:")
# Displays the kurtosis heading.

# print(numeric_data.apply(kurtosis, nan_policy="omit"))
# Applies the kurtosis() function to every numerical column.
# Kurtosis describes the tail heaviness of a distribution
# compared with a normal distribution.
# SciPy uses Fisher's definition by default.

# Positive kurtosis:
# Indicates heavier tails than a normal distribution.

# Negative kurtosis:
# Indicates lighter tails than a normal distribution.

# Zero kurtosis:
# Indicates kurtosis similar to a normal distribution.


# ============================================================
# 15. CREATING HISTOGRAMS
# ============================================================

# numeric_data.hist(figsize=(12, 8))
# Creates histograms for all numerical columns.
# A histogram displays the distribution of numerical values.
# figsize=(12, 8) sets the width and height of the figure.

# plt.suptitle("Univariate Analysis - " + name)
# Sets the main title of the histogram figure.
# It includes the name of the current dataset.

# plt.tight_layout()
# Adjusts spacing between histogram plots.
# It prevents graph titles and labels from overlapping.

# plt.show()
# Displays all the histograms.


# ============================================================
# 16. CREATING THE REGRESSION ANALYSIS FUNCTION
# ============================================================

# def regression_analysis(df, name):
# Defines a function named regression_analysis.
# It performs bivariate, linear, logistic, and multiple regression analysis.
# df is the dataset passed to the function.
# name is the name of the dataset.


# ============================================================
# 17. DISPLAYING THE REGRESSION HEADING
# ============================================================

# print("\n===================================")
# Displays a separator line.

# print("REGRESSION ANALYSIS:", name)
# Displays the regression analysis heading and dataset name.

# print("===================================")
# Displays another separator line.


# ============================================================
# 18. IDENTIFYING THE TARGET COLUMN
# ============================================================

# target = "Outcome"
# Stores the name of the target column.
# Outcome represents whether a person has diabetes or not.
# Usually, 0 means non-diabetic and 1 means diabetic.

# if target not in df.columns:
# Checks whether the Outcome column exists in the dataset.

#     print("Outcome column not found in", name)
# Displays an error message if the Outcome column is missing.

#     return None
# Stops the function and returns None.
# No regression analysis is performed for this dataset.


# ============================================================
# 19. COPYING AND CLEANING THE DATASET
# ============================================================

# df = df.copy()
# Creates a separate copy of the dataset.
# Changes made to this copy do not directly modify the original DataFrame.

# df = df.select_dtypes(include=np.number)
# Keeps only numerical columns.
# Regression algorithms require numerical input features.

# df = df.replace([np.inf, -np.inf], np.nan)
# Replaces positive and negative infinity with missing values.
# np.inf represents positive infinity.
# -np.inf represents negative infinity.
# np.nan represents a missing numerical value.

# df = df.dropna(subset=[target])
# Removes rows where the Outcome column is missing.
# A missing target value cannot be used for supervised training.

# df = df[df[target].isin([0, 1])]
# Keeps only rows where Outcome is either 0 or 1.
# isin() checks whether values belong to the given list.
# This ensures that the target contains valid binary classes.


# ============================================================
# 20. SEPARATING INPUT AND OUTPUT
# ============================================================

# X = df.drop(columns=[target])
# Creates the input feature dataset.
# The Outcome column is removed from X.
# The remaining numerical columns are used for prediction.

# y = df[target]
# Stores the Outcome column in y.
# y is the target variable.
# It contains the values 0 and 1.

# if X.shape[1] == 0:
# Checks whether any numerical input features are available.
# X.shape[1] gives the number of columns in X.

#     print("No numeric input features available.")
# Displays an error message if there are no input features.

#     return None
# Stops the function if no input features are available.


# ============================================================
# 21. HANDLING MISSING INPUT VALUES
# ============================================================

# imputer = SimpleImputer(strategy="median")
# Creates a SimpleImputer object.
# strategy="median" means missing values are replaced
# with the median of the corresponding column.

# X_imputed = pd.DataFrame(
#     imputer.fit_transform(X),
#     columns=X.columns,
#     index=X.index
# )

# fit_transform() calculates the median and replaces missing values.
# X contains the original input features.
# pd.DataFrame() converts the processed array back into a DataFrame.
# columns=X.columns preserves the original column names.
# index=X.index preserves the original row indexes.

# Missing values are replaced so that machine learning
# algorithms can process the data.


# ============================================================
# 22. BIVARIATE ANALYSIS
# ============================================================

# print("\nBivariate Correlation with Outcome:")
# Displays the bivariate analysis heading.

# print(X_imputed.corrwith(y))
# Calculates the correlation between every input feature and Outcome.
# Bivariate analysis studies the relationship between two variables.
# Here, each input feature is compared with the Outcome column.

# Correlation ranges from -1 to +1.

# Positive correlation:
# As one variable increases, the other tends to increase.

# Negative correlation:
# As one variable increases, the other tends to decrease.

# Correlation near zero:
# Indicates a weak linear relationship.

# Correlation does not prove cause and effect.


# ============================================================
# 23. LINEAR REGRESSION USING ONE FEATURE
# ============================================================

# first_feature = X.columns[0]
# Selects the first input feature from the dataset.
# X.columns contains all input feature names.
# [0] selects the first feature.

# X_single = X_imputed[[first_feature]]
# Creates a dataset containing only the first input feature.
# Double square brackets preserve the DataFrame structure.
# This is used for simple linear regression.

# Linear Regression predicts a continuous numerical value.
# It tries to fit a straight-line relationship between
# the input feature and the target.


# ============================================================
# 24. SPLITTING DATA FOR LINEAR REGRESSION
# ============================================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X_single,
#     y,
#     test_size=0.2,
#     random_state=42
# )

# train_test_split() divides the dataset into training and testing data.

# X_train contains the training input feature.
# X_test contains the testing input feature.
# y_train contains the training target values.
# y_test contains the testing target values.

# test_size=0.2 means 20% of data is used for testing.
# The remaining 80% is used for training.

# random_state=42 makes the split reproducible.


# ============================================================
# 25. CREATING THE LINEAR REGRESSION MODEL
# ============================================================

# linear_model = make_pipeline(
#     StandardScaler(),
#     LinearRegression()
# )

# make_pipeline() combines scaling and regression into one pipeline.

# StandardScaler() standardizes the input feature.
# LinearRegression() creates the linear regression model.

# The pipeline first scales the input feature.
# Then it applies linear regression to make predictions.


# ============================================================
# 26. TRAINING THE LINEAR REGRESSION MODEL
# ============================================================

# linear_model.fit(X_train, y_train)
# Trains the linear regression model using training data.
# It learns the relationship between the selected feature and Outcome.

# linear_prediction = linear_model.predict(X_test)
# Predicts Outcome values for the testing data.
# The predictions are continuous numerical values.


# ============================================================
# 27. EVALUATING LINEAR REGRESSION
# ============================================================

# print("\nLinear Regression using:", first_feature)
# Displays the name of the feature used for linear regression.

# print("R2 Score:", r2_score(y_test, linear_prediction))
# Calculates the R-squared score.
# R2 measures how well the model explains the variation in the target.
# A score closer to 1 generally indicates a better fit.
# R2 can also be negative if the model performs worse
# than predicting the mean of the actual target values.

# print(
#     "RMSE:",
#     np.sqrt(mean_squared_error(y_test, linear_prediction))
# )

# mean_squared_error() calculates the average squared prediction error.
# np.sqrt() calculates its square root.
# RMSE stands for Root Mean Squared Error.
# It measures the typical size of prediction errors.
# Lower RMSE generally indicates better prediction performance.


# ============================================================
# 28. LOGISTIC REGRESSION
# ============================================================

# Logistic Regression is used for classification.
# It predicts the probability of belonging to a class.
# Here, it predicts whether Outcome is 0 or 1.

# X_train, X_test, y_train, y_test = train_test_split(
#     X_imputed,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# Splits the complete input dataset into training and testing data.
# Unlike simple linear regression, all input features are used.

# test_size=0.2 means 20% of the dataset is used for testing.
# random_state=42 makes the split reproducible.
# stratify=y maintains the proportion of Outcome classes
# in both training and testing data.


# ============================================================
# 29. CREATING THE LOGISTIC REGRESSION MODEL
# ============================================================

# logistic_model = make_pipeline(
#     StandardScaler(),
#     LogisticRegression(max_iter=1000)
# )

# make_pipeline() combines scaling and classification.

# StandardScaler() standardizes all input features.
# LogisticRegression() creates a logistic regression classifier.

# max_iter=1000 allows up to 1000 iterations during model training.
# It gives the algorithm more iterations to find suitable parameters.

# Logistic Regression uses a logistic or sigmoid function
# to estimate the probability of the positive class.
# The predicted probability is converted into a class label.


# ============================================================
# 30. TRAINING THE LOGISTIC REGRESSION MODEL
# ============================================================

# logistic_model.fit(X_train, y_train)
# Trains the logistic regression model using training data.
# It learns the relationship between the input features and Outcome.

# logistic_prediction = logistic_model.predict(X_test)
# Predicts the diabetes class for each testing record.
# The predictions are either 0 or 1.


# ============================================================
# 31. CALCULATING LOGISTIC REGRESSION ACCURACY
# ============================================================

# accuracy = accuracy_score(y_test, logistic_prediction)
# Compares actual and predicted classes.
# Calculates the proportion of correct classifications.

# print("\nLogistic Regression Results:")
# Displays the logistic regression results heading.

# print("Accuracy:", accuracy * 100, "%")
# Displays classification accuracy as a percentage.
# Multiplying by 100 converts the score into a percentage.
# Higher accuracy means more correct predictions.


# ============================================================
# 32. DISPLAYING THE CONFUSION MATRIX
# ============================================================

# print("\nConfusion Matrix:")
# Displays the confusion matrix heading.

# print(confusion_matrix(y_test, logistic_prediction))
# Creates and displays the confusion matrix.

# For binary classification, the matrix contains:
# True Negative (TN): Correctly predicted class 0.
# False Positive (FP): Actual class 0 predicted as class 1.
# False Negative (FN): Actual class 1 predicted as class 0.
# True Positive (TP): Correctly predicted class 1.

# The diagonal values represent correct predictions.
# The other values represent incorrect predictions.


# ============================================================
# 33. DISPLAYING THE CLASSIFICATION REPORT
# ============================================================

# print("\nClassification Report:")
# Displays the classification report heading.

# print(classification_report(y_test, logistic_prediction))
# Displays precision, recall, F1-score, and support.

# Precision:
# Measures how many predicted positive cases are actually positive.

# Recall:
# Measures how many actual positive cases are correctly identified.

# F1-score:
# It is the harmonic mean of precision and recall.

# Support:
# Represents the number of actual records in each class.

# Class 0 represents people without diabetes.
# Class 1 represents people with diabetes.


# ============================================================
# 34. MULTIPLE REGRESSION ANALYSIS
# ============================================================

# Multiple Regression uses more than one input feature
# to predict a target variable.

# multiple_model = make_pipeline(
#     StandardScaler(),
#     LinearRegression()
# )

# Creates a pipeline containing StandardScaler and LinearRegression.
# StandardScaler standardizes all input features.
# LinearRegression creates the multiple linear regression model.


# ============================================================
# 35. TRAINING THE MULTIPLE REGRESSION MODEL
# ============================================================

# multiple_model.fit(X_train, y_train)
# Trains the multiple regression model using all input features.
# It learns how multiple features are related to Outcome.

# multiple_prediction = multiple_model.predict(X_test)
# Predicts continuous Outcome values for the testing records.
# These predictions may be decimal values because LinearRegression
# predicts continuous numerical outputs.


# ============================================================
# 36. CALCULATING MULTIPLE REGRESSION R2 SCORE
# ============================================================

# multiple_r2 = r2_score(y_test, multiple_prediction)
# Calculates the R-squared score for multiple regression.
# It measures how much variation in the target is explained
# by the multiple regression model.

# A score closer to 1 generally indicates a better fit.
# A negative score indicates poor performance compared
# with predicting the mean of the target values.


# ============================================================
# 37. CALCULATING MULTIPLE REGRESSION RMSE
# ============================================================

# multiple_rmse = np.sqrt(
#     mean_squared_error(y_test, multiple_prediction)
# )

# mean_squared_error() calculates the average squared error.
# np.sqrt() calculates the square root of that error.
# RMSE measures the size of prediction errors.
# Lower RMSE generally indicates better performance.


# ============================================================
# 38. CALCULATING MULTIPLE REGRESSION MAE
# ============================================================

# multiple_mae = mean_absolute_error(
#     y_test,
#     multiple_prediction
# )

# mean_absolute_error() calculates the average absolute difference
# between actual and predicted values.

# MAE stands for Mean Absolute Error.
# Lower MAE indicates smaller average prediction errors.


# ============================================================
# 39. DISPLAYING MULTIPLE REGRESSION RESULTS
# ============================================================

# print("\nMultiple Regression Results:")
# Displays the multiple regression heading.

# print("R2 Score:", multiple_r2)
# Displays the R-squared score.

# print("RMSE:", multiple_rmse)
# Displays the Root Mean Squared Error.

# print("MAE:", multiple_mae)
# Displays the Mean Absolute Error.


# ============================================================
# 40. RETURNING THE RESULTS
# ============================================================

# return {
#     "Dataset": name,
#     "Linear Regression R2": r2_score(y_test, multiple_prediction),
#     "Logistic Regression Accuracy": accuracy,
#     "Multiple Regression R2": multiple_r2,
#     "Multiple Regression RMSE": multiple_rmse,
#     "Multiple Regression MAE": multiple_mae
# }

# return sends the calculated results back to the main program.

# Dataset stores the dataset name.
# Linear Regression R2 stores a regression performance score.
# Logistic Regression Accuracy stores the classification accuracy.
# Multiple Regression R2 stores the multiple regression score.
# Multiple Regression RMSE stores the root mean squared error.
# Multiple Regression MAE stores the mean absolute error.

# The results are stored as a dictionary.

# Important note:
# The Linear Regression R2 value in this return statement
# is calculated using multiple_prediction and the multiple-regression
# testing data, not the earlier linear_prediction.
# Therefore, it is actually the multiple regression R2 value.
# The displayed linear regression R2 value is calculated separately.


# ============================================================
# 41. PERFORMING ANALYSIS ON BOTH DATASETS
# ============================================================

# results = []
# Creates an empty list.
# It stores the final results of both datasets.

# for name, df in datasets.items():
# Loops through each dataset stored in the dictionary.
# name stores the dataset name.
# df stores the corresponding DataFrame.

#     print("\n\nDATASET:", name)
# Displays the current dataset name.

#     print("\nFirst Five Records:")
# Displays a heading for the first five records.

#     print(df.head())
# Displays the first five rows of the current dataset.

#     univariate_analysis(df, name)
# Calls the univariate analysis function.
# Calculates frequency, mean, median, mode, variance,
# standard deviation, skewness, and kurtosis.
# It also displays histograms.

#     result = regression_analysis(df, name)
# Calls the regression analysis function.
# Performs bivariate correlation, linear regression,
# logistic regression, and multiple regression.
# Stores the returned result in result.

#     if result is not None:
# Checks whether the regression function returned valid results.

#         results.append(result)
# Adds the returned result dictionary to the results list.
# This stores the results for later comparison.


# ============================================================
# 42. COMPARING BOTH DATASETS
# ============================================================

# print("\n===================================")
# Displays a separator line.

# print("COMPARISON OF BOTH DATASETS")
# Displays the dataset comparison heading.

# print("===================================")
# Displays another separator line.

# comparison = pd.DataFrame(results)
# Converts the list of result dictionaries into a DataFrame.
# Each row represents one dataset.
# Each column represents one evaluation metric.

# print(comparison.to_string(index=False))
# Displays the complete comparison table.
# to_string() converts the DataFrame into readable text.
# index=False hides the row index numbers.


# ============================================================
# 43. PLOTTING LOGISTIC REGRESSION ACCURACY COMPARISON
# ============================================================

# if not comparison.empty:
# Checks whether the comparison DataFrame contains any results.
# empty returns True if the DataFrame has no rows.

#     plt.figure(figsize=(8, 5))
# Creates a new figure.
# Width is 8 inches and height is 5 inches.

#     sns.barplot(
#         data=comparison,
#         x="Dataset",
#         y="Logistic Regression Accuracy"
#     )

# Creates a bar graph using Seaborn.
# data=comparison specifies the comparison DataFrame.
# x="Dataset" places the dataset names on the horizontal axis.
# y="Logistic Regression Accuracy" displays accuracy on the vertical axis.
# Each bar represents one dataset's logistic regression accuracy.

#     plt.title("Diabetes Dataset Accuracy Comparison")
# Sets the graph title.

#     plt.ylabel("Accuracy")
# Sets the vertical axis label.

#     plt.xticks(rotation=15)
# Rotates the dataset names by 15 degrees.
# It makes the labels easier to read.

#     plt.tight_layout()
# Adjusts the spacing of the graph.
# It prevents labels from being cut off.

#     plt.show()
# Displays the final comparison bar graph.


# ============================================================
# 44. UNDERSTANDING THE COMPLETE OUTPUT
# ============================================================

# First Five Records:
# Displays the first five rows of each diabetes dataset.

# Univariate Analysis:
# Displays frequency, mean, median, mode, variance,
# standard deviation, skewness, and kurtosis.

# Histograms:
# Show the distribution of individual numerical features.

# Bivariate Correlation:
# Shows the relationship between each input feature and Outcome.

# Linear Regression Results:
# Displays R2 Score and RMSE using the first input feature.

# Logistic Regression Results:
# Displays classification accuracy for diabetes prediction.

# Confusion Matrix:
# Shows correct and incorrect diabetes classifications.

# Classification Report:
# Displays precision, recall, F1-score, and support.

# Multiple Regression Results:
# Displays R2 Score, RMSE, and MAE using multiple input features.

# Comparison Table:
# Compares the calculated metrics for both datasets.

# Accuracy Comparison Graph:
# Compares logistic regression accuracy for both datasets.


# ============================================================
# 45. IMPORTANT STATISTICAL CONCEPTS FOR VIVA
# ============================================================

# Q1. What is data analysis?
# Data analysis is the process of inspecting and studying data
# to identify patterns, relationships, and useful information.

# Q2. What is univariate analysis?
# Univariate analysis studies one variable at a time.
# It uses mean, median, mode, variance, and graphs.

# Q3. What is bivariate analysis?
# Bivariate analysis studies the relationship between two variables.
# In this practical, input features are compared with Outcome.

# Q4. What is multiple regression?
# Multiple regression uses two or more input features
# to predict a target variable.

# Q5. What is frequency?
# Frequency tells us how many times a particular value occurs.

# Q6. What is mean?
# Mean is the average of all values.
# Mean = Sum of values / Number of values.

# Q7. What is median?
# Median is the middle value after sorting the data.

# Q8. What is mode?
# Mode is the value that occurs most frequently.

# Q9. What is variance?
# Variance measures how far values are spread out
# from their mean.

# Q10. What is standard deviation?
# Standard deviation is the square root of variance.
# It measures the spread of data around the mean.

# Q11. What is skewness?
# Skewness measures the asymmetry of a data distribution.
# Positive skewness means a longer right tail.
# Negative skewness means a longer left tail.

# Q12. What is kurtosis?
# Kurtosis describes the heaviness of the tails
# of a distribution compared with a normal distribution.

# Q13. What is regression?
# Regression is a supervised learning technique.
# It is used to model relationships and predict numerical values.

# Q14. What is Linear Regression?
# Linear Regression predicts continuous numerical values
# using a linear relationship between input and output.

# Q15. What is Logistic Regression?
# Logistic Regression is a classification algorithm.
# It predicts the probability of a class.
# Here, it predicts diabetes Outcome as 0 or 1.

# Q16. What is simple linear regression?
# Simple Linear Regression uses only one input feature
# to predict the target.

# Q17. What is multiple linear regression?
# Multiple Linear Regression uses two or more input features
# to predict the target.

# Q18. What is the difference between Linear and Logistic Regression?
# Linear Regression predicts continuous numerical values.
# Logistic Regression predicts categories or class labels.

# Q19. What is R2 Score?
# R2 Score measures how well a regression model explains
# variation in the target variable.
# A score closer to 1 generally indicates a better fit.

# Q20. What is RMSE?
# RMSE stands for Root Mean Squared Error.
# It measures the square root of the average squared prediction error.
# Lower RMSE generally indicates better performance.

# Q21. What is MAE?
# MAE stands for Mean Absolute Error.
# It measures the average absolute difference
# between actual and predicted values.

# Q22. What is correlation?
# Correlation measures the strength and direction
# of a relationship between two variables.
# Its value ranges from -1 to +1.

# Q23. What is SimpleImputer?
# SimpleImputer is used to replace missing values.
# In this practical, missing values are replaced with the median.

# Q24. Why is median used for missing values?
# Median is less affected by extreme values than the mean.
# It is useful when data contains outliers.

# Q25. Why is StandardScaler used?
# It standardizes input features to a similar scale.
# It is useful for algorithms such as Logistic Regression.

# Q26. What is a pipeline?
# A pipeline combines multiple machine learning steps.
# Here, it combines StandardScaler and regression models.

# Q27. What is a confusion matrix?
# A confusion matrix compares actual and predicted classes.
# It shows true positives, true negatives,
# false positives, and false negatives.

# Q28. What is accuracy?
# Accuracy is the proportion of correct classifications.
# Accuracy = Correct Predictions / Total Predictions.

# Q29. What is precision?
# Precision measures how many predicted positive cases are correct.

# Q30. What is recall?
# Recall measures how many actual positive cases are correctly identified.

# Q31. What is F1-score?
# F1-score is the harmonic mean of precision and recall.

# Q32. What is a histogram?
# A histogram displays the frequency distribution
# of numerical data using bars.

# Q33. What is the difference between univariate and bivariate analysis?
# Univariate analysis studies one variable.
# Bivariate analysis studies two variables.

# Q34. What is the difference between bivariate and multiple regression?
# Bivariate analysis studies the relationship between two variables.
# Multiple regression uses multiple input features
# to predict one target variable.

# Q35. What are the applications of regression analysis?
# Diabetes prediction.
# House price prediction.
# Sales forecasting.
# Medical risk analysis.
# Financial forecasting.


# ============================================================
# 46. CONCLUSION
# ============================================================

# In this practical, univariate, bivariate, and regression analysis
# are performed using UCI Diabetes and Pima Indians Diabetes datasets.

# First, both datasets are loaded and inspected.
# Univariate analysis calculates frequency, mean, median, mode,
# variance, standard deviation, skewness, and kurtosis.

# Histograms are created to understand the distribution
# of numerical features.

# Bivariate analysis calculates the correlation
# between input features and the Outcome column.

# Simple Linear Regression is performed using one input feature.
# Logistic Regression is used to classify diabetes outcomes.
# Multiple Linear Regression is performed using multiple features.

# Finally, both datasets are compared using regression metrics
# and a logistic regression accuracy bar graph.

# Therefore, statistical analysis and regression techniques
# help understand diabetes data and predict diabetes outcomes.
