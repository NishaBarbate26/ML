# AIM: Implement K-Nearest Neighbours (KNN) Algorithm on
# Social Network Ads Dataset and calculate:
# Confusion Matrix, Accuracy, Error Rate, Precision and Recall.

# Installation commands:
# Windows:
# python -m pip install pandas numpy matplotlib scikit-learn
# python practical_knn.py
#
# Ubuntu/Linux:
# pip install pandas numpy matplotlib scikit-learn
# python3 -m pip install pandas numpy matplotlib scikit-learn
# python3 practical_knn.py


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)

# Load dataset

df = pd.read_csv("Social_Network_Ads.csv")

print("Original Dataset:")
print(df.head())

# Remove unnecessary columns

df = df.drop(columns=["User ID"], errors="ignore")

# Remove missing values

df = df.dropna()

# Separate input and output

X = df.drop(columns=["Purchased"])
y = df["Purchased"]

# Convert categorical columns into numeric columns

X = pd.get_dummies(X, drop_first=True, dtype=int)

# Split dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Standardize the features

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Create KNN model

knn = KNeighborsClassifier(n_neighbors=5)

# Train the model

knn.fit(X_train, y_train)

# Predict the test data

y_pred = knn.predict(X_test)

# Calculate confusion matrix

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nConfusion Matrix:")
print(cm)

# Calculate performance metrics

accuracy = accuracy_score(y_test, y_pred)

error_rate = 1 - accuracy

precision = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

print("\nKNN Classification Results")

print("Accuracy:", accuracy * 100, "%")

print("Error Rate:", error_rate * 100, "%")

print("Precision:", precision)

print("Recall:", recall)

print("\nClassification Report:")

print(classification_report(y_test, y_pred, zero_division=0))

# Display confusion matrix

plt.figure(figsize=(6, 5))

plt.imshow(cm, cmap="Blues")

plt.title("KNN Confusion Matrix")

plt.xlabel("Predicted Values")

plt.ylabel("Actual Values")

plt.xticks([0, 1], ["Not Purchased", "Purchased"])

plt.yticks([0, 1], ["Not Purchased", "Purchased"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()

plt.tight_layout()

plt.show()

# Predict purchase for a new user

print("\nEnter New User Details:")

age = float(input("Enter Age: "))

salary = float(input("Enter Estimated Salary: "))

new_user = pd.DataFrame(
    [[age, salary]],
    columns=["Age", "EstimatedSalary"]
)

if "Gender" in df.columns:
    gender = input("Enter Gender (Male/Female): ").strip().capitalize()

    new_user["Gender_Male"] = 1 if gender == "Male" else 0

new_user = new_user.reindex(
    columns=X_train.shape[1] * [None]
) if False else new_user

# Match the new user's features with training feature columns

new_user = pd.DataFrame(
    0,
    index=[0],
    columns=X.columns
)

new_user["Age"] = age
new_user["EstimatedSalary"] = salary

if "Gender" in df.columns:
    new_user["Gender_Male"] = 1 if gender == "Male" else 0

new_user_scaled = scaler.transform(new_user)

prediction = knn.predict(new_user_scaled)

if prediction[0] == 1:
    print("\nPrediction: User will purchase the product.")
else:
    print("\nPrediction: User will not purchase the product.")


# ============================================================
# AIM:
# Implement K-Nearest Neighbors (KNN) Classification Algorithm
# Using the Social Network Ads Dataset.
# ============================================================


# ============================================================
# 1. INSTALLATION AND EXECUTION COMMANDS
# ============================================================

# Windows:
# python -m pip install pandas numpy matplotlib scikit-learn
# python practical_knn.py

# Ubuntu/Linux:
# pip install pandas numpy matplotlib scikit-learn
# python3 -m pip install pandas numpy matplotlib scikit-learn
# python3 practical_knn.py


# ============================================================
# 2. IMPORTING LIBRARIES
# ============================================================

# import pandas as pd
# Imports Pandas for loading and processing the dataset.
# pd is the short name used for Pandas.

# import numpy as np
# Imports NumPy for numerical calculations.
# np is the short name used for NumPy.
# NumPy is imported here but is not directly used in this program.

# import matplotlib.pyplot as plt
# Imports Matplotlib for creating graphs and visualizations.
# plt is the short name used for pyplot.

# from sklearn.model_selection import train_test_split
# Imports train_test_split.
# It divides the dataset into training and testing data.

# from sklearn.preprocessing import StandardScaler
# Imports StandardScaler.
# It standardizes numerical features to a similar scale.

# from sklearn.neighbors import KNeighborsClassifier
# Imports the KNN classification algorithm.
# KNN predicts a class based on the nearest data points.

# from sklearn.metrics import (
#     confusion_matrix,
#     accuracy_score,
#     precision_score,
#     recall_score,
#     classification_report
# )

# confusion_matrix() compares actual and predicted classes.
# accuracy_score() calculates the percentage of correct predictions.
# precision_score() calculates precision.
# recall_score() calculates recall.
# classification_report() displays precision, recall, F1-score,
# and support for each class.


# ============================================================
# 3. LOADING THE DATASET
# ============================================================

# df = pd.read_csv("Social_Network_Ads.csv")
# Loads the Social Network Ads dataset from a CSV file.
# The dataset is stored in the DataFrame named df.
# It contains information about users and their purchase decisions.

# print("Original Dataset:")
# Displays the heading for the original dataset.

# print(df.head())
# Displays the first five rows of the dataset.
# head() is used to inspect the dataset.


# ============================================================
# 4. REMOVING UNNECESSARY COLUMNS
# ============================================================

# df = df.drop(columns=["User ID"], errors="ignore")
# Removes the User ID column from the dataset.
# User ID is only an identification number.
# It does not provide useful information for predicting purchases.
# columns=["User ID"] specifies the column to remove.
# errors="ignore" prevents an error if the column does not exist.


# ============================================================
# 5. REMOVING MISSING VALUES
# ============================================================

# df = df.dropna()
# Removes rows containing missing values.
# Missing values are empty or unavailable data.
# KNN requires valid input values to calculate distances.
# dropna() removes rows containing missing values.


# ============================================================
# 6. SEPARATING INPUT AND OUTPUT
# ============================================================

# X = df.drop(columns=["Purchased"])
# Creates the input feature dataset.
# The Purchased column is removed from X.
# The remaining columns are used to predict the purchase decision.

# y = df["Purchased"]
# Stores the Purchased column in y.
# y is the target variable.
# It contains the purchase decision.
# 0 means the user did not purchase the product.
# 1 means the user purchased the product.


# ============================================================
# 7. CONVERTING CATEGORICAL DATA INTO NUMERICAL DATA
# ============================================================

# X = pd.get_dummies(X, drop_first=True, dtype=int)
# Converts categorical columns into numerical columns.
# Machine learning algorithms require numerical input values.

# get_dummies() performs One-Hot Encoding.
# It creates separate columns for categorical values.

# drop_first=True removes the first category column.
# This avoids unnecessary duplicate information in encoded columns.

# dtype=int stores the encoded values as integers.
# For example:
# Gender_Male = 1 means Male.
# Gender_Male = 0 means Female, assuming Male is the encoded category.


# ============================================================
# 8. SPLITTING THE DATASET
# ============================================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.25,
#     random_state=42,
#     stratify=y
# )

# train_test_split() divides the dataset into training
# and testing data.

# X_train contains the training input features.
# X_test contains the testing input features.
# y_train contains the training target values.
# y_test contains the testing target values.

# test_size=0.25 means 25% of the data is used for testing.
# The remaining 75% is used for training.

# random_state=42 makes the split reproducible.
# It helps produce the same split when the program is run again.

# stratify=y maintains the proportion of purchase classes
# in both training and testing datasets.

# Example:
# If the original dataset contains more users who did not purchase,
# stratify helps maintain a similar class proportion
# in the training and testing data.


# ============================================================
# 9. STANDARDIZING THE FEATURES
# ============================================================

# scaler = StandardScaler()
# Creates a StandardScaler object.
# It is used to standardize the input features.

# X_train = scaler.fit_transform(X_train)
# fit() calculates the mean and standard deviation
# of each training feature.
# transform() standardizes the training data.
# The standardized training data replaces X_train.

# X_test = scaler.transform(X_test)
# Standardizes the testing data using the mean and
# standard deviation calculated from training data.
# The test data must not be used to calculate the scaling values.

# Standardization formula:
# Standardized Value = (Value - Mean) / Standard Deviation

# Standardization converts features to a comparable scale.
# It is important for KNN because KNN uses distance calculations.

# Example:
# Age may range from 18 to 60.
# Estimated Salary may range from 15,000 to 150,000.
# Without standardization, salary could dominate the distance
# calculation because its numerical values are much larger.


# ============================================================
# 10. CREATING THE KNN MODEL
# ============================================================

# knn = KNeighborsClassifier(n_neighbors=5)
# Creates a KNN classification model.
# n_neighbors=5 means the model considers the five nearest
# training data points when making a prediction.

# KNN stands for K-Nearest Neighbors.
# It is a supervised machine learning algorithm.
# It can be used for classification and regression.

# K represents the number of nearest neighbors considered.

# KNN works by:
# 1. Selecting a value of K.
# 2. Calculating the distance between the new point
#    and all training data points.
# 3. Finding the K nearest data points.
# 4. Checking the classes of those neighbors.
# 5. Assigning the majority class to the new point.

# Example:
# If K = 5 and the nearest neighbors contain:
# 3 Purchased users and 2 Not Purchased users,
# the predicted class will be Purchased.


# ============================================================
# 11. TRAINING THE KNN MODEL
# ============================================================

# knn.fit(X_train, y_train)
# Trains the KNN model using training data.
# It stores the training features and their class labels.
# KNN is called a lazy learning algorithm because
# most of its work happens during prediction.


# ============================================================
# 12. PREDICTING THE TEST DATA
# ============================================================

# y_pred = knn.predict(X_test)
# Predicts the purchase decision for every testing record.
# The predicted values are stored in y_pred.
# 0 means Not Purchased.
# 1 means Purchased.


# ============================================================
# 13. CALCULATING THE CONFUSION MATRIX
# ============================================================

# cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
# Creates a confusion matrix using actual and predicted values.
# y_test contains actual classes.
# y_pred contains predicted classes.
# labels=[0, 1] fixes the order of the classes.

# The confusion matrix is a table used to evaluate classification.

# It has four important values:

# True Negative (TN):
# Actual class is 0 and predicted class is 0.
# The user did not purchase, and the model predicted correctly.

# False Positive (FP):
# Actual class is 0 but predicted class is 1.
# The user did not purchase, but the model predicted purchase.

# False Negative (FN):
# Actual class is 1 but predicted class is 0.
# The user purchased, but the model predicted no purchase.

# True Positive (TP):
# Actual class is 1 and predicted class is 1.
# The user purchased, and the model predicted correctly.

# For labels=[0, 1], the matrix format is:
# [[TN, FP],
#  [FN, TP]]

# print("\nConfusion Matrix:")
# Displays the confusion matrix heading.

# print(cm)
# Displays the confusion matrix values.


# ============================================================
# 14. CALCULATING ACCURACY
# ============================================================

# accuracy = accuracy_score(y_test, y_pred)
# Calculates the accuracy of the KNN model.
# It compares actual and predicted values.

# Accuracy formula:
# Accuracy = Correct Predictions / Total Predictions

# print("\nKNN Classification Results")
# Displays the KNN results heading.

# print("Accuracy:", accuracy * 100, "%")
# Displays accuracy as a percentage.
# Multiplying by 100 converts the decimal score into a percentage.

# Higher accuracy means more testing records
# were classified correctly.


# ============================================================
# 15. CALCULATING ERROR RATE
# ============================================================

# error_rate = 1 - accuracy
# Calculates the error rate.
# Error rate represents the proportion of incorrect predictions.

# Error Rate Formula:
# Error Rate = 1 - Accuracy

# print("Error Rate:", error_rate * 100, "%")
# Displays the error rate as a percentage.
# Lower error rate indicates fewer incorrect predictions.


# ============================================================
# 16. CALCULATING PRECISION
# ============================================================

# precision = precision_score(
#     y_test,
#     y_pred,
#     pos_label=1,
#     zero_division=0
# )

# precision_score() calculates precision for the positive class.
# Here, class 1 represents Purchased.

# pos_label=1 means the Purchased class is treated as positive.

# zero_division=0 returns zero if precision cannot be calculated
# because there are no positive predictions.

# Precision formula:
# Precision = TP / (TP + FP)

# Precision tells us how many users predicted as Purchased
# actually purchased the product.

# Higher precision means fewer false positive predictions.


# ============================================================
# 17. CALCULATING RECALL
# ============================================================

# recall = recall_score(
#     y_test,
#     y_pred,
#     pos_label=1,
#     zero_division=0
# )

# recall_score() calculates recall for the positive class.
# Here, class 1 represents Purchased.

# Recall formula:
# Recall = TP / (TP + FN)

# Recall tells us how many actual purchasers
# were correctly identified by the model.

# Higher recall means fewer actual purchasers are missed.


# ============================================================
# 18. DISPLAYING PERFORMANCE METRICS
# ============================================================

# print("Precision:", precision)
# Displays the precision value.

# print("Recall:", recall)
# Displays the recall value.

# These metrics help evaluate the performance of the KNN model.
# Accuracy measures overall correctness.
# Precision measures correctness among predicted purchasers.
# Recall measures how many actual purchasers were identified.


# ============================================================
# 19. DISPLAYING THE CLASSIFICATION REPORT
# ============================================================

# print("\nClassification Report:")
# Displays the classification report heading.

# print(classification_report(y_test, y_pred, zero_division=0))
# Displays the classification performance for both classes.

# Precision:
# Measures how many predicted positive cases are actually positive.

# Recall:
# Measures how many actual positive cases are correctly identified.

# F1-score:
# It is the harmonic mean of precision and recall.
# It balances precision and recall.

# Support:
# Represents the number of actual records in each class.

# The report displays results separately for:
# Class 0: Not Purchased.
# Class 1: Purchased.

# Macro Average:
# Calculates the unweighted average across classes.

# Weighted Average:
# Calculates the average based on the number of records
# in each class.


# ============================================================
# 20. DISPLAYING THE CONFUSION MATRIX GRAPH
# ============================================================

# plt.figure(figsize=(6, 5))
# Creates a new graph.
# Width is 6 inches and height is 5 inches.

# plt.imshow(cm, cmap="Blues")
# Displays the confusion matrix as a colored image.
# cm contains the confusion matrix values.
# cmap="Blues" uses different shades of blue.

# plt.title("KNN Confusion Matrix")
# Sets the graph title.

# plt.xlabel("Predicted Values")
# Sets the horizontal axis label.
# It represents the classes predicted by the model.

# plt.ylabel("Actual Values")
# Sets the vertical axis label.
# It represents the actual classes.

# plt.xticks([0, 1], ["Not Purchased", "Purchased"])
# Sets the horizontal axis tick positions and labels.
# 0 represents Not Purchased.
# 1 represents Purchased.

# plt.yticks([0, 1], ["Not Purchased", "Purchased"])
# Sets the vertical axis tick positions and labels.
# 0 represents Not Purchased.
# 1 represents Purchased.

# for i in range(2):
# Starts a loop for the two actual classes.
# i represents the row position.

#     for j in range(2):
#     Starts another loop for the two predicted classes.
#     j represents the column position.

#         plt.text(j, i, cm[i, j], ha="center", va="center")
#         Displays the confusion matrix value inside each cell.
#         j represents the horizontal position.
#         i represents the vertical position.
#         cm[i, j] represents the value in that cell.
#         ha="center" aligns the text horizontally.
#         va="center" aligns the text vertically.

# plt.colorbar()
# Displays a color scale beside the confusion matrix.
# It helps understand the different cell colors.

# plt.tight_layout()
# Adjusts spacing so that graph labels are not cut off.

# plt.show()
# Displays the confusion matrix graph.


# ============================================================
# 21. PREDICTING PURCHASE FOR A NEW USER
# ============================================================

# print("\nEnter New User Details:")
# Displays a heading asking for new user information.

# age = float(input("Enter Age: "))
# Takes the user's age as input.
# input() accepts information from the keyboard.
# float() converts the entered value into a decimal number.
# The value is stored in age.

# salary = float(input("Enter Estimated Salary: "))
# Takes the estimated salary as input.
# float() converts the input into a numerical value.
# The value is stored in salary.

# new_user = pd.DataFrame(
#     [[age, salary]],
#     columns=["Age", "EstimatedSalary"]
# )

# Creates a DataFrame for the new user.
# It initially contains Age and EstimatedSalary.
# The column names must match the training feature names.


# ============================================================
# 22. HANDLING GENDER INPUT
# ============================================================

# if "Gender" in df.columns:
# Checks whether the original dataset contains a Gender column.

#     gender = input("Enter Gender (Male/Female): ").strip().capitalize()
#     Takes the gender as input.
#     strip() removes extra spaces from the beginning and end.
#     capitalize() makes the first letter uppercase.
#     Example: "male" becomes "Male".

#     new_user["Gender_Male"] = 1 if gender == "Male" else 0
#     Creates the encoded Gender_Male column.
#     If the entered gender is Male, the value becomes 1.
#     Otherwise, the value becomes 0.
#     This matches the assumed One-Hot Encoding used during training.

# Important:
# This encoding assumes the only categories are Male and Female.
# If the dataset has different gender categories, the encoding
# must be adjusted to match the training data.


# ============================================================
# 23. AN UNNECESSARY CONDITIONAL STATEMENT
# ============================================================

# new_user = new_user.reindex(
#     columns=X_train.shape[1] * [None]
# ) if False else new_user

# This statement does not change new_user.
# The condition is False.
# Therefore, the else part returns the existing new_user.
# This line is unnecessary and can be removed.
# It has no effect on the prediction.


# ============================================================
# 24. MATCHING NEW USER FEATURES WITH TRAINING FEATURES
# ============================================================

# new_user = pd.DataFrame(
#     0,
#     index=[0],
#     columns=X.columns
# )

# Creates a new DataFrame with exactly the same feature columns as X.
# All feature values are initially set to zero.
# index=[0] creates one row for the new user.
# columns=X.columns ensures the feature names and order match training.

# This step is important because the model expects
# the same input features used during training.


# ============================================================
# 25. ADDING NEW USER VALUES
# ============================================================

# new_user["Age"] = age
# Stores the entered age in the Age column.

# new_user["EstimatedSalary"] = salary
# Stores the entered salary in the EstimatedSalary column.

# if "Gender" in df.columns:
# Checks whether Gender exists in the original dataset.

#     new_user["Gender_Male"] = 1 if gender == "Male" else 0
# Stores the encoded gender value.
# Male is represented by 1.
# Female is represented by 0.

# All other feature columns remain zero.
# This matches the one-hot encoded feature structure.


# ============================================================
# 26. STANDARDIZING THE NEW USER DATA
# ============================================================

# new_user_scaled = scaler.transform(new_user)
# Standardizes the new user's input features.
# It uses the same scaler that was fitted on training data.
# The new user's data must be scaled before prediction.
# This ensures that the distance calculation is consistent
# with the training data.


# ============================================================
# 27. PREDICTING THE NEW USER'S PURCHASE DECISION
# ============================================================

# prediction = knn.predict(new_user_scaled)
# Uses the trained KNN model to predict the new user's purchase decision.
# The prediction is stored in prediction.
# The output is either 0 or 1.

# if prediction[0] == 1:
# Checks whether the predicted class is 1.
# prediction[0] accesses the first prediction.

#     print("\nPrediction: User will purchase the product.")
# Displays the message if the predicted class is 1.

# else:
# Executes when the predicted class is not 1.

#     print("\nPrediction: User will not purchase the product.")
# Displays the message if the predicted class is 0.


# ============================================================
# 28. UNDERSTANDING THE COMPLETE OUTPUT
# ============================================================

# Original Dataset:
# Displays the first five records of Social_Network_Ads.csv.

# Confusion Matrix:
# Displays the number of correct and incorrect predictions.

# Accuracy:
# Displays the percentage of correctly classified users.

# Error Rate:
# Displays the percentage of incorrect predictions.

# Precision:
# Displays how many predicted purchasers actually purchased.

# Recall:
# Displays how many actual purchasers were correctly identified.

# Classification Report:
# Displays precision, recall, F1-score, and support for both classes.

# Confusion Matrix Graph:
# Shows the confusion matrix using different shades of blue.

# New User Prediction:
# Takes age, estimated salary, and optionally gender as input.
# Predicts whether the user will purchase the product.


# ============================================================
# 29. IMPORTANT KNN CONCEPTS FOR VIVA
# ============================================================

# Q1. What is KNN?
# KNN stands for K-Nearest Neighbors.
# It is a supervised machine learning algorithm.
# It predicts a class based on the nearest training data points.

# Q2. Why is KNN called a supervised learning algorithm?
# Because it learns from training data that already contains labels.

# Q3. What is K in KNN?
# K represents the number of nearest neighbors considered
# while making a prediction.

# Q4. What is the value of K in this practical?
# The value of K is 5.
# The model considers five nearest neighbors.

# Q5. Why do we use KNN?
# KNN is used for classification and regression problems.
# Here, it predicts whether a user will purchase a product.

# Q6. What is the working principle of KNN?
# It calculates distances between data points.
# It selects the K nearest neighbors.
# It predicts the majority class among those neighbors.

# Q7. What is Euclidean distance?
# Euclidean distance measures the straight-line distance
# between two points.

# Formula:
# d = sqrt((x2 - x1)^2 + (y2 - y1)^2)

# Q8. Why is StandardScaler used in KNN?
# KNN uses distance calculations.
# Features with larger numerical values can dominate the distance.
# StandardScaler brings features to a comparable scale.

# Q9. What happens if K is too small?
# The model may become sensitive to noise and overfit.

# Q10. What happens if K is too large?
# The model may become too simple and underfit.
# It may ignore local patterns in the data.

# Q11. What is the difference between KNN and K-Means?
# KNN is a supervised learning algorithm.
# K-Means is an unsupervised clustering algorithm.
# KNN predicts known classes.
# K-Means groups data into clusters.

# Q12. What is classification?
# Classification is the process of predicting categories or class labels.

# Q13. What is the target variable in this practical?
# Purchased is the target variable.

# Q14. What are the input features?
# The input features are Age, EstimatedSalary,
# and Gender if it is present in the dataset.

# Q15. What does 0 represent in the Purchased column?
# 0 represents Not Purchased.

# Q16. What does 1 represent in the Purchased column?
# 1 represents Purchased.

# Q17. What is accuracy?
# Accuracy is the proportion of correct predictions.
# Accuracy = Correct Predictions / Total Predictions.

# Q18. What is error rate?
# Error Rate = 1 - Accuracy.
# It represents the proportion of incorrect predictions.

# Q19. What is precision?
# Precision measures how many predicted positive cases are correct.
# Precision = TP / (TP + FP).

# Q20. What is recall?
# Recall measures how many actual positive cases are correctly identified.
# Recall = TP / (TP + FN).

# Q21. What is a confusion matrix?
# A confusion matrix compares actual and predicted class labels.
# It contains TN, FP, FN, and TP values.

# Q22. What is a false positive?
# A false positive occurs when the model predicts Purchased,
# but the actual class is Not Purchased.

# Q23. What is a false negative?
# A false negative occurs when the model predicts Not Purchased,
# but the actual class is Purchased.

# Q24. What is One-Hot Encoding?
# One-Hot Encoding converts categorical values into numerical columns.

# Q25. Why is User ID removed?
# User ID is an identification number.
# It does not provide useful information for predicting purchases.

# Q26. What is train_test_split?
# It divides the dataset into training and testing data.

# Q27. Why do we use random_state=42?
# It makes the train-test split reproducible.

# Q28. What is stratify=y?
# It maintains the proportion of target classes
# in training and testing datasets.

# Q29. What is Standardization?
# Standardization transforms numerical features
# so that they have approximately zero mean and unit variance.

# Q30. What is the advantage of KNN?
# It is simple to understand and implement.
# It does not require a complex training process.
# It can work well for suitable datasets.

# Q31. What is the disadvantage of KNN?
# Prediction can be slow for large datasets.
# It is sensitive to feature scaling.
# Its performance depends on the choice of K.

# Q32. What are the applications of KNN?
# Recommendation systems.
# Customer purchase prediction.
# Image classification.
# Medical diagnosis.
# Pattern recognition.


# ============================================================
# 30. CONCLUSION
# ============================================================

# In this practical, the K-Nearest Neighbors classification
# algorithm is implemented using the Social Network Ads dataset.

# First, the dataset is loaded and unnecessary columns
# and missing values are removed.

# Input features and the target variable are separated.
# Categorical data is converted into numerical data.

# The dataset is divided into training and testing data.
# StandardScaler is used to standardize the features.

# A KNN model with K=5 is created and trained.
# The model predicts whether users will purchase a product.

# Accuracy, error rate, precision, recall,
# confusion matrix, and classification report
# are used to evaluate model performance.

# Finally, a confusion matrix graph is displayed
# and the model predicts the purchase decision
# for a new user using age, salary, and gender.

# Therefore, KNN can be used to predict customer
# purchasing decisions based on user information.
