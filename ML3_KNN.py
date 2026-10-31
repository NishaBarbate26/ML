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
# EXPLANATION LINE BY LINE
# ============================================================

# import pandas as pd
# Imports Pandas to load and process the dataset.

# import numpy as np
# Imports NumPy for numerical operations.

# import matplotlib.pyplot as plt
# Imports Matplotlib to display the confusion matrix.

# train_test_split
# Divides the dataset into training and testing data.

# StandardScaler
# Standardizes the input features.

# KNeighborsClassifier
# Imports the K-Nearest Neighbours classification algorithm.

# confusion_matrix
# Calculates the confusion matrix.

# accuracy_score
# Calculates the classification accuracy.

# precision_score
# Calculates the precision of the model.

# recall_score
# Calculates the recall of the model.

# classification_report
# Displays precision, recall and F1-score.

# pd.read_csv("Social_Network_Ads.csv")
# Loads the Social Network Ads dataset.

# df.head()
# Displays the first five records.

# df.drop(columns=["User ID"], errors="ignore")
# Removes the User ID column because it is not useful for prediction.

# df.dropna()
# Removes records containing missing values.

# X = df.drop(columns=["Purchased"])
# Selects the input features.

# y = df["Purchased"]
# Selects Purchased as the target variable.
# 0 means Not Purchased and 1 means Purchased.

# pd.get_dummies(...)
# Converts categorical features such as Gender into numeric values.

# train_test_split(...)
# Divides the dataset into 75% training and 25% testing data.
# stratify=y maintains the class proportions.

# scaler = StandardScaler()
# Creates a feature standardization object.

# scaler.fit_transform(X_train)
# Learns scaling values from training data and standardizes it.

# scaler.transform(X_test)
# Standardizes testing data using the training scaling values.

# KNeighborsClassifier(n_neighbors=5)
# Creates a KNN model using five nearest neighbours.

# knn.fit(X_train, y_train)
# Trains the KNN model using the training dataset.

# knn.predict(X_test)
# Predicts whether each testing user will purchase the product.

# confusion_matrix(...)
# Creates a matrix comparing actual and predicted results.
# The matrix contains True Negative, False Positive,
# False Negative and True Positive values.

# accuracy_score(...)
# Calculates the proportion of correct predictions.

# error_rate = 1 - accuracy
# Calculates the proportion of incorrect predictions.

# precision_score(...)
# Calculates the proportion of predicted purchases
# that are actually purchases.

# recall_score(...)
# Calculates the proportion of actual purchases
# correctly identified by the model.

# classification_report(...)
# Displays precision, recall, F1-score and support.

# plt.imshow(...)
# Displays the confusion matrix as a heatmap.

# plt.title(...)
# Sets the graph title.

# plt.xlabel(...)
# Sets the horizontal axis label.

# plt.ylabel(...)
# Sets the vertical axis label.

# plt.xticks(...)
# Displays the predicted class names.

# plt.yticks(...)
# Displays the actual class names.

# plt.text(...)
# Displays the confusion matrix values inside the graph.

# plt.colorbar()
# Displays the color scale.

# plt.show()
# Displays the confusion matrix graph.

# input("Enter Age:")
# Accepts the age of a new user.

# input("Enter Estimated Salary:")
# Accepts the estimated salary of a new user.

# new_user
# Creates a new user record with the required input features.

# new_user.reindex / DataFrame construction
# Ensures the new user has the same feature columns as the training data.

# scaler.transform(new_user)
# Standardizes the new user's measurements.

# knn.predict(new_user_scaled)
# Predicts whether the new user will purchase the product.

# if prediction[0] == 1:
# Checks whether the predicted class is 1.

# print(...)
# Displays the final purchase prediction.