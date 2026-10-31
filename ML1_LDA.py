# AIM: To Apply LDA Algorithm on Iris Dataset and Classify
# Which Species a Given Flower Belongs To.

# Installation commands:
# Windows:
# python -m pip install pandas scikit-learn matplotlib
# python practical2_lda.py
#
# Ubuntu/Linux:
# pip install pandas scikit-learn matplotlib
# python3 -m pip install pandas scikit-learn matplotlib
# python3 practical2_lda.py


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("Iris.csv")

print("Original Iris Dataset:")
print(df.head())

df = df.drop(columns=["Id"], errors="ignore")

X = df.drop(columns=["Species"])
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lda = LinearDiscriminantAnalysis(n_components=2)

X_train_lda = lda.fit_transform(X_train_scaled, y_train)

y_pred = lda.predict(X_test_scaled)

print("\nActual Species:")
print(y_test.values)

print("\nPredicted Species:")
print(y_pred)

print("\nAccuracy:", accuracy_score(y_test, y_pred) * 100, "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))

for species in y.unique():
    plt.scatter(
        X_train_lda[y_train.values == species, 0],
        X_train_lda[y_train.values == species, 1],
        label=species
    )

plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.title("LDA - Iris Dataset")
plt.legend()
plt.grid(True)
plt.show()

print("\nEnter Flower Measurements to Predict Species:")

sepal_length = float(input("Enter Sepal Length (cm): "))
sepal_width = float(input("Enter Sepal Width (cm): "))
petal_length = float(input("Enter Petal Length (cm): "))
petal_width = float(input("Enter Petal Width (cm): "))

new_flower = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=X.columns
)

new_flower_scaled = scaler.transform(new_flower)

prediction = lda.predict(new_flower_scaled)

print("\nPredicted Flower Species:", prediction[0])


# ============================================================
# EXPLANATION LINE BY LINE
# ============================================================

# import pandas as pd
# Imports Pandas to read and process the Iris dataset.

# import matplotlib.pyplot as plt
# Imports Matplotlib to display the LDA graph.

# from sklearn.model_selection import train_test_split
# Imports the function to divide data into training and testing sets.

# from sklearn.preprocessing import StandardScaler
# Imports StandardScaler to standardize the input features.

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Imports the Linear Discriminant Analysis algorithm.

# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Imports functions to evaluate the classification model.

# df = pd.read_csv("Iris.csv")
# Loads the Iris dataset from the local CSV file.

# print(df.head())
# Displays the first five rows of the dataset.

# df = df.drop(columns=["Id"], errors="ignore")
# Removes the unnecessary Id column if it exists.
# errors="ignore" prevents an error if the column is absent.

# X = df.drop(columns=["Species"])
# Selects flower measurements as input features.

# y = df["Species"]
# Stores the flower species as the target variable.

# train_test_split(...)
# Divides the dataset into training and testing data.
# test_size=0.2 means 20% testing data and 80% training data.
# random_state=42 ensures reproducible results.
# stratify=y maintains the class proportions in both sets.

# scaler = StandardScaler()
# Creates a standardization object.

# X_train_scaled = scaler.fit_transform(X_train)
# Learns scaling values from training data and standardizes it.

# X_test_scaled = scaler.transform(X_test)
# Standardizes testing data using the training data's scaling values.

# lda = LinearDiscriminantAnalysis(n_components=2)
# Creates an LDA model with two discriminant components.
# Two components are suitable for the three Iris species.

# X_train_lda = lda.fit_transform(X_train_scaled, y_train)
# Trains the LDA model and transforms training data into two components.

# y_pred = lda.predict(X_test_scaled)
# Predicts the species of flowers in the testing dataset.

# print(y_test.values)
# Displays the actual flower species.

# print(y_pred)
# Displays the predicted flower species.

# accuracy_score(y_test, y_pred) * 100
# Calculates the classification accuracy as a percentage.

# confusion_matrix(y_test, y_pred)
# Displays the confusion matrix to compare actual and predicted classes.

# classification_report(y_test, y_pred)
# Displays precision, recall, F1-score and support for each species.

# plt.figure(figsize=(8, 6))
# Creates a figure for the LDA graph.

# for species in y.unique():
# Iterates through all unique flower species.

# plt.scatter(...)
# Plots the two LDA components for each flower species.

# y_train.values == species
# Selects training samples belonging to the current species.

# plt.xlabel(...)
# Sets the horizontal axis label.

# plt.ylabel(...)
# Sets the vertical axis label.

# plt.title("LDA - Iris Dataset")
# Sets the graph title.

# plt.legend()
# Displays the species names in the graph legend.

# plt.grid(True)
# Displays grid lines on the graph.

# plt.show()
# Displays the LDA visualization.

# sepal_length = float(input(...))
# Accepts the sepal length entered by the user.

# sepal_width = float(input(...))
# Accepts the sepal width entered by the user.

# petal_length = float(input(...))
# Accepts the petal length entered by the user.

# petal_width = float(input(...))
# Accepts the petal width entered by the user.

# new_flower = pd.DataFrame(...)
# Creates a DataFrame containing the new flower measurements.
# The column names match the training dataset.

# new_flower_scaled = scaler.transform(new_flower)
# Standardizes the new flower measurements using the training scaler.

# prediction = lda.predict(new_flower_scaled)
# Predicts the species of the new flower using the trained LDA model.

# print(prediction[0])
# Displays the predicted flower species.