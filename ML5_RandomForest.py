# AIM: Implement Random Forest Classifier to Predict the Safety of a Car.

# Installation commands:
# Windows:
# python -m pip install pandas matplotlib scikit-learn
# python practical_random_forest.py
#
# Ubuntu/Linux:
# pip install pandas matplotlib scikit-learn
# python3 -m pip install pandas matplotlib scikit-learn
# python3 practical_random_forest.py


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load dataset

df = pd.read_csv(
    "car_evaluation.csv",
    header=None,
    names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
)

print("Car Evaluation Dataset:")
print(df.head())

# Display dataset information

print("\nDataset Information:")
print(df.info())

# Remove missing values

df = df.dropna()

# Separate input and output

X = df.drop("class", axis=1)

y = df["class"]

# Convert categorical input columns into numeric values

X = pd.get_dummies(X, dtype=int)

# Encode target column

encoder = LabelEncoder()

y = encoder.fit_transform(y)

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Random Forest model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model

model.fit(X_train, y_train)

# Predict results

y_pred = model.predict(X_test)

# Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Display confusion matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display classification report

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        zero_division=0
    )
)

# Display feature importance

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# Plot feature importance

plt.figure(figsize=(9, 5))

importance.plot(kind="bar")

plt.title("Random Forest Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# Predict safety for a new car

new_car = pd.DataFrame(
    {
        "buying": ["low"],
        "maint": ["low"],
        "doors": ["4"],
        "persons": ["4"],
        "lug_boot": ["big"],
        "safety": ["high"]
    }
)

new_car = pd.get_dummies(new_car, dtype=int)

new_car = new_car.reindex(
    columns=X.columns,
    fill_value=0
)

prediction = model.predict(new_car)

print("\nPredicted Car Evaluation:")

print(encoder.inverse_transform(prediction)[0])


# AIM:
# Implement Random Forest Classifier to Predict the Safety of a Car.

# ============================================================
# 1. INSTALLATION AND EXECUTION COMMANDS
# ============================================================

# Windows:
# python -m pip install pandas matplotlib scikit-learn
# python practical_random_forest.py

# Ubuntu/Linux:
# pip install pandas matplotlib scikit-learn
# python3 practical_random_forest.py


# ============================================================
# 2. IMPORTING LIBRARIES
# ============================================================

# import pandas as pd
# Imports the Pandas library.
# Pandas is used to read, store, and process tabular data.
# pd is the short name used for Pandas.

# import matplotlib.pyplot as plt
# Imports the Pyplot module from Matplotlib.
# It is used to create graphs and charts.
# plt is the short name used for pyplot.

# from sklearn.model_selection import train_test_split
# Imports the train_test_split function.
# It divides the dataset into training data and testing data.

# from sklearn.preprocessing import LabelEncoder
# Imports LabelEncoder.
# It converts categorical target values into numerical values.

# from sklearn.ensemble import RandomForestClassifier
# Imports the Random Forest classification algorithm.
# Random Forest combines multiple decision trees to make predictions.

# from sklearn.metrics import accuracy_score
# Imports accuracy_score.
# It calculates how many predictions are correct.

# from sklearn.metrics import confusion_matrix
# Imports confusion_matrix.
# It displays correct and incorrect predictions in a table.

# from sklearn.metrics import classification_report
# Imports classification_report.
# It displays precision, recall, F1-score, and support.


# ============================================================
# 3. LOADING THE DATASET
# ============================================================

# df = pd.read_csv(
#     "car_evaluation.csv",
#     header=None,
#     names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
# )

# pd.read_csv() reads the CSV dataset.
# car_evaluation.csv is the local dataset filename.
# header=None means the dataset does not contain column headings.
# names assigns meaningful names to all seven columns.

# buying represents the buying price of the car.
# maint represents the maintenance cost of the car.
# doors represents the number of doors.
# persons represents the passenger capacity.
# lug_boot represents the luggage boot size.
# safety represents the safety level of the car.
# class represents the final car evaluation category.

# df stores the complete dataset in a DataFrame.
# A DataFrame is a table containing rows and columns.

# print("Car Evaluation Dataset:")
# Displays the dataset heading.

# print(df.head())
# Displays the first five rows of the dataset.
# head() helps us understand the structure of the data.


# ============================================================
# 4. DISPLAYING DATASET INFORMATION
# ============================================================

# print("\nDataset Information:")
# Displays the dataset information heading.
# \n moves the output to a new line.

# print(df.info())
# Displays information about the dataset.
# It shows column names, data types, and non-null values.
# It also helps identify missing values.


# ============================================================
# 5. REMOVING MISSING VALUES
# ============================================================

# df = df.dropna()
# dropna() removes rows containing missing values.
# Missing values are empty or unavailable data.
# This helps prevent problems during model training.


# ============================================================
# 6. SEPARATING INPUT AND OUTPUT
# ============================================================

# X = df.drop("class", axis=1)
# X contains the input features used for prediction.
# drop("class") removes the target column from the input data.
# axis=1 means that a column is being removed.
# The remaining six columns are used as input features.

# y = df["class"]
# y contains the output or target column.
# It stores the final car evaluation category.
# The possible categories are unacc, acc, good, and vgood.


# ============================================================
# 7. CONVERTING CATEGORICAL INPUT DATA INTO NUMBERS
# ============================================================

# X = pd.get_dummies(X, dtype=int)
# The input columns contain text values such as low, high, and big.
# Machine learning algorithms require numerical input.
# get_dummies() converts these categories into separate columns.
# This process is called One-Hot Encoding.
# dtype=int stores the encoded values as integers.
# The new columns contain 0 or 1.
# 1 means that the category is present.
# 0 means that the category is not present.


# ============================================================
# 8. ENCODING THE TARGET COLUMN
# ============================================================

# encoder = LabelEncoder()
# Creates a LabelEncoder object.
# It converts target categories into numerical labels.

# y = encoder.fit_transform(y)
# fit_transform() learns the target categories and converts them into numbers.
# Each category receives a numerical label.
# For example, the categories may be encoded as:
# acc = 0
# good = 1
# unacc = 2
# vgood = 3
# The exact numerical labels depend on alphabetical ordering.
# The encoder remembers the original category names.


# ============================================================
# 9. SPLITTING THE DATASET
# ============================================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# train_test_split() divides the dataset into training and testing parts.

# X_train contains input features used for training.
# X_test contains input features used for testing.
# y_train contains actual output values for training.
# y_test contains actual output values for testing.

# test_size=0.2 means 20% of the dataset is used for testing.
# The remaining 80% is used for training.

# random_state=42 ensures that the same random split is obtained every time.
# It makes the experiment reproducible.

# stratify=y maintains the proportion of each target category.
# It helps ensure that all categories are represented in both datasets.


# ============================================================
# 10. CREATING THE RANDOM FOREST MODEL
# ============================================================

# model = RandomForestClassifier(
#     n_estimators=100,
#     random_state=42
# )

# RandomForestClassifier() creates the Random Forest model.

# Random Forest is an ensemble learning algorithm.
# Ensemble learning means combining multiple machine learning models.
# Random Forest combines many decision trees.

# n_estimators=100 means the model contains 100 decision trees.
# Each tree makes its own prediction.
# The final prediction is selected using majority voting.

# random_state=42 makes the model results reproducible.


# ============================================================
# 11. TRAINING THE MODEL
# ============================================================

# model.fit(X_train, y_train)
# fit() trains the Random Forest model.
# It learns patterns from the training data.
# It uses the input features and their correct output categories.
# Each decision tree learns different patterns from the dataset.


# ============================================================
# 12. PREDICTING THE RESULTS
# ============================================================

# y_pred = model.predict(X_test)
# predict() uses the trained model to predict car evaluation categories.
# X_test contains the testing input features.
# y_pred stores the predicted categories.
# These predictions are compared with the actual values in y_test.


# ============================================================
# 13. CALCULATING ACCURACY
# ============================================================

# accuracy = accuracy_score(y_test, y_pred)
# accuracy_score() compares actual and predicted values.
# It calculates the percentage of correct predictions.

# print("\nAccuracy:", accuracy)
# Displays the accuracy of the Random Forest model.
# Accuracy ranges from 0 to 1.
# An accuracy closer to 1 indicates better overall correctness.
# Multiply accuracy by 100 to express it as a percentage.


# ============================================================
# 14. DISPLAYING THE CONFUSION MATRIX
# ============================================================

# cm = confusion_matrix(y_test, y_pred)
# Creates a confusion matrix.
# It compares actual categories with predicted categories.

# print("\nConfusion Matrix:")
# Displays the confusion matrix heading.

# print(cm)
# Displays the confusion matrix as a numerical table.

# Rows represent actual categories.
# Columns represent predicted categories.
# Diagonal values represent correct predictions.
# Values outside the diagonal represent incorrect predictions.

# A higher number on the diagonal generally indicates better performance.


# ============================================================
# 15. DISPLAYING THE CLASSIFICATION REPORT
# ============================================================

# print("\nClassification Report:")
# Displays the classification report heading.

# print(
#     classification_report(
#         y_test,
#         y_pred,
#         target_names=encoder.classes_,
#         zero_division=0
#     )
# )

# classification_report() displays the performance of each category.

# y_test contains the actual target values.
# y_pred contains the predicted target values.

# target_names=encoder.classes_ displays the original category names
# instead of their numerical labels.

# precision measures how many positive predictions are correct.
# High precision means fewer false positive predictions.

# recall measures how many actual positive cases are correctly identified.
# High recall means fewer false negative predictions.

# F1-score is the harmonic mean of precision and recall.
# It balances precision and recall.

# support represents the number of actual samples in each category.

# zero_division=0 avoids division-by-zero warnings.
# It displays 0 when a metric cannot be calculated.


# ============================================================
# 16. DISPLAYING FEATURE IMPORTANCE
# ============================================================

# importance = pd.Series(
#     model.feature_importances_,
#     index=X.columns
# ).sort_values(ascending=False)

# model.feature_importances_ calculates the importance of each input feature.
# It indicates how much each feature contributes to the decision trees.

# pd.Series() creates a Pandas Series.
# A Series is a one-dimensional labelled data structure.

# index=X.columns assigns the feature names to their importance values.

# sort_values(ascending=False) sorts the values from highest to lowest.
# The most important feature appears first.

# print("\nFeature Importance:")
# Displays the feature importance heading.

# print(importance)
# Displays the importance score of every encoded feature.

# Higher importance means that the feature contributed more
# to reducing impurity across the Random Forest trees.
# Feature importance does not prove that a feature causes the result.


# ============================================================
# 17. PLOTTING FEATURE IMPORTANCE
# ============================================================

# plt.figure(figsize=(9, 5))
# Creates a graph with a width of 9 inches and height of 5 inches.

# importance.plot(kind="bar")
# Creates a bar graph of feature importance.
# Each bar represents one encoded input feature.
# Taller bars represent higher importance.

# plt.title("Random Forest Feature Importance")
# Sets the title of the graph.

# plt.xlabel("Features")
# Sets the horizontal axis label.

# plt.ylabel("Importance")
# Sets the vertical axis label.

# plt.xticks(rotation=45)
# Rotates feature names by 45 degrees.
# This makes long feature names easier to read.

# plt.tight_layout()
# Adjusts the graph spacing to prevent labels from being cut off.

# plt.show()
# Displays the feature importance graph.


# ============================================================
# 18. PREDICTING THE EVALUATION OF A NEW CAR
# ============================================================

# new_car = pd.DataFrame(
#     {
#         "buying": ["low"],
#         "maint": ["low"],
#         "doors": ["4"],
#         "persons": ["4"],
#         "lug_boot": ["big"],
#         "safety": ["high"]
#     }
# )

# Creates a new DataFrame containing the details of a car.
# This car has low buying price.
# It has low maintenance cost.
# It has four doors.
# It can accommodate four persons.
# It has a big luggage boot.
# It has high safety.

# This new record is given to the trained model for prediction.


# ============================================================
# 19. ENCODING THE NEW CAR DATA
# ============================================================

# new_car = pd.get_dummies(new_car, dtype=int)
# Converts the new car's categorical values into numerical columns.
# It uses the same One-Hot Encoding method as the training data.

# new_car = new_car.reindex(
#     columns=X.columns,
#     fill_value=0
# )

# reindex() arranges the new car columns in the same order
# as the columns used during model training.

# columns=X.columns specifies the exact training feature columns.
# fill_value=0 fills any missing encoded columns with zero.

# This step is important because the model expects the same
# input features during training and prediction.


# ============================================================
# 20. MAKING THE FINAL PREDICTION
# ============================================================

# prediction = model.predict(new_car)
# Uses the trained Random Forest model to predict the new car's category.
# The output is a numerical category label.

# print("\nPredicted Car Evaluation:")
# Displays the prediction heading.

# print(encoder.inverse_transform(prediction)[0])
# inverse_transform() converts the numerical prediction
# back into the original category name.
# [0] selects the prediction for the first car.
# The result may be unacc, acc, good, or vgood.


# ============================================================
# 21. UNDERSTANDING THE COMPLETE OUTPUT
# ============================================================

# Car Evaluation Dataset:
# Displays the first five rows of the dataset.

# Dataset Information:
# Displays the column names, data types, and number of records.

# Accuracy:
# Shows the proportion of correctly predicted car categories.

# Confusion Matrix:
# Shows the number of correct and incorrect predictions
# for each car evaluation category.

# Classification Report:
# Shows precision, recall, F1-score, and support for each category.

# Feature Importance:
# Shows which encoded input features contributed most to predictions.

# Feature Importance Graph:
# Visually compares the importance of all input features.

# Predicted Car Evaluation:
# Displays the predicted evaluation category of the new car.


# ============================================================
# 22. IMPORTANT CONCEPTS FOR VIVA
# ============================================================

# Q1. What is Random Forest?
# Random Forest is a supervised machine learning algorithm.
# It is used for classification and regression.
# It combines multiple decision trees to improve prediction.

# Q2. What is a Decision Tree?
# A Decision Tree is a tree-shaped model.
# It makes decisions using conditions on input features.
# It contains root nodes, decision nodes, branches, and leaf nodes.

# Q3. What is Ensemble Learning?
# Ensemble Learning combines multiple machine learning models.
# It generally improves prediction performance and stability.

# Q4. Why is it called Random Forest?
# It is called Random Forest because it creates many decision trees.
# Each tree is trained using random samples and random feature selections.

# Q5. What is n_estimators?
# It specifies the number of decision trees in the Random Forest.
# In this practical, n_estimators=100 means 100 trees.

# Q6. What is classification?
# Classification is a supervised learning task.
# It predicts a category or class for given input data.

# Q7. What is One-Hot Encoding?
# One-Hot Encoding converts categorical values into binary columns.
# Each category is represented using 0 and 1.

# Q8. What is Label Encoding?
# Label Encoding converts categorical labels into numerical values.
# It is used here to encode the target column.

# Q9. Why do we split the dataset?
# We split the dataset to train and test the model separately.
# Training data teaches the model.
# Testing data checks how well the model performs on unseen records.

# Q10. What is accuracy?
# Accuracy is the ratio of correct predictions to total predictions.
# Accuracy = Correct Predictions / Total Predictions.

# Q11. What is a confusion matrix?
# A confusion matrix compares actual and predicted classes.
# It helps identify correct and incorrect predictions.

# Q12. What is precision?
# Precision tells us how many predicted positive cases are actually positive.

# Q13. What is recall?
# Recall tells us how many actual positive cases are correctly predicted.

# Q14. What is F1-score?
# F1-score is the harmonic mean of precision and recall.

# Q15. What is feature importance?
# Feature importance indicates how much each input feature contributes
# to the predictions made by the Random Forest model.

# Q16. What is random_state?
# random_state makes random operations reproducible.
# It helps obtain consistent results when the program is executed again.

# Q17. What is stratify=y?
# It maintains the proportion of target categories
# in both training and testing datasets.

# Q18. What are the applications of Random Forest?
# Car safety prediction.
# Medical diagnosis.
# Fraud detection.
# Customer classification.
# Credit risk prediction.
# Disease prediction.

# Q19. What are the advantages of Random Forest?
# It combines multiple decision trees.
# It can handle classification and regression problems.
# It can work with complex datasets.
# It generally reduces overfitting compared with a single decision tree.

# Q20. What is the limitation of Random Forest?
# It can require more memory and processing time.
# It is less easy to interpret than a single decision tree.
# Its performance depends on the dataset and model settings.


# ============================================================
# 23. CONCLUSION
# ============================================================

# In this practical, the Random Forest Classifier is implemented
# using the Car Evaluation dataset.

# The categorical input features are converted into numerical values.
# The dataset is divided into training and testing data.
# The Random Forest model is trained using 100 decision trees.
# Its performance is evaluated using accuracy, confusion matrix,
# precision, recall, and F1-score.
# Feature importance is displayed using a bar graph.
# Finally, the trained model predicts the evaluation category
# of a new car based on its features.
