# AIM: Apply Voting Classifier, AdaBoost, Gradient Boosting
# and XGBoost on Iris Dataset and Compare Their Performance.

# Installation commands:
# Windows:
# python -m pip install pandas matplotlib scikit-learn xgboost
# python practical_ensemble_iris.py
#
# Ubuntu/Linux:
# pip install pandas matplotlib scikit-learn xgboost
# python3 -m pip install pandas matplotlib scikit-learn xgboost
# python3 practical_ensemble_iris.py


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

# Load dataset

df = pd.read_csv("Iris.csv")

print("Iris Dataset:")
print(df.head())

# Remove unnecessary Id column

df = df.drop(columns=["Id"], errors="ignore")

# Remove missing values

df = df.dropna()

# Separate input and output

X = df.drop("Species", axis=1)

y = df["Species"]

# Encode target values

y = y.astype("category").cat.codes

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardize input features

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Create base models for Voting Classifier

model1 = DecisionTreeClassifier(random_state=42)

model2 = RandomForestClassifier(
    n_estimators=50,
    random_state=42
)

model3 = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

# Hard Voting Classifier

hard_voting = VotingClassifier(
    estimators=[
        ("dt", model1),
        ("rf", model2),
        ("dt2", model3)
    ],
    voting="hard"
)

# Soft Voting Classifier

soft_voting = VotingClassifier(
    estimators=[
        ("dt", model1),
        ("rf", model2),
        ("dt2", model3)
    ],
    voting="soft"
)

# AdaBoost Classifier

adaboost = AdaBoostClassifier(
    n_estimators=50,
    random_state=42
)

# Gradient Boosting Classifier

gbm = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)

# XGBoost Classifier

xgboost = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42
)

# Store all models

models = {
    "Hard Voting": hard_voting,
    "Soft Voting": soft_voting,
    "AdaBoost": adaboost,
    "Gradient Boosting": gbm,
    "XGBoost": xgboost
}

# Store performance results

results = []

# Train and evaluate each model

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    results.append(
        [
            name,
            accuracy,
            precision,
            recall,
            f1
        ]
    )

    print("\n", name)

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

# Create comparison table

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\nModel Performance Comparison:")

print(results_df.to_string(index=False))

# Plot model accuracy

plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.title("Ensemble Models Accuracy Comparison")

plt.xlabel("Models")

plt.ylabel("Accuracy")

plt.xticks(rotation=30)

plt.ylim(0, 1.1)

plt.tight_layout()

plt.show()

# Find the best model

best_model = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print("\nBest Performing Model:")

print(best_model)


# ============================================================
# AIM:
# Apply Voting Classifier, AdaBoost, Gradient Boosting
# and XGBoost on Iris Dataset and Compare Their Performance.
# ============================================================


# -------------------- INSTALLATION COMMANDS -----------------

# Windows:
# python -m pip install pandas matplotlib scikit-learn xgboost
# python practical_ensemble_iris.py

# Ubuntu/Linux:
# pip install pandas matplotlib scikit-learn xgboost
# python3 -m pip install pandas matplotlib scikit-learn xgboost


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

# import pandas as pd
# Imports the Pandas library.
# It is used to load, store and process tabular data.
# pd is the short name used for Pandas.

# import matplotlib.pyplot as plt
# Imports the Pyplot module from Matplotlib.
# It is used to create graphs and visualize model performance.
# plt is the short name used for pyplot.

# from sklearn.model_selection import train_test_split
# Imports train_test_split.
# It divides the dataset into training and testing data.

# from sklearn.preprocessing import StandardScaler
# Imports StandardScaler.
# It standardizes numerical features to bring them
# to a comparable scale.

# from sklearn.ensemble import VotingClassifier
# Imports the Voting Classifier algorithm.
# It combines predictions from multiple machine learning models
# to produce a final prediction.

# from sklearn.ensemble import RandomForestClassifier
# Imports the Random Forest classification algorithm.
# It combines multiple decision trees to make predictions.

# from sklearn.ensemble import AdaBoostClassifier
# Imports the AdaBoost classification algorithm.
# AdaBoost gives more attention to incorrectly classified
# training samples during successive training rounds.

# from sklearn.ensemble import GradientBoostingClassifier
# Imports the Gradient Boosting classification algorithm.
# It builds models sequentially to reduce prediction errors.

# from sklearn.tree import DecisionTreeClassifier
# Imports the Decision Tree classification algorithm.
# A decision tree makes predictions using a tree-like
# structure of questions and decisions.

# from sklearn.metrics import accuracy_score
# Imports accuracy_score.
# It calculates the proportion of correctly predicted samples.

# from sklearn.metrics import precision_score
# Imports precision_score.
# It measures how many predicted positive cases are actually correct.

# from sklearn.metrics import recall_score
# Imports recall_score.
# It measures how many actual positive cases are correctly identified.

# from sklearn.metrics import f1_score
# Imports f1_score.
# It calculates the harmonic mean of precision and recall.

# from sklearn.metrics import classification_report
# Imports classification_report.
# It displays precision, recall, F1-score and support
# for each class.

# from xgboost import XGBClassifier
# Imports the XGBoost classification algorithm.
# XGBoost is an optimized gradient boosting algorithm
# used for classification and regression tasks.


# ============================================================
# 2. LOAD THE IRIS DATASET
# ============================================================

# df = pd.read_csv("Iris.csv")
# Loads the Iris dataset from the local CSV file.
# The dataset is stored in a DataFrame named df.
# A DataFrame is a table containing rows and columns.
# Iris.csv should be present in the same folder as the Python file.

# print("Iris Dataset:")
# Displays the heading before printing the dataset.

# print(df.head())
# Displays the first five rows of the Iris dataset.
# head() displays the first five records by default.

# IRIS DATASET INFORMATION:
# The Iris dataset contains 150 flower records.
# Each record represents one iris flower.
# It contains three different flower species.

# The three species are:
# 1. Iris-setosa
# 2. Iris-versicolor
# 3. Iris-virginica

# The dataset contains four main numerical features:

# SepalLengthCm
# Represents the length of the sepal in centimeters.

# SepalWidthCm
# Represents the width of the sepal in centimeters.

# PetalLengthCm
# Represents the length of the petal in centimeters.

# PetalWidthCm
# Represents the width of the petal in centimeters.

# Species
# Represents the actual flower category or class.

# Id
# Represents the identification number of each flower.


# ============================================================
# 3. REMOVE UNNECESSARY ID COLUMN
# ============================================================

# df = df.drop(columns=["Id"], errors="ignore")
# Removes the Id column from the dataset.
# Id is only an identification number.
# It does not provide useful information about flower characteristics.

# columns=["Id"]
# Specifies the column that needs to be removed.

# errors="ignore"
# Prevents an error if the Id column does not exist.
# The program continues even if the column is missing.


# ============================================================
# 4. REMOVE MISSING VALUES
# ============================================================

# df = df.dropna()
# Removes all rows containing missing values.
# Missing values are empty or unavailable data entries.

# Why do we remove missing values?
# Machine learning algorithms require properly prepared input data.
# Missing values may cause errors during model training.
# Removing incomplete rows helps prepare the dataset.

# Important:
# dropna() removes the entire row if any column contains a missing value.
# Therefore, the number of records may decrease if missing values exist.


# ============================================================
# 5. SEPARATE INPUT FEATURES AND OUTPUT TARGET
# ============================================================

# X = df.drop("Species", axis=1)
# Selects all columns except Species as input features.
# X contains the four numerical flower measurements.
# These measurements are used by the machine learning models.

# axis=1
# Indicates that a column is being removed.
# axis=0 would indicate removing a row.

# y = df["Species"]
# Selects the Species column as the target variable.
# It contains the actual flower category.
# The model learns to predict this category using X.

# What is X?
# X represents input features or independent variables.

# What is y?
# y represents the target variable or dependent variable.

# Example:
# Input: Sepal length, sepal width, petal length and petal width.
# Output: Setosa, Versicolor or Virginica.


# ============================================================
# 6. ENCODE TARGET VALUES
# ============================================================

# y = y.astype("category").cat.codes
# Converts the flower species names into numerical class labels.
# Machine learning models generally work with numerical target labels.

# astype("category")
# Converts the Species column into a categorical data type.

# cat.codes
# Assigns a numerical code to every unique category.

# The three species are represented using numerical values:
# 0
# 1
# 2

# Example:
# Iris-setosa may become 0.
# Iris-versicolor may become 1.
# Iris-virginica may become 2.

# Important:
# The numerical codes represent categories.
# They do not indicate that one species is greater than another.

# Why is encoding required?
# It converts text labels into numerical values
# that can be used as target classes by the models.


# ============================================================
# 7. SPLIT DATASET INTO TRAINING AND TESTING DATA
# ============================================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )
# Divides the dataset into training and testing datasets.

# X
# Contains the input flower measurements.

# y
# Contains the actual numerical species labels.

# X_train
# Contains the input features used for model training.

# X_test
# Contains the input features used for model testing.

# y_train
# Contains the actual output labels used during training.

# y_test
# Contains the actual output labels used for testing.

# test_size=0.2
# Means that 20% of the dataset is used for testing.
# The remaining 80% is used for training.

# For 150 records:
# Training data contains 120 records.
# Testing data contains 30 records.

# random_state=42
# Controls the random splitting process.
# It helps produce the same train-test split when the program
# is executed again.

# stratify=y
# Maintains the proportion of each species
# in both training and testing datasets.

# Why do we split the dataset?
# Training data teaches the model to recognize patterns.
# Testing data checks how well the trained model
# predicts previously unseen samples.


# ============================================================
# 8. STANDARDIZE INPUT FEATURES
# ============================================================

# scaler = StandardScaler()
# Creates a StandardScaler object.
# It is used to standardize the numerical input features.

# X_train = scaler.fit_transform(X_train)
# Calculates the mean and standard deviation using training data.
# Then standardizes the training features.

# X_test = scaler.transform(X_test)
# Standardizes the testing features using the same
# mean and standard deviation learned from training data.

# Why do we use fit_transform() for training data?
# Because the scaler must first learn the scaling values
# from the training dataset and then transform it.

# Why do we use only transform() for testing data?
# Because testing data must use the same scaling values
# learned from the training data.

# Why is this important?
# It prevents information from the testing dataset
# from influencing the training process.

# Standardization formula:
# Z = (X - Mean) / Standard Deviation

# After standardization:
# Features have approximately zero mean
# and unit standard deviation.

# Why standardize?
# Features may have different numerical ranges.
# Standardization makes their scales comparable.

# Important:
# Standardization is especially useful for algorithms
# that are sensitive to feature scales.
# Decision trees and tree-based ensemble models generally
# do not require standardization, but this practical
# applies it to all models for a common preprocessing workflow.


# ============================================================
# 9. CREATE BASE MODELS FOR VOTING CLASSIFIER
# ============================================================

# What is a base model?
# A base model is an individual machine learning algorithm
# used as a member of an ensemble.

# An ensemble combines multiple models to make predictions.
# The idea is that combining different models may produce
# better and more reliable predictions than using one model alone.


# -------------------- MODEL 1: DECISION TREE -----------------

# model1 = DecisionTreeClassifier(random_state=42)
# Creates the first Decision Tree classification model.

# Decision Tree:
# A Decision Tree is a supervised machine learning algorithm.
# It makes predictions using a tree-like structure.

# It contains:
# Root node: The starting point of the tree.
# Decision nodes: Nodes that check feature conditions.
# Branches: Paths based on the decisions.
# Leaf nodes: Final predicted classes.

# random_state=42
# Helps make the random operations repeatable.


# -------------------- MODEL 2: RANDOM FOREST ----------------

# model2 = RandomForestClassifier(
#     n_estimators=50,
#     random_state=42
# )
# Creates a Random Forest classification model.

# Random Forest:
# Random Forest is an ensemble learning algorithm.
# It combines multiple decision trees.
# Each tree makes a prediction.
# The final prediction is decided using voting.

# n_estimators=50
# Specifies that the Random Forest contains 50 decision trees.

# random_state=42
# Makes the random operations repeatable.

# Why use Random Forest?
# A group of decision trees can produce more stable predictions
# than relying on only one decision tree.


# -------------------- MODEL 3: DECISION TREE -----------------

# model3 = DecisionTreeClassifier(
#     max_depth=3,
#     random_state=42
# )
# Creates another Decision Tree model with a limited depth.

# max_depth=3
# Limits the maximum depth of the decision tree to 3 levels.
# It controls how complex the tree can become.

# Why use max_depth?
# Limiting tree depth helps control model complexity.
# It can reduce overfitting.

# Overfitting:
# Occurs when a model learns the training data too closely
# and performs poorly on new, unseen data.


# ============================================================
# 10. HARD VOTING CLASSIFIER
# ============================================================

# hard_voting = VotingClassifier(
#     estimators=[
#         ("dt", model1),
#         ("rf", model2),
#         ("dt2", model3)
#     ],
#     voting="hard"
# )
# Creates a Hard Voting Classifier.
# It combines the predictions of three base models:
# 1. Decision Tree.
# 2. Random Forest.
# 3. Decision Tree with maximum depth 3.

# estimators
# Contains the list of models that will participate in voting.

# ("dt", model1)
# Gives the first Decision Tree the name dt.

# ("rf", model2)
# Gives the Random Forest model the name rf.

# ("dt2", model3)
# Gives the second Decision Tree the name dt2.

# voting="hard"
# Selects the final class based on majority voting.

# How does Hard Voting work?
# Step 1: Each model predicts a class.
# Step 2: The predictions are collected.
# Step 3: The class receiving the most votes is selected.
# Step 4: The selected class becomes the final prediction.

# Example:
# Decision Tree predicts: Class 0.
# Random Forest predicts: Class 1.
# Second Decision Tree predicts: Class 1.

# Votes:
# Class 0 = 1 vote.
# Class 1 = 2 votes.

# Final prediction = Class 1.

# Important:
# Hard Voting uses predicted class labels.
# It does not use prediction probabilities.


# ============================================================
# 11. SOFT VOTING CLASSIFIER
# ============================================================

# soft_voting = VotingClassifier(
#     estimators=[
#         ("dt", model1),
#         ("rf", model2),
#         ("dt2", model3)
#     ],
#     voting="soft"
# )
# Creates a Soft Voting Classifier.
# It combines the class probabilities predicted by the models.

# voting="soft"
# Uses predicted probabilities instead of only class labels.

# How does Soft Voting work?
# Step 1: Each model predicts the probability of every class.
# Step 2: The probabilities from all models are averaged.
# Step 3: The class with the highest average probability
#         is selected as the final prediction.

# Example:
# Suppose three models predict the following probabilities:

# Model 1:
# Class 0 = 0.70
# Class 1 = 0.20
# Class 2 = 0.10

# Model 2:
# Class 0 = 0.30
# Class 1 = 0.60
# Class 2 = 0.10

# Model 3:
# Class 0 = 0.20
# Class 1 = 0.70
# Class 2 = 0.10

# Average probabilities:
# Class 0 = 0.40
# Class 1 = 0.50
# Class 2 = 0.10

# Final prediction = Class 1.

# Important:
# Soft Voting requires the base models to support
# probability prediction.
# The Decision Tree and Random Forest models used here
# support probability prediction.

# Difference between Hard Voting and Soft Voting:
# Hard Voting uses the majority of predicted class labels.
# Soft Voting uses the average predicted class probabilities.


# ============================================================
# 12. ADABOOST CLASSIFIER
# ============================================================

# adaboost = AdaBoostClassifier(
#     n_estimators=50,
#     random_state=42
# )
# Creates an AdaBoost classification model.

# What is AdaBoost?
# AdaBoost stands for Adaptive Boosting.
# It is an ensemble learning algorithm.
# It combines multiple weak learners to create
# a stronger classification model.

# What is a weak learner?
# A weak learner is a simple model that performs
# slightly better than random guessing.
# A shallow decision tree is commonly used as a weak learner.

# How does AdaBoost work?
# Step 1:
# Assign weights to the training samples.

# Step 2:
# Train a weak learner using the training samples.

# Step 3:
# Identify incorrectly classified samples.

# Step 4:
# Give more attention to incorrectly classified samples
# in the next training round.

# Step 5:
# Train another weak learner using the updated sample weights.

# Step 6:
# Repeat the process for multiple learners.

# Step 7:
# Combine the predictions of all weak learners
# to produce the final prediction.

# Why is it called Adaptive Boosting?
# Because it adapts by giving more importance
# to samples that were incorrectly classified.

# n_estimators=50
# Specifies that the model can use up to 50 boosting estimators.
# Each estimator contributes to the final prediction.

# random_state=42
# Helps make the model's random operations repeatable.

# Important:
# AdaBoost focuses on correcting the mistakes made
# by earlier weak learners.


# ============================================================
# 13. GRADIENT BOOSTING CLASSIFIER
# ============================================================

# gbm = GradientBoostingClassifier(
#     n_estimators=100,
#     random_state=42
# )
# Creates a Gradient Boosting classification model.

# What is Gradient Boosting?
# Gradient Boosting is an ensemble learning technique.
# It builds multiple weak learners sequentially.
# Each new learner tries to improve the predictions
# made by the previous learners.

# It commonly uses decision trees as weak learners.

# How does Gradient Boosting work?
# Step 1:
# Create an initial prediction.

# Step 2:
# Calculate the errors or residual information
# from the current model.

# Step 3:
# Train a new decision tree to improve the current predictions.

# Step 4:
# Add the new tree's contribution to the existing model.

# Step 5:
# Repeat the process to gradually improve performance.

# Step 6:
# Combine all the trees to produce the final prediction.

# What is a residual?
# A residual is the difference between the actual value
# and the predicted value in regression problems.
# In classification, boosting uses gradients of a loss function
# to determine how the model should improve.

# n_estimators=100
# Specifies the maximum number of boosting stages.
# Here, the model can build 100 sequential boosting stages.

# random_state=42
# Helps make the model's random operations repeatable.

# Important:
# Gradient Boosting builds models sequentially.
# Each new model tries to correct the remaining errors
# of the combined model.


# ============================================================
# 14. XGBOOST CLASSIFIER
# ============================================================

# xgboost = XGBClassifier(
#     n_estimators=100,
#     max_depth=3,
#     learning_rate=0.1,
#     objective="multi:softprob",
#     eval_metric="mlogloss",
#     random_state=42
# )
# Creates an XGBoost classification model.

# What is XGBoost?
# XGBoost stands for Extreme Gradient Boosting.
# It is an optimized and efficient implementation
# of the gradient boosting technique.

# It builds decision trees sequentially.
# Each new tree improves the combined model's predictions.

# n_estimators=100
# Specifies the number of boosting trees or stages.
# Here, the model uses up to 100 boosting estimators.

# max_depth=3
# Sets the maximum depth of each tree to 3.
# It controls how complex each tree can become.

# learning_rate=0.1
# Controls how much each new tree contributes
# to the overall model.
# A smaller learning rate means each tree makes
# a smaller contribution.
# It can help make learning more gradual.

# objective="multi:softprob"
# Specifies the objective function for multiclass classification.
# It calculates class probabilities for multiple classes.
# The model predicts probabilities for all three Iris species.

# eval_metric="mlogloss"
# Specifies Multiclass Log Loss as the evaluation metric.
# It measures how well the predicted class probabilities
# match the actual class labels.
# Lower log loss generally indicates better probability predictions.

# random_state=42
# Helps make random operations repeatable.

# Important:
# XGBoost is designed to be efficient and powerful.
# It includes regularization techniques to help control
# model complexity and overfitting.


# ============================================================
# 15. STORE ALL MODELS
# ============================================================

# models = {
#     "Hard Voting": hard_voting,
#     "Soft Voting": soft_voting,
#     "AdaBoost": adaboost,
#     "Gradient Boosting": gbm,
#     "XGBoost": xgboost
# }
# Stores all five models in a Python dictionary.

# What is a dictionary?
# A dictionary stores information in key-value pairs.

# Here:
# The model name is the key.
# The corresponding machine learning model is the value.

# Hard Voting
# Represents the Hard Voting Classifier.

# Soft Voting
# Represents the Soft Voting Classifier.

# AdaBoost
# Represents the Adaptive Boosting model.

# Gradient Boosting
# Represents the Gradient Boosting model.

# XGBoost
# Represents the Extreme Gradient Boosting model.

# Why store models in a dictionary?
# It makes it easier to train and evaluate
# all models using one loop instead of writing
# separate training code for every model.


# ============================================================
# 16. CREATE A LIST TO STORE PERFORMANCE RESULTS
# ============================================================

# results = []
# Creates an empty list named results.
# It stores the evaluation metrics of all five models.

# The following information will be stored:
# Model name.
# Accuracy.
# Precision.
# Recall.
# F1-score.


# ============================================================
# 17. TRAIN AND EVALUATE EACH MODEL
# ============================================================

# for name, model in models.items():
# Starts a loop to process every model in the dictionary.

# name
# Stores the name of the current model.

# model
# Stores the current machine learning model.

# models.items()
# Returns the model names and their corresponding model objects.

# Why use a loop?
# It avoids repeating the same training and evaluation
# statements for every individual model.


# -------------------- TRAIN THE MODEL -----------------------

# model.fit(X_train, y_train)
# Trains the current model using the training dataset.

# X_train
# Contains the standardized input flower measurements.

# y_train
# Contains the numerical species labels.

# Each model learns patterns from the training data
# using its own algorithm.


# -------------------- PREDICT THE OUTPUT ---------------------

# y_pred = model.predict(X_test)
# Uses the trained model to predict the species
# of flowers in the testing dataset.

# X_test
# Contains testing flower measurements.

# y_pred
# Stores the predicted numerical species labels.


# ============================================================
# 18. CALCULATE ACCURACY
# ============================================================

# accuracy = accuracy_score(y_test, y_pred)
# Compares the actual labels with the predicted labels.
# Calculates the proportion of correctly classified samples.

# Formula:
# Accuracy = Correct Predictions / Total Predictions

# Example:
# Suppose 27 out of 30 flowers are correctly classified.

# Accuracy = 27 / 30
# Accuracy = 0.90
# Accuracy = 90%

# Higher accuracy generally indicates better classification
# performance on the testing dataset.


# ============================================================
# 19. CALCULATE PRECISION
# ============================================================

# precision = precision_score(
#     y_test,
#     y_pred,
#     average="weighted",
#     zero_division=0
# )
# Calculates weighted precision for all three classes.

# What is Precision?
# Precision measures how many samples predicted
# as a particular class actually belong to that class.

# Formula:
# Precision = True Positive / (True Positive + False Positive)

# True Positive:
# A sample is correctly predicted as the class being considered.

# False Positive:
# A sample is incorrectly predicted as the class being considered.

# average="weighted"
# Calculates precision separately for each class.
# Then combines the results using the number of actual samples
# in each class as weights.

# Why use weighted average?
# It gives more importance to classes containing more samples.
# It is useful when classes have different numbers of samples.

# zero_division=0
# Returns 0 instead of producing a warning
# if a precision calculation has a zero denominator.


# ============================================================
# 20. CALCULATE RECALL
# ============================================================

# recall = recall_score(
#     y_test,
#     y_pred,
#     average="weighted",
#     zero_division=0
# )
# Calculates weighted recall for all three classes.

# What is Recall?
# Recall measures how many actual samples of a class
# are correctly identified by the model.

# Formula:
# Recall = True Positive / (True Positive + False Negative)

# False Negative:
# A sample actually belongs to a class,
# but the model predicts another class.

# average="weighted"
# Calculates recall for each class and combines the values
# according to the number of actual samples in each class.

# zero_division=0
# Handles cases where the recall calculation
# has a zero denominator.


# ============================================================
# 21. CALCULATE F1-SCORE
# ============================================================

# f1 = f1_score(
#     y_test,
#     y_pred,
#     average="weighted",
#     zero_division=0
# )
# Calculates the weighted F1-score.

# What is F1-score?
# F1-score is the harmonic mean of precision and recall.
# It combines both measures into a single performance value.

# Formula:
# F1-score = 2 * (Precision * Recall) / (Precision + Recall)

# Why use F1-score?
# It provides a balance between precision and recall.
# It is useful when we want to consider both types of performance.

# average="weighted"
# Calculates the F1-score for every class.
# Then calculates the weighted average using class support.

# zero_division=0
# Handles cases where the calculation has a zero denominator.


# ============================================================
# 22. STORE THE RESULTS
# ============================================================

# results.append(
#     [
#         name,
#         accuracy,
#         precision,
#         recall,
#         f1
#     ]
# )
# Adds the performance details of the current model
# to the results list.

# name
# Stores the model name.

# accuracy
# Stores the accuracy value.

# precision
# Stores the weighted precision value.

# recall
# Stores the weighted recall value.

# f1
# Stores the weighted F1-score.

# After all five models are evaluated:
# The results list contains five sets of performance values.


# ============================================================
# 23. DISPLAY CLASSIFICATION REPORT
# ============================================================

# print("\n", name)
# Displays the name of the current model.

# print(
#     classification_report(
#         y_test,
#         y_pred,
#         zero_division=0
#     )
# )
# Displays a detailed classification report
# for the current model.

# The report contains:

# Precision:
# Measures how many predicted samples of a class are correct.

# Recall:
# Measures how many actual samples of a class are correctly identified.

# F1-score:
# Combines precision and recall.

# Support:
# Displays the number of actual testing samples
# belonging to each class.

# Macro Average:
# Calculates the average metric across all classes
# by treating each class equally.

# Weighted Average:
# Calculates the average metric by considering
# the number of actual samples in each class.

# Important:
# The classification report is displayed separately
# for every model.


# ============================================================
# 24. CREATE MODEL PERFORMANCE COMPARISON TABLE
# ============================================================

# results_df = pd.DataFrame(
#     results,
#     columns=[
#         "Model",
#         "Accuracy",
#         "Precision",
#         "Recall",
#         "F1 Score"
#     ]
# )
# Converts the results list into a Pandas DataFrame.

# What is a DataFrame?
# A DataFrame is a table containing rows and columns.

# Model
# Contains the names of the five machine learning models.

# Accuracy
# Contains the accuracy of each model.

# Precision
# Contains the weighted precision of each model.

# Recall
# Contains the weighted recall of each model.

# F1 Score
# Contains the weighted F1-score of each model.

# print("\nModel Performance Comparison:")
# Displays the heading for the comparison table.

# print(results_df.to_string(index=False))
# Displays the complete comparison table.
# to_string() converts the DataFrame into a readable text format.
# index=False hides the default row numbering.

# Why create a comparison table?
# It allows us to compare all five models together.
# We can identify which model performs better
# according to accuracy, precision, recall and F1-score.


# ============================================================
# 25. PLOT MODEL ACCURACY
# ============================================================

# plt.figure(figsize=(9, 5))
# Creates a new figure with width 9 inches and height 5 inches.

# plt.bar(
#     results_df["Model"],
#     results_df["Accuracy"]
# )
# Creates a bar graph to compare the accuracy of all five models.

# results_df["Model"]
# Provides the model names for the horizontal axis.

# results_df["Accuracy"]
# Provides the accuracy values for the vertical axis.

# What is a bar graph?
# A bar graph uses rectangular bars to compare numerical values.
# A taller bar represents a higher accuracy value.

# plt.title("Ensemble Models Accuracy Comparison")
# Sets the title of the graph.

# plt.xlabel("Models")
# Sets the horizontal axis label.
# It represents the names of the machine learning models.

# plt.ylabel("Accuracy")
# Sets the vertical axis label.
# It represents the accuracy values.

# plt.xticks(rotation=30)
# Rotates the model names by 30 degrees.
# It makes long labels easier to read.

# plt.ylim(0, 1.1)
# Sets the vertical axis limits from 0 to 1.1.
# Accuracy values range from 0 to 1.
# The extra space makes the graph easier to view.

# plt.tight_layout()
# Automatically adjusts graph spacing.
# Prevents labels from being cut off or overlapping.

# plt.show()
# Displays the model accuracy comparison graph.


# ============================================================
# 26. FIND THE BEST PERFORMING MODEL
# ============================================================

# best_model = results_df.loc[
#     results_df["Accuracy"].idxmax()
# ]
# Finds the row containing the highest accuracy value.

# results_df["Accuracy"]
# Selects the Accuracy column from the comparison table.

# idxmax()
# Returns the index of the row containing the maximum accuracy.

# results_df.loc[]
# Retrieves the complete row using its index.

# best_model
# Stores the performance details of the model
# having the highest accuracy.

# print("\nBest Performing Model:")
# Displays the heading for the best model.

# print(best_model)
# Displays the name, accuracy, precision, recall
# and F1-score of the best-performing model.

# Important:
# The best model in this program is selected
# only according to the highest accuracy.
# A model with the highest accuracy may not always
# have the highest precision or F1-score.

# If two or more models have the same highest accuracy,
# idxmax() returns the first matching row.


# ============================================================
# 27. UNDERSTAND ENSEMBLE LEARNING
# ============================================================

# What is Ensemble Learning?
# Ensemble Learning is a machine learning technique
# that combines multiple models to make a final prediction.

# Instead of depending on only one model,
# it uses multiple models to improve prediction performance.

# Main idea:
# Combining several models may produce better predictions
# than using an individual model.

# Example:
# Suppose three models predict the following classes:
# Model 1 predicts Class 0.
# Model 2 predicts Class 1.
# Model 3 predicts Class 1.

# The ensemble can combine their predictions
# to produce the final result.

# Two important ensemble approaches used in this practical:

# 1. Bagging:
# Trains multiple models independently using different
# samples of the training data.
# Random Forest is an example of bagging.

# 2. Boosting:
# Trains models sequentially.
# Each new model tries to improve the performance
# of the combined previous models.
# AdaBoost, Gradient Boosting and XGBoost are examples.

# Voting:
# Combines predictions from multiple trained models.
# It can use hard voting or soft voting.


# ============================================================
# 28. DIFFERENCE BETWEEN VOTING AND BOOSTING
# ============================================================

# VOTING:
# Combines predictions from multiple models.
# The models are generally trained independently.
# The final prediction is selected using voting.

# BOOSTING:
# Trains weak learners sequentially.
# Each new learner tries to improve the existing model.
# It combines the learners to produce the final prediction.

# HARD VOTING:
# Uses the majority of predicted class labels.

# SOFT VOTING:
# Uses the average of predicted class probabilities.

# ADABOOST:
# Gives more attention to incorrectly classified samples
# during successive training rounds.

# GRADIENT BOOSTING:
# Uses gradients of a loss function to improve predictions
# by adding new learners sequentially.

# XGBOOST:
# An optimized gradient boosting implementation
# designed for efficient and powerful learning.


# ============================================================
# 29. IMPORTANT DIFFERENCE BETWEEN THE FIVE MODELS
# ============================================================

# HARD VOTING:
# Combines the class predictions of multiple models.
# Selects the class receiving the most votes.

# SOFT VOTING:
# Combines the class probabilities of multiple models.
# Selects the class with the highest average probability.

# ADABOOST:
# Focuses more on samples incorrectly classified
# by earlier weak learners.

# GRADIENT BOOSTING:
# Adds new decision trees sequentially to reduce
# the errors of the combined model.

# XGBOOST:
# Uses an optimized gradient boosting approach.
# It includes regularization and efficient computation techniques.


# ============================================================
# 30. IMPORTANT EVALUATION METRICS
# ============================================================

# ACCURACY:
# Measures the proportion of correctly predicted samples.
# Higher accuracy generally means better overall classification.

# PRECISION:
# Measures how many predicted samples of a class
# actually belong to that class.
# High precision means fewer false positive predictions.

# RECALL:
# Measures how many actual samples of a class
# are correctly identified.
# High recall means fewer false negative predictions.

# F1-SCORE:
# Combines precision and recall using their harmonic mean.
# It gives a balanced measure of both.

# SUPPORT:
# Represents the number of actual samples in each class.

# MACRO AVERAGE:
# Gives equal importance to all classes.

# WEIGHTED AVERAGE:
# Gives importance to each class according to its number
# of actual samples.


# ============================================================
# 31. IMPORTANT VIVA QUESTIONS AND ANSWERS
# ============================================================

# Q1. What is Ensemble Learning?
# Answer:
# Ensemble Learning combines multiple machine learning models
# to make a final prediction.
# It can improve prediction performance and stability.

# Q2. What are the two main ensemble techniques?
# Answer:
# 1. Bagging.
# 2. Boosting.

# Q3. What is Voting Classifier?
# Answer:
# Voting Classifier combines predictions from multiple models
# and selects a final class using voting.

# Q4. What is Hard Voting?
# Answer:
# Hard Voting selects the class that receives
# the maximum number of votes from the models.

# Q5. What is Soft Voting?
# Answer:
# Soft Voting averages the predicted probabilities
# of the models and selects the class with the highest average.

# Q6. What is the difference between Hard and Soft Voting?
# Answer:
# Hard Voting uses predicted class labels.
# Soft Voting uses predicted class probabilities.

# Q7. What is AdaBoost?
# Answer:
# AdaBoost stands for Adaptive Boosting.
# It gives more attention to incorrectly classified samples
# during successive training rounds.

# Q8. What is a weak learner?
# Answer:
# A weak learner is a simple model that performs
# slightly better than random guessing.
# Decision stumps are commonly used as weak learners.

# Q9. What is Gradient Boosting?
# Answer:
# Gradient Boosting is an ensemble technique
# that builds models sequentially to improve predictions
# by reducing errors using gradients of a loss function.

# Q10. What is XGBoost?
# Answer:
# XGBoost stands for Extreme Gradient Boosting.
# It is an optimized implementation of gradient boosting.

# Q11. What is the difference between AdaBoost and Gradient Boosting?
# Answer:
# AdaBoost gives more importance to incorrectly classified
# training samples.
# Gradient Boosting uses gradients of a loss function
# to improve the model step by step.

# Q12. What is Random Forest?
# Answer:
# Random Forest is an ensemble algorithm that combines
# multiple decision trees to make predictions.

# Q13. What is a Decision Tree?
# Answer:
# A Decision Tree is a supervised learning algorithm
# that makes decisions using a tree-like structure.

# Q14. What is n_estimators?
# Answer:
# It specifies the number of estimators or learners
# used by an ensemble model.

# Q15. What is max_depth?
# Answer:
# It controls the maximum depth of a decision tree.

# Q16. What is learning_rate?
# Answer:
# It controls how much each new boosting learner
# contributes to the overall model.

# Q17. What is accuracy?
# Answer:
# Accuracy is the ratio of correct predictions
# to the total number of predictions.

# Q18. What is precision?
# Answer:
# Precision measures how many predicted positive samples
# are actually correct.

# Q19. What is recall?
# Answer:
# Recall measures how many actual positive samples
# are correctly identified.

# Q20. What is F1-score?
# Answer:
# F1-score is the harmonic mean of precision and recall.

# Q21. Why is the Species column encoded?
# Answer:
# Because the target labels are text values.
# Encoding converts them into numerical class labels.

# Q22. Why do we use train_test_split?
# Answer:
# It divides the dataset into training and testing data
# to evaluate how well the model predicts unseen samples.

# Q23. Why is stratify=y used?
# Answer:
# It maintains the proportion of each species
# in both training and testing datasets.

# Q24. Why is StandardScaler used?
# Answer:
# It standardizes numerical features to a comparable scale.

# Q25. What is the purpose of the comparison table?
# Answer:
# It allows us to compare the performance of all five models
# using accuracy, precision, recall and F1-score.

# Q26. How is the best model selected in this practical?
# Answer:
# The model with the highest accuracy is selected
# using idxmax().

# Q27. What are the applications of Ensemble Learning?
# Answer:
# 1. Medical diagnosis.
# 2. Fraud detection.
# 3. Disease prediction.
# 4. Image classification.
# 5. Customer analysis.
# 6. Spam detection.

# Q28. What is overfitting?
# Answer:
# Overfitting occurs when a model learns the training data
# too closely and performs poorly on unseen data.

# Q29. What is the difference between Bagging and Boosting?
# Answer:
# Bagging trains multiple models independently.
# Boosting trains models sequentially,
# with each new model improving the existing model.

# Q30. Is XGBoost a supervised or unsupervised algorithm?
# Answer:
# XGBoost is a supervised machine learning algorithm.
# It is used for classification and regression.


# ============================================================
# 32. EXPECTED OUTPUT EXPLANATION
# ============================================================

# OUTPUT 1: IRIS DATASET

# The program displays:
# Iris Dataset:

# It prints the first five rows of the Iris dataset.
# This confirms that the dataset has been loaded successfully.

# OUTPUT 2: CLASSIFICATION REPORTS

# The program displays a separate classification report
# for each of the five models.

# The model names are:
# 1. Hard Voting.
# 2. Soft Voting.
# 3. AdaBoost.
# 4. Gradient Boosting.
# 5. XGBoost.

# Each report contains:
# Precision.
# Recall.
# F1-score.
# Support.

# OUTPUT 3: MODEL PERFORMANCE COMPARISON TABLE

# The program displays a table containing five rows,
# one for each machine learning model.

# The columns are:
# Model.
# Accuracy.
# Precision.
# Recall.
# F1 Score.

# All values can be compared to understand
# the performance of the five models.

# OUTPUT 4: ACCURACY COMPARISON GRAPH

# The program displays a bar graph.

# X-axis:
# Names of the five machine learning models.

# Y-axis:
# Accuracy values.

# Each bar represents one model's accuracy.

# A taller bar indicates higher accuracy.

# OUTPUT 5: BEST PERFORMING MODEL

# The program displays the model with the highest accuracy.

# It also displays:
# Model name.
# Accuracy.
# Precision.
# Recall.
# F1-score.

# Important:
# Exact accuracy values may vary depending on the dataset,
# software versions and model settings.


# ============================================================
# 33. CONCLUSION
# ============================================================

# In this practical, different ensemble learning algorithms
# are implemented on the Iris dataset.

# First, the Iris dataset is loaded and preprocessed.
# The unnecessary Id column is removed.
# Missing values are removed.
# The input features and target labels are separated.
# The target species are converted into numerical labels.

# The dataset is divided into training and testing data.
# StandardScaler is used to standardize the input features.

# Five ensemble models are created:
# 1. Hard Voting Classifier.
# 2. Soft Voting Classifier.
# 3. AdaBoost Classifier.
# 4. Gradient Boosting Classifier.
# 5. XGBoost Classifier.

# All five models are trained and tested.
# Their performance is evaluated using accuracy,
# precision, recall and F1-score.

# A comparison table and accuracy graph are generated.
# Finally, the model with the highest accuracy is selected.

# Thus, different ensemble learning algorithms are
# successfully implemented and compared using the Iris dataset.

# ============================================================
