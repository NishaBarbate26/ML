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
# EXPLANATION LINE BY LINE
# ============================================================

# import pandas as pd
# Imports Pandas to load and process the dataset.

# import matplotlib.pyplot as plt
# Imports Matplotlib to display graphs.

# train_test_split
# Splits the dataset into training and testing data.

# StandardScaler
# Standardizes the numerical input features.

# VotingClassifier
# Combines predictions from multiple machine learning models.

# RandomForestClassifier
# Imports the Random Forest classification algorithm.

# AdaBoostClassifier
# Imports the Adaptive Boosting classification algorithm.

# GradientBoostingClassifier
# Imports the Gradient Tree Boosting algorithm.

# DecisionTreeClassifier
# Imports the Decision Tree classification algorithm.

# accuracy_score
# Calculates the percentage of correct predictions.

# precision_score
# Measures how many positive predictions are correct.

# recall_score
# Measures how many actual positive cases are identified.

# f1_score
# Calculates the harmonic mean of precision and recall.

# classification_report
# Displays precision, recall, F1-score and support.

# XGBClassifier
# Imports the XGBoost classification algorithm.

# pd.read_csv("Iris.csv")
# Loads the Iris dataset from the local CSV file.

# df.head()
# Displays the first five records.

# df.drop(columns=["Id"], errors="ignore")
# Removes the unnecessary Id column if it exists.

# df.dropna()
# Removes rows containing missing values.

# X = df.drop("Species", axis=1)
# Selects flower measurements as input features.

# y = df["Species"]
# Stores the flower species as the target variable.

# y.astype("category").cat.codes
# Converts the three species names into numerical classes.

# train_test_split()
# Divides the dataset into 80% training and 20% testing data.

# stratify=y
# Maintains the proportion of each species in both datasets.

# StandardScaler()
# Creates an object for standardizing the input features.

# scaler.fit_transform(X_train)
# Learns the scaling values from training data and standardizes it.

# scaler.transform(X_test)
# Standardizes testing data using the training data's scaling values.

# DecisionTreeClassifier()
# Creates a Decision Tree model.

# RandomForestClassifier()
# Creates a Random Forest model using 50 decision trees.

# VotingClassifier()
# Combines multiple models to make a final prediction.

# voting="hard"
# Selects the class receiving the highest number of votes.

# voting="soft"
# Selects the class with the highest average predicted probability.

# AdaBoostClassifier()
# Creates an Adaptive Boosting model.
# It gives more attention to incorrectly classified training samples.

# GradientBoostingClassifier()
# Creates a Gradient Boosting model.
# It builds trees sequentially to improve prediction errors.

# XGBClassifier()
# Creates an XGBoost classification model.

# n_estimators
# Specifies the number of boosting trees or estimators.

# max_depth
# Controls the maximum depth of each tree.

# learning_rate
# Controls the contribution of each boosting step.

# models = {...}
# Stores all five models in a dictionary.

# results = []
# Creates an empty list to store evaluation results.

# for name, model in models.items()
# Iterates through all the models.

# model.fit()
# Trains each model using the training dataset.

# model.predict()
# Predicts the flower species for the testing dataset.

# accuracy_score()
# Calculates the accuracy of the model.

# precision_score()
# Calculates weighted precision for all three classes.

# recall_score()
# Calculates weighted recall for all three classes.

# f1_score()
# Calculates weighted F1-score for all three classes.

# results.append()
# Stores the evaluation measures of each model.

# classification_report()
# Displays detailed performance measures for each species.

# pd.DataFrame()
# Creates a comparison table of all models.

# results_df.to_string()
# Displays the comparison table.

# plt.bar()
# Creates a bar graph to compare model accuracy.

# plt.title()
# Sets the graph title.

# plt.xlabel()
# Sets the horizontal axis label.

# plt.ylabel()
# Sets the vertical axis label.

# plt.show()
# Displays the accuracy comparison graph.

# idxmax()
# Finds the row containing the highest accuracy.

# results_df.loc[]
# Retrieves the performance details of the best model.

# print(best_model)
# Displays the best-performing model and its evaluation scores.
