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


# ============================================================
# EXPLANATION LINE BY LINE
# ============================================================

# import pandas as pd
# Imports Pandas to load and process the dataset.

# import matplotlib.pyplot as plt
# Imports Matplotlib to display graphs.

# train_test_split
# Divides the dataset into training and testing data.

# LabelEncoder
# Converts categorical target labels into numerical values.

# RandomForestClassifier
# Imports the Random Forest classification algorithm.

# accuracy_score
# Calculates the accuracy of the model.

# confusion_matrix
# Shows correct and incorrect predictions.

# classification_report
# Displays precision, recall and F1-score.

# pd.read_csv("car_evaluation.csv")
# Loads the car evaluation dataset.

# df.head()
# Displays the first five records.

# df.info()
# Displays column names, data types and non-null values.

# df.dropna()
# Removes rows containing missing values.

# df.drop("class", axis=1)
# Selects input features by removing the target column.

# y = df["class"]
# Stores the target column containing car evaluation categories.

# pd.get_dummies()
# Converts categorical input features into numerical columns.

# LabelEncoder()
# Creates an encoder for the target column.

# encoder.fit_transform(y)
# Converts target categories into numerical labels.

# train_test_split()
# Splits the data into 80% training and 20% testing data.

# stratify=y
# Maintains class proportions in both datasets.

# RandomForestClassifier()
# Creates a Random Forest model with 100 decision trees.

# model.fit()
# Trains the model using training data.

# model.predict()
# Predicts car evaluation categories for testing data.

# accuracy_score()
# Calculates the percentage of correct predictions.

# confusion_matrix()
# Creates a table showing actual and predicted classes.

# classification_report()
# Displays precision, recall and F1-score for each class.

# feature_importances_
# Calculates the importance of every input feature.

# pd.Series()
# Creates a Pandas Series containing feature importance values.

# sort_values()
# Sorts features from highest to lowest importance.

# importance.plot()
# Displays a bar graph of feature importance.

# plt.show()
# Displays the graph.

# pd.DataFrame()
# Creates a new car record for prediction.

# pd.get_dummies()
# Converts the new car's categorical values into numerical columns.

# reindex()
# Ensures that the new car has the same input columns as training data.

# model.predict(new_car)
# Predicts the evaluation category of the new car.

# encoder.inverse_transform()
# Converts the numerical prediction back into its original category.