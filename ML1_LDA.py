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

# AIM: Perform Linear Discriminant Analysis (LDA) on the Iris Dataset.
#
# Installation commands:
# Windows:
# python -m pip install pandas matplotlib scikit-learn
# python practical_lda.py
#
# Ubuntu/Linux:
# pip install pandas matplotlib scikit-learn
# python3 practical_lda.py
#
# ------------------------------------------------------------
# IMPORTING REQUIRED LIBRARIES
# ------------------------------------------------------------

# import pandas as pd
# Imports the Pandas library and gives it the short name 'pd'.
# Pandas is used to load, store, manipulate and analyze tabular data.
# It provides DataFrame structures to work with rows and columns.

# import matplotlib.pyplot as plt
# Imports the Pyplot module from Matplotlib with the short name 'plt'.
# It is used to create graphs and visualize the LDA results.

# from sklearn.model_selection import train_test_split
# Imports train_test_split from Scikit-learn.
# It divides the dataset into training data and testing data.
# Training data is used to train the model.
# Testing data is used to evaluate the model.

# from sklearn.preprocessing import StandardScaler
# Imports StandardScaler from Scikit-learn.
# It standardizes numerical features so they have a similar scale.
# Standardization helps prevent large-valued features from dominating.

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Imports the LinearDiscriminantAnalysis algorithm.
# LDA is a supervised machine learning algorithm.
# It is used for classification and dimensionality reduction.
# It tries to separate different classes by finding suitable directions.

# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Imports three evaluation metrics.
# accuracy_score calculates the percentage of correct predictions.
# classification_report displays precision, recall, F1-score and support.
# confusion_matrix displays correct and incorrect predictions for each class.

# ------------------------------------------------------------
# LOADING THE DATASET
# ------------------------------------------------------------

# df = pd.read_csv("Iris.csv")
# Reads the Iris.csv file and stores its contents in a DataFrame named df.
# The dataset must be present in the same folder as this Python program.
# The Iris dataset contains flower measurements and their species.
# It generally contains 150 records, 4 measurement columns and 3 species.
# The four features are sepal length, sepal width, petal length and petal width.

# print("Original Iris Dataset:")
# Displays the heading Original Iris Dataset on the screen.

# print(df.head())
# Displays the first five rows of the dataset.
# head() is a Pandas function used to view the first five records by default.
# It helps us check whether the dataset has loaded correctly.

# ------------------------------------------------------------
# DATA PREPROCESSING
# ------------------------------------------------------------

# df = df.drop(columns=["Id"], errors="ignore")
# Removes the Id column from the dataset if it exists.
# The Id column is only an identification number.
# It does not provide useful information for predicting flower species.
# columns=["Id"] specifies the column to remove.
# errors="ignore" prevents an error if the Id column does not exist.

# X = df.drop(columns=["Species"])
# Stores the input features in X.
# It removes the Species column because Species is the output to predict.
# X contains four flower measurements:
# SepalLengthCm, SepalWidthCm, PetalLengthCm and PetalWidthCm.
# These are also called independent variables or input variables.

# y = df["Species"]
# Stores the target output column in y.
# Species contains the actual flower categories.
# These are Setosa, Versicolor and Virginica.
# y is also called the dependent variable or target variable.

# ------------------------------------------------------------
# SPLITTING THE DATASET
# ------------------------------------------------------------

# X_train, X_test, y_train, y_test = train_test_split(
# Divides the input features and target labels into training and testing sets.
# X_train contains input features for training.
# X_test contains input features for testing.
# y_train contains correct species labels for training.
# y_test contains correct species labels for testing.

#     X,
#     Passes all input features to the train_test_split function.

#     y,
#     Passes the target species labels to the function.

#     test_size=0.2,
#     Assigns 20% of the dataset to testing.
#     The remaining 80% is used for training.
#     For 150 records, this gives approximately 120 training and 30 testing records.

#     random_state=42,
#     Fixes the random number generator so the same split can be reproduced.
#     The value 42 is an arbitrary fixed number.
#     It helps obtain consistent results every time the program runs.

#     stratify=y
#     Maintains the same class proportions in training and testing datasets.
#     It ensures that all three species are represented proportionally.
#     For the standard Iris dataset, each species has 50 records.
#     Therefore, approximately 40 records per species go into training
#     and 10 records per species go into testing.

# )
# Closes the train_test_split function call.

# ------------------------------------------------------------
# FEATURE SCALING
# ------------------------------------------------------------

# scaler = StandardScaler()
# Creates an object of the StandardScaler class.
# It is used to standardize the input features.
# Standardization converts features to a similar scale.
# The formula is:
# Z = (X - Mean) / Standard Deviation
# After standardization, each feature has approximately:
# Mean = 0
# Standard Deviation = 1

# X_train_scaled = scaler.fit_transform(X_train)
# fit() calculates the mean and standard deviation from training data.
# transform() uses those values to standardize the training data.
# fit_transform() performs both operations together.
# The result is stored in X_train_scaled.
# We calculate scaling values only from training data to avoid data leakage.

# X_test_scaled = scaler.transform(X_test)
# Standardizes the testing data using the training data's mean and standard deviation.
# We use transform() instead of fit_transform().
# This ensures that testing data does not influence the training process.
# The standardized testing data is stored in X_test_scaled.

# ------------------------------------------------------------
# APPLYING LINEAR DISCRIMINANT ANALYSIS
# ------------------------------------------------------------

# lda = LinearDiscriminantAnalysis(n_components=2)
# Creates an LDA model and stores it in the variable lda.
# n_components=2 means that we want to reduce the data to two dimensions.
# The Iris dataset has three classes.
# Maximum LDA components = Number of classes - 1.
# Therefore, maximum components = 3 - 1 = 2.
# LDA tries to maximize separation between different species.
# It is a supervised technique because it uses the known species labels.

# X_train_lda = lda.fit_transform(X_train_scaled, y_train)
# fit() trains the LDA model using standardized training features and their labels.
# transform() converts the training data into two new LDA components.
# fit_transform() performs both operations together.
# X_train_lda contains the reduced two-dimensional training data.
# The two new dimensions are called Linear Discriminant 1 and 2.
# LDA tries to keep different flower species as separate as possible.

# y_pred = lda.predict(X_test_scaled)
# Uses the trained LDA model to predict the species of testing flowers.
# X_test_scaled contains standardized testing measurements.
# predict() returns the predicted species for every testing record.
# The predicted labels are stored in y_pred.

# ------------------------------------------------------------
# DISPLAYING ACTUAL AND PREDICTED SPECIES
# ------------------------------------------------------------

# print("\nActual Species:")
# Displays the heading Actual Species.
# \n moves the output to a new line before printing the heading.

# print(y_test.values)
# Displays the actual species labels from the testing dataset.
# values converts the Pandas Series into a NumPy array.

# print("\nPredicted Species:")
# Displays the heading Predicted Species on a new line.

# print(y_pred)
# Displays the species predicted by the trained LDA model.
# We can compare these predictions with the actual species labels.

# ------------------------------------------------------------
# MODEL EVALUATION
# ------------------------------------------------------------

# print("\nAccuracy:", accuracy_score(y_test, y_pred) * 100, "%")
# Calculates and displays the model's accuracy as a percentage.
# accuracy_score() compares actual labels with predicted labels.
# Formula:
# Accuracy = (Number of Correct Predictions / Total Predictions) * 100
# Multiplication by 100 converts the accuracy into a percentage.
# Higher accuracy generally indicates better classification performance.

# print("\nConfusion Matrix:")
# Displays the heading Confusion Matrix.

# print(confusion_matrix(y_test, y_pred))
# Creates and displays the confusion matrix.
# It compares actual species with predicted species.
# Rows represent actual classes.
# Columns represent predicted classes.
# Diagonal values represent correctly classified records.
# Values outside the diagonal represent incorrect classifications.
# For three species, the confusion matrix has 3 rows and 3 columns.

# print("\nClassification Report:")
# Displays the heading Classification Report.

# print(classification_report(y_test, y_pred))
# Displays the detailed classification performance.
# Precision = Correct Positive Predictions / Total Positive Predictions.
# Recall = Correct Positive Predictions / Actual Positive Records.
# F1-score = 2 * (Precision * Recall) / (Precision + Recall).
# Support = Number of actual records belonging to each species.
# Macro average calculates the unweighted average across classes.
# Weighted average calculates the average considering the number of records
# belonging to each class.
# Higher precision, recall and F1-score generally indicate better performance.

# ------------------------------------------------------------
# VISUALIZATION OF LDA RESULTS
# ------------------------------------------------------------

# plt.figure(figsize=(8, 6))
# Creates a new graph with width 8 inches and height 6 inches.
# figsize controls the size of the graph.

# for species in y.unique():
# Starts a loop that visits every unique flower species in the target column.
# unique() returns the different species without repeating them.
# The loop runs once for Setosa, Versicolor and Virginica.

#     plt.scatter(
#     Creates a scatter plot for the current flower species.
#     A scatter plot displays data points using dots.

#         X_train_lda[y_train.values == species, 0],
#         Selects the first LDA component for the current species.
#         Column index 0 represents Linear Discriminant 1.
#         y_train.values == species selects only the records of that species.

#         X_train_lda[y_train.values == species, 1],
#         Selects the second LDA component for the current species.
#         Column index 1 represents Linear Discriminant 2.
#         These values are displayed on the vertical axis.

#         label=species
#         Assigns the current species name as the label in the graph legend.

#     )
#     Closes the scatter plot function call.

# plt.xlabel("Linear Discriminant 1")
# Sets the horizontal axis label to Linear Discriminant 1.

# plt.ylabel("Linear Discriminant 2")
# Sets the vertical axis label to Linear Discriminant 2.

# plt.title("LDA - Iris Dataset")
# Sets the title of the graph.

# plt.legend()
# Displays the legend containing the names of the three species.

# plt.grid(True)
# Displays grid lines in the graph.
# True enables the grid.

# plt.show()
# Displays the completed LDA scatter plot.
# Each species is represented using a separate group of points.
# Well-separated groups indicate that LDA has separated the classes effectively.
# Some overlap between Versicolor and Virginica may occur.

# ------------------------------------------------------------
# PREDICTING THE SPECIES OF A NEW FLOWER
# ------------------------------------------------------------

# print("\nEnter Flower Measurements to Predict Species:")
# Displays a message asking the user to enter flower measurements.

# sepal_length = float(input("Enter Sepal Length (cm): "))
# input() accepts the sepal length entered by the user.
# float() converts the entered value into a decimal number.
# The value is stored in the variable sepal_length.
# Sepal length is measured in centimeters.

# sepal_width = float(input("Enter Sepal Width (cm): "))
# Accepts the sepal width from the user.
# float() converts the input into a decimal number.
# The value is stored in sepal_width.

# petal_length = float(input("Enter Petal Length (cm): "))
# Accepts the petal length from the user.
# float() converts the input into a decimal number.
# The value is stored in petal_length.

# petal_width = float(input("Enter Petal Width (cm): "))
# Accepts the petal width from the user.
# float() converts the input into a decimal number.
# The value is stored in petal_width.

# new_flower = pd.DataFrame(
# Creates a new Pandas DataFrame containing the measurements of one flower.
# A DataFrame stores data in rows and columns.

#     [[sepal_length, sepal_width, petal_length, petal_width]],
# Stores the four measurements in a single row.
# The outer list represents the row.
# The inner list contains the four feature values.

#     columns=X.columns
# Assigns the same column names and order used by the training dataset.
# This ensures that the new flower measurements match the model's input features.

# )
# Closes the DataFrame function call.

# new_flower_scaled = scaler.transform(new_flower)
# Standardizes the new flower measurements.
# It uses the same mean and standard deviation calculated from training data.
# We must not fit the scaler again on the new flower.
# The standardized measurements are stored in new_flower_scaled.

# prediction = lda.predict(new_flower_scaled)
# Passes the standardized new flower measurements to the trained LDA model.
# predict() identifies the most likely flower species.
# The predicted species is stored in prediction.

# print("\nPredicted Flower Species:", prediction[0])
# Displays the predicted species of the new flower.
# prediction[0] accesses the first prediction because only one flower was entered.

# ------------------------------------------------------------
# CONCLUSION
# ------------------------------------------------------------

# Linear Discriminant Analysis was successfully applied to the Iris dataset.
# The dataset was divided into training and testing sets.
# The features were standardized using StandardScaler.
# LDA reduced the four input features into two linear discriminant components.
# The model classified the flower species and its performance was evaluated
# using accuracy, confusion matrix and classification report.
# A scatter plot was generated to visualize the separation between species.
# Finally, the trained model was used to predict the species of a new flower.

# ------------------------------------------------------------
# IMPORTANT VIVA QUESTIONS
# ------------------------------------------------------------

# Q1. What is LDA?
# Ans: LDA stands for Linear Discriminant Analysis.
# It is a supervised machine learning technique used for classification
# and dimensionality reduction.

# Q2. Is LDA supervised or unsupervised?
# Ans: LDA is supervised because it uses class labels during training.

# Q3. What is the main objective of LDA?
# Ans: Its main objective is to maximize the separation between different classes
# while minimizing the spread of data within each class.

# Q4. What is dimensionality reduction?
# Ans: Dimensionality reduction is the process of reducing the number of
# input features while preserving useful information.

# Q5. What is the difference between PCA and LDA?
# Ans: PCA is an unsupervised technique that finds directions of maximum variance.
# LDA is a supervised technique that finds directions that best separate classes.

# Q6. Why do we use StandardScaler?
# Ans: It standardizes numerical features so they have approximately
# zero mean and unit standard deviation.

# Q7. Why do we use train_test_split?
# Ans: It divides the dataset into training and testing data.
# Training data trains the model and testing data evaluates it.

# Q8. What is the purpose of random_state=42?
# Ans: It makes the random splitting reproducible.

# Q9. What is the purpose of stratify=y?
# Ans: It maintains the class proportions in training and testing data.

# Q10. What is a confusion matrix?
# Ans: It is a table used to compare actual class labels with predicted labels.

# Q11. What is accuracy?
# Ans: Accuracy is the percentage of correctly classified records.

# Q12. What are precision and recall?
# Ans: Precision measures how many predicted positive records are correct.
# Recall measures how many actual positive records are correctly identified.

# Q13. Why is n_components=2 used?
# Ans: The Iris dataset has three classes.
# The maximum number of LDA components is classes minus one.
# Therefore, the maximum number of components is 3 - 1 = 2.

# Q14. What is the Iris dataset?
# Ans: It is a standard dataset containing measurements of three iris flower
# species: Setosa, Versicolor and Virginica.

# Q15. What is the purpose of plt.scatter()?
# Ans: It creates a scatter plot to visualize the relationship between
# two numerical variables.

# Q16. What is the difference between fit_transform() and transform()?
# Ans: fit_transform() learns the required parameters and transforms the data.
# transform() uses already learned parameters to transform new data.

# Q17. Why do we use prediction[0]?
# Ans: It accesses the first predicted result because only one new flower
# is entered by the user.

