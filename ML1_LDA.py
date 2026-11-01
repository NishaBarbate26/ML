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
# AIM:
# Perform Linear Discriminant Analysis (LDA) on the Iris Dataset
# for Dimensionality Reduction and Classification.
# ============================================================


# ============================================================
# 1. INSTALLATION AND EXECUTION COMMANDS
# ============================================================

# WINDOWS:
# python -m pip install pandas matplotlib scikit-learn
# python practical_lda.py

# UBUNTU / LINUX:
# pip install pandas matplotlib scikit-learn
# python3 practical_lda.py

# DATASET:
# Iris.csv

# The Iris.csv file must be present in the same folder
# where the Python program is saved.


# ============================================================
# 2. IMPORTING REQUIRED LIBRARIES
# ============================================================

# import pandas as pd

# Pandas is a Python library used for data manipulation,
# data analysis and handling structured datasets.

# It provides a DataFrame data structure that stores data
# in tabular form, containing rows and columns.

# pd is an alias (short name) for pandas.

# Uses:
# 1. Reading CSV files.
# 2. Displaying dataset records.
# 3. Removing unnecessary columns.
# 4. Selecting input and output features.
# 5. Creating new DataFrames.


# import matplotlib.pyplot as plt

# Matplotlib is a Python library used for data visualization.

# pyplot is a module of Matplotlib used to create graphs.

# plt is an alias for matplotlib.pyplot.

# Uses:
# 1. Creating scatter plots.
# 2. Displaying data distributions.
# 3. Adding graph titles and labels.
# 4. Displaying legends and grid lines.


# from sklearn.model_selection import train_test_split

# Scikit-learn is a Python machine learning library.

# train_test_split() divides a dataset into training
# and testing datasets.

# Training data is used to train the machine learning model.

# Testing data is used to evaluate the model's performance
# on previously unseen observations.


# from sklearn.preprocessing import StandardScaler

# StandardScaler is used for feature scaling.

# It transforms numerical features so that:
# Mean = 0
# Standard Deviation = 1

# Formula:
# Z = (X - Mean) / Standard Deviation

# Here:
# X = Original feature value.
# Mean = Mean of the feature.
# Standard Deviation = Standard deviation of the feature.
# Z = Standardized feature value.

# Why is scaling required?
# Features may have different ranges.
# Scaling brings them to a common scale.
# It prevents features with larger numerical values
# from dominating features with smaller numerical values.


# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LinearDiscriminantAnalysis is the LDA algorithm
# provided by the Scikit-learn library.

# LDA stands for Linear Discriminant Analysis.

# It is a SUPERVISED machine learning algorithm.

# It is used for:
# 1. Dimensionality reduction.
# 2. Classification.
# 3. Finding directions that separate different classes.

# It uses both input features and their known class labels
# to learn the differences between classes.


# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# These functions are used to evaluate the classification model.

# accuracy_score():
# Calculates the percentage of correctly predicted observations.

# confusion_matrix():
# Displays correct and incorrect predictions for each class.

# classification_report():
# Displays precision, recall, F1-score and support
# for every target class.


# ============================================================
# 3. LOADING THE IRIS DATASET
# ============================================================

# df = pd.read_csv("Iris.csv")

# pd.read_csv() is used to read a CSV file.

# Iris.csv is the dataset filename.

# df is the DataFrame variable in which the dataset is stored.

# A DataFrame is a two-dimensional tabular data structure
# containing rows and columns.

# ABOUT THE IRIS DATASET:

# The Iris dataset is a popular dataset used in machine learning
# for classification and data analysis.

# It contains measurements of three different Iris flower species.

# Total observations = 150.
# Total input features = 4.
# Total target classes = 3.

# FOUR INPUT FEATURES:

# 1. SepalLengthCm:
#    Length of the sepal measured in centimetres.

# 2. SepalWidthCm:
#    Width of the sepal measured in centimetres.

# 3. PetalLengthCm:
#    Length of the petal measured in centimetres.

# 4. PetalWidthCm:
#    Width of the petal measured in centimetres.

# THREE TARGET CLASSES:

# 1. Iris-setosa
# 2. Iris-versicolor
# 3. Iris-virginica

# Each species contains 50 observations in the standard dataset.


# print("Original Iris Dataset:")
# print(df.head())

# print() displays information on the screen.

# df.head() displays the first five rows of the dataset.

# It helps us:
# 1. Check whether the dataset loaded successfully.
# 2. Understand the dataset structure.
# 3. Identify column names.
# 4. Observe the values stored in each column.

# EXPECTED OUTPUT:

# Displays the first five records of the Iris dataset
# with flower measurements and their corresponding species.


# ============================================================
# 4. DATA PREPROCESSING
# ============================================================

# df = df.drop(columns=["Id"], errors="ignore")

# drop() removes unnecessary rows or columns from a DataFrame.

# columns=["Id"] specifies that the Id column must be removed.

# The Id column is only a unique identification number.
# It does not provide useful information for predicting
# the flower species.

# errors="ignore" prevents an error if the Id column
# is not available in the dataset.

# WHY REMOVE ID?

# Identification numbers do not represent actual flower
# characteristics and should not be used as input features.


# X = df.drop(columns=["Species"])

# X represents the independent variables or input features.

# The Species column is removed because it is the target variable.

# X contains four input features:
# 1. Sepal Length.
# 2. Sepal Width.
# 3. Petal Length.
# 4. Petal Width.

# These measurements are given to the machine learning model
# to predict the flower species.

# In machine learning:
# X = Input features.
# y = Target variable.


# y = df["Species"]

# y represents the dependent variable or target variable.

# It contains the actual species names of the flowers.

# Possible values:
# Iris-setosa.
# Iris-versicolor.
# Iris-virginica.

# The model learns the relationship between X and y
# during the training process.


# ============================================================
# 5. SPLITTING DATA INTO TRAINING AND TESTING SETS
# ============================================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# train_test_split() divides the dataset into training
# and testing datasets.

# PARAMETERS:

# X:
# Contains the input features.

# y:
# Contains the actual species labels.

# test_size=0.2:
# Allocates 20% of the observations for testing.
# The remaining 80% is used for training.

# Total observations = 150.

# Training observations = 80% of 150 = 120.

# Testing observations = 20% of 150 = 30.

# random_state=42:
# Fixes the random seed so that the same data split
# is generated whenever the program is executed.

# 42 is simply a selected random seed value.

# stratify=y:
# Maintains the proportion of each species in both
# training and testing datasets.

# Since each species has 50 observations:

# Training:
# Iris-setosa = 40.
# Iris-versicolor = 40.
# Iris-virginica = 40.

# Testing:
# Iris-setosa = 10.
# Iris-versicolor = 10.
# Iris-virginica = 10.

# FOUR OUTPUT VARIABLES:

# X_train:
# Input features used for training the model.

# X_test:
# Input features used for testing the model.

# y_train:
# Actual species labels used during training.

# y_test:
# Actual species labels used for model evaluation.

# WHY SPLIT DATA?

# If we evaluate the model using the same data on which
# it was trained, the performance may appear misleading.

# Separate testing data helps us check whether the model
# can correctly classify new, unseen observations.


# ============================================================
# 6. FEATURE SCALING USING STANDARD SCALER
# ============================================================

# scaler = StandardScaler()

# Creates an object of the StandardScaler class.

# StandardScaler standardizes the numerical input features.

# FORMULA:

# Z = (X - Mean) / Standard Deviation

# After standardization:
# Mean is approximately 0.
# Standard deviation is approximately 1.

# WHY STANDARDIZATION?

# Different features may have different numerical ranges.

# For example:
# Sepal length may have values around 4 to 8.
# Petal width may have values around 0 to 3.

# Standardization puts these features on a comparable scale.

# It also helps numerical algorithms work with
# consistently scaled input features.


# X_train_scaled = scaler.fit_transform(X_train)

# fit_transform() performs two operations.

# STEP 1: fit()

# Calculates the mean and standard deviation
# of each feature using the training dataset.

# STEP 2: transform()

# Uses the calculated mean and standard deviation
# to standardize the training dataset.

# X_train_scaled stores the standardized training features.

# IMPORTANT:
# Scaling parameters are learned only from training data.

# This prevents DATA LEAKAGE.


# X_test_scaled = scaler.transform(X_test)

# transform() standardizes the testing dataset.

# It uses the mean and standard deviation already calculated
# from the training dataset.

# WHY NOT fit_transform() ON TEST DATA?

# The testing dataset must not influence the training process.

# Using its own mean and standard deviation would introduce
# information from the testing data into preprocessing.

# Both training and testing datasets must use
# the same scaling parameters.


# ============================================================
# 7. APPLYING LINEAR DISCRIMINANT ANALYSIS (LDA)
# ============================================================

# lda = LinearDiscriminantAnalysis(n_components=2)

# Creates an LDA model.

# LDA is a supervised dimensionality reduction
# and classification algorithm.

# SUPERVISED LEARNING:

# Supervised learning means the algorithm learns using
# input data and already-known output labels.

# In this practical:
# Input = Four flower measurements.
# Output = Three flower species.

# MAIN OBJECTIVE OF LDA:

# LDA finds new directions that maximize the separation
# between different classes and minimize the spread
# of observations belonging to the same class.

# TWO IMPORTANT CONCEPTS:

# 1. WITHIN-CLASS SCATTER:

# It measures how much observations belonging to the same
# class are spread around their class mean.

# Lower within-class scatter is desirable.

# 2. BETWEEN-CLASS SCATTER:

# It measures how far apart the means of different classes are.

# Higher between-class scatter is desirable.

# LDA tries to maximize the ratio:

# Between-Class Scatter / Within-Class Scatter

# This helps separate different classes effectively.


# n_components=2:

# Specifies the number of new dimensions to retain.

# Original dimensions = 4.

# Reduced dimensions = 2.

# IMPORTANT FORMULA:

# Maximum number of LDA components = Number of classes - 1.

# Here:
# Number of classes = 3.

# Maximum components = 3 - 1 = 2.

# Therefore, we can retain a maximum of two LDA components.

# These components are:
# 1. Linear Discriminant 1 (LD1).
# 2. Linear Discriminant 2 (LD2).

# LD1 and LD2 are new features formed as linear combinations
# of the original four input features.


# X_train_lda = lda.fit_transform(X_train_scaled, y_train)

# fit_transform() performs two operations.

# STEP 1: fit()

# Trains the LDA model using standardized training features
# and their corresponding actual species labels.

# The model learns the directions that best separate
# the three flower species.

# STEP 2: transform()

# Transforms the original four-dimensional training data
# into a two-dimensional representation.

# Original features = 4.

# New LDA components = 2.

# X_train_lda contains:
# First column = LD1.
# Second column = LD2.

# PURPOSE:

# Reduces the number of features while preserving
# information useful for class separation.

# The two-dimensional representation is also useful
# for visualizing the different flower species.


# ============================================================
# 8. PREDICTING TESTING DATA
# ============================================================

# y_pred = lda.predict(X_test_scaled)

# predict() uses the already-trained LDA model
# to predict the species of testing observations.

# X_test_scaled contains 30 unseen flower observations.

# The model examines their four standardized measurements
# and predicts one of the three species.

# y_pred stores the predicted species labels.

# IMPORTANT DIFFERENCE:

# fit():
# Learns the model from the training data.

# transform():
# Converts the input features into new dimensions.

# fit_transform():
# Performs both fitting and transformation.

# predict():
# Predicts the output class for given observations.


# ============================================================
# 9. DISPLAYING ACTUAL AND PREDICTED SPECIES
# ============================================================

# print("\nActual Species:")
# print(y_test.values)

# Displays the actual species labels of testing observations.

# y_test contains the correct answers from the original dataset.

# values converts the pandas Series into a NumPy array.

# There are 30 actual species labels.

# These are used to check the correctness of predictions.


# print("\nPredicted Species:")
# print(y_pred)

# Displays the species predicted by the trained LDA model.

# Each predicted label corresponds to the same-position
# observation in the actual species array.

# If actual and predicted labels match:
# The prediction is CORRECT.

# If actual and predicted labels differ:
# The prediction is INCORRECT.


# ============================================================
# 10. CALCULATING ACCURACY
# ============================================================

# print("\nAccuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# accuracy_score() calculates the overall classification accuracy.

# It compares:
# y_test = Actual species labels.
# y_pred = Predicted species labels.

# FORMULA:

# Accuracy =
# (Number of Correct Predictions / Total Predictions) * 100

# EXAMPLE:

# Total testing observations = 30.

# If 29 predictions are correct:

# Accuracy = (29 / 30) * 100.

# Accuracy = 96.67%.

# The actual accuracy is calculated by the program.

# INTERPRETATION:

# Higher accuracy indicates that the model correctly
# classified a larger percentage of testing observations.

# Accuracy alone does not explain every type of classification
# error, so we also calculate the confusion matrix
# and classification report.


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# confusion_matrix() compares actual class labels
# with predicted class labels.

# It displays correct and incorrect predictions
# for each flower species.

# Since there are three species, the confusion matrix
# contains three rows and three columns.

# DEFAULT ARRANGEMENT:

# Rows = Actual classes.
# Columns = Predicted classes.

# EXAMPLE CONFUSION MATRIX:

# [[10  0  0]
#  [ 0  9  1]
#  [ 0  0 10]]

# EXPLANATION:

# FIRST ROW:

# 10 Iris-setosa flowers were correctly classified
# as Iris-setosa.

# SECOND ROW:

# 9 Iris-versicolor flowers were correctly classified.

# 1 Iris-versicolor flower was incorrectly classified
# as Iris-virginica.

# THIRD ROW:

# 10 Iris-virginica flowers were correctly classified.

# DIAGONAL ELEMENTS:

# Values on the main diagonal represent correct predictions.

# OFF-DIAGONAL ELEMENTS:

# Values outside the main diagonal represent incorrect predictions.

# PERFECT CONFUSION MATRIX:

# All values should be on the main diagonal.
# All remaining values should be zero.

# NOTE:
# The above matrix is only an example.
# Actual values depend on the model's predictions.


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# classification_report() displays the performance
# of the classification model.

# It provides four important evaluation metrics:
# 1. Precision.
# 2. Recall.
# 3. F1-score.
# 4. Support.


# PRECISION:

# Precision tells us how many observations predicted
# as a particular class actually belong to that class.

# FORMULA:

# Precision = TP / (TP + FP)

# TP = True Positive.
# FP = False Positive.

# Example:
# If the model predicts 10 flowers as Iris-setosa
# and all 10 actually belong to Iris-setosa:

# Precision = 10 / 10 = 1.00.


# RECALL:

# Recall tells us how many actual observations of a class
# were correctly identified by the model.

# FORMULA:

# Recall = TP / (TP + FN)

# FN = False Negative.

# Example:
# If there are 10 actual Iris-setosa flowers
# and the model correctly identifies all 10:

# Recall = 10 / 10 = 1.00.


# F1-SCORE:

# F1-score is the harmonic mean of precision and recall.

# It provides a balance between precision and recall.

# FORMULA:

# F1-score = 2 * (Precision * Recall) /
#            (Precision + Recall)

# A higher F1-score indicates better classification performance.


# SUPPORT:

# Support represents the actual number of observations
# belonging to a particular class in the testing dataset.

# In this practical:

# Iris-setosa = 10.
# Iris-versicolor = 10.
# Iris-virginica = 10.


# ACCURACY:

# Represents the overall proportion of correctly
# classified testing observations.


# MACRO AVERAGE:

# Calculates the average metric across all classes.

# Every class receives equal importance,
# regardless of the number of observations in that class.


# WEIGHTED AVERAGE:

# Calculates the average metric by considering
# the number of observations in each class.

# Since the Iris dataset is balanced,
# macro and weighted averages may be similar.


# ============================================================
# 13. VISUALIZATION USING A SCATTER PLOT
# ============================================================

# plt.figure(figsize=(8, 6))

# figure() creates a new graph.

# figsize=(8, 6) specifies the dimensions of the graph.

# Width = 8 inches.
# Height = 6 inches.

# A larger graph makes the observations easier to understand.


# for species in y.unique():

# unique() returns all different species present
# in the target column.

# Three unique species:
# Iris-setosa.
# Iris-versicolor.
# Iris-virginica.

# The for loop processes one species at a time.

# This allows us to plot each species separately.


# plt.scatter(
#     X_train_lda[y_train.values == species, 0],
#     X_train_lda[y_train.values == species, 1],
#     label=species
# )

# scatter() creates a scatter plot.

# A scatter plot represents observations using individual points.

# FIRST ARGUMENT:

# X_train_lda[y_train.values == species, 0]

# Selects the first LDA component (LD1)
# for observations belonging to the current species.

# This represents the horizontal X-axis.

# SECOND ARGUMENT:

# X_train_lda[y_train.values == species, 1]

# Selects the second LDA component (LD2)
# for observations belonging to the current species.

# This represents the vertical Y-axis.

# CONDITION:

# y_train.values == species

# Checks which training observations belong to the
# current species being processed by the loop.

# label=species:

# Assigns the species name to the plotted points.

# Each species gets a separate label in the graph legend.

# GRAPH PURPOSE:

# To visualize how well LDA separates the three flower species
# after reducing the original four features into two dimensions.


# plt.xlabel("Linear Discriminant 1")

# Sets the horizontal axis label.

# LD1 represents the first new feature generated by LDA.

# It is a linear combination of the original four measurements.


# plt.ylabel("Linear Discriminant 2")

# Sets the vertical axis label.

# LD2 represents the second new feature generated by LDA.

# It is another linear combination of the original four measurements.


# plt.title("LDA - Iris Dataset")

# Sets the title of the graph.

# It indicates that the graph represents LDA results
# for the Iris flower dataset.


# plt.legend()

# Displays the legend containing the names of the three species.

# It helps identify which points belong to each flower species.


# plt.grid(True)

# Adds horizontal and vertical grid lines.

# Grid lines help us understand the position of points
# in the scatter plot.


# plt.show()

# Displays the completed scatter plot.


# ============================================================
# 14. DETAILED GRAPH INTERPRETATION
# ============================================================

# GRAPH TYPE:
# Scatter Plot.

# X-AXIS:
# Linear Discriminant 1 (LD1).

# Y-AXIS:
# Linear Discriminant 2 (LD2).

# EACH POINT:
# Represents one flower observation from the training dataset.

# DIFFERENT COLOURS:
# Represent different Iris flower species.

# CLUSTER:
# A group of points located close to one another.

# CLASS SEPARATION:
# Indicates how effectively different species are separated
# by the LDA algorithm.

# EXPECTED OBSERVATIONS:

# 1. Iris-setosa is generally clearly separated
#    from the other two species.

# 2. Iris-versicolor and Iris-virginica may show some overlap.

# 3. Less overlap indicates better class separation.

# 4. Separate clusters indicate that LDA has found
#    useful discriminant directions.

# 5. Overlapping clusters indicate that some flowers
#    have similar measurements.

# IMPORTANT:
# The graph displays only training observations.

# Testing observations are not plotted in this graph.

# The graph is used for visualization,
# whereas the testing dataset is used for model evaluation.


# ============================================================
# 15. PREDICTING THE SPECIES OF A NEW FLOWER
# ============================================================

# print("\nEnter Flower Measurements to Predict Species:")

# Displays a message asking the user to enter
# the measurements of a new flower.

# This demonstrates how a trained machine learning model
# can classify a new, previously unseen observation.


# sepal_length = float(input("Enter Sepal Length (cm): "))

# input() accepts the sepal length entered by the user.

# float() converts the entered value into a decimal number.

# Example:
# 5.1


# sepal_width = float(input("Enter Sepal Width (cm): "))

# Accepts the sepal width of the new flower.

# Example:
# 3.5


# petal_length = float(input("Enter Petal Length (cm): "))

# Accepts the petal length of the new flower.

# Example:
# 1.4


# petal_width = float(input("Enter Petal Width (cm): "))

# Accepts the petal width of the new flower.

# Example:
# 0.2

# IMPORTANT:
# All measurements must be entered in centimetres.
# The order must match the original input features.


# new_flower = pd.DataFrame(
#     [[sepal_length, sepal_width, petal_length, petal_width]],
#     columns=X.columns
# )

# pd.DataFrame() creates a new DataFrame.

# It stores the four measurements entered by the user.

# [[...]] represents one flower observation
# containing four feature values.

# columns=X.columns assigns the same feature names
# and feature order used during model training.

# This is important because the model expects
# the same four input features in the same order.

# The new DataFrame contains:
# 1. Sepal Length.
# 2. Sepal Width.
# 3. Petal Length.
# 4. Petal Width.


# new_flower_scaled = scaler.transform(new_flower)

# Standardizes the new flower measurements.

# transform() uses the mean and standard deviation
# previously calculated from the training dataset.

# We do not use fit_transform() here.

# WHY?

# The new flower must be processed using the same scaling
# parameters used during model training.

# This ensures consistency between training data
# and new input data.


# prediction = lda.predict(new_flower_scaled)

# predict() uses the trained LDA model to classify
# the new flower.

# The model receives four standardized measurements.

# It predicts one of the three flower species.

# prediction stores the predicted class label.


# print("\nPredicted Flower Species:", prediction[0])

# Displays the predicted flower species.

# prediction[0] selects the first prediction
# because the user has entered only one flower.

# POSSIBLE OUTPUTS:

# Iris-setosa.
# Iris-versicolor.
# Iris-virginica.

# The result depends on the measurements entered by the user.


# ============================================================
# 16. EXPECTED OUTPUT EXPLANATION
# ============================================================

# OUTPUT 1: ORIGINAL DATASET

# Displays the first five records of Iris.csv.

# OUTPUT 2: ACTUAL SPECIES

# Displays the actual species labels of the 30 testing flowers.

# OUTPUT 3: PREDICTED SPECIES

# Displays the species predicted by the LDA model.

# OUTPUT 4: ACCURACY

# Displays the percentage of correctly classified
# testing observations.

# OUTPUT 5: CONFUSION MATRIX

# Displays correct and incorrect predictions
# for all three species.

# OUTPUT 6: CLASSIFICATION REPORT

# Displays precision, recall, F1-score, support,
# accuracy, macro average and weighted average.

# OUTPUT 7: SCATTER PLOT

# Displays the two-dimensional representation
# of the training observations after LDA.

# OUTPUT 8: NEW FLOWER PREDICTION

# Accepts four flower measurements from the user
# and predicts the corresponding species.


# ============================================================
# 17. IMPORTANT LDA VIVA QUESTIONS AND ANSWERS
# ============================================================

# Q1. What is LDA?

# Ans:
# LDA stands for Linear Discriminant Analysis.
# It is a supervised machine learning algorithm used
# for dimensionality reduction and classification.


# Q2. Is LDA supervised or unsupervised?

# Ans:
# LDA is a supervised learning algorithm because
# it uses known class labels during training.


# Q3. What is dimensionality reduction?

# Ans:
# Dimensionality reduction is the process of reducing
# the number of input features while retaining
# important information from the original dataset.


# Q4. What is the main objective of LDA?

# Ans:
# To maximize the separation between different classes
# and minimize the spread within the same class.


# Q5. What is the difference between PCA and LDA?

# Ans:
# PCA is an unsupervised dimensionality reduction technique.
# It finds directions of maximum overall variance.

# LDA is a supervised dimensionality reduction technique.
# It finds directions that best separate different classes.


# Q6. How many features are present in the Iris dataset?

# Ans:
# Four input features.


# Q7. How many classes are present in the Iris dataset?

# Ans:
# Three classes.


# Q8. Why is n_components set to 2?

# Ans:
# Because the maximum number of LDA components is
# the number of classes minus one.

# Maximum components = 3 - 1 = 2.


# Q9. What is LD1?

# Ans:
# LD1 is the first linear discriminant component.
# It represents a new feature that helps separate classes.


# Q10. What is LD2?

# Ans:
# LD2 is the second linear discriminant component.
# It provides another direction for class separation.


# Q11. Why is StandardScaler used?

# Ans:
# It standardizes the input features so that their
# mean is approximately zero and standard deviation is one.


# Q12. What is the difference between fit() and transform()?

# Ans:
# fit() learns the required parameters from
