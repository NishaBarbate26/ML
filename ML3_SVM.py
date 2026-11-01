# AIM: Implement Support Vector Machine (SVM) for classifying
# handwritten digit images into numerical classes 0 to 9.

# Installation commands:
# Windows:
# python -m pip install scikit-learn matplotlib numpy
# python practical_svm.py
#
# Ubuntu/Linux:
# pip install scikit-learn matplotlib numpy
# python3 -m pip install scikit-learn matplotlib numpy
# python3 practical_svm.py


import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load handwritten digits dataset

digits = load_digits()

X = digits.data
y = digits.target

print("Handwritten Digits Dataset")
print("Total Images:", len(X))
print("Number of Features:", X.shape[1])
print("Digit Classes:", digits.target_names)

# Display sample handwritten digits

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for ax, image, label in zip(
    axes.ravel(),
    digits.images[:10],
    digits.target[:10]
):
    ax.imshow(image, cmap="gray")
    ax.set_title("Digit: " + str(label))
    ax.axis("off")

plt.tight_layout()
plt.show()

# Split dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create SVM model

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=10, gamma="scale")
)

# Train the model

model.fit(X_train, y_train)

# Predict digit classes

y_pred = model.predict(X_test)

# Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nSVM Classification Results")

print("Accuracy:", accuracy * 100, "%")

# Display confusion matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display classification report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display actual and predicted digits

print("\nActual Digits:", y_test[:20])

print("Predicted Digits:", y_pred[:20])

# Display one test image and its prediction

plt.figure(figsize=(4, 4))

plt.imshow(X_test[0].reshape(8, 8), cmap="gray")

plt.title("Predicted Digit: " + str(y_pred[0]))

plt.axis("off")

plt.show()


# ============================================================
# AIM: Implement Support Vector Machine (SVM) for classifying
# handwritten digit images into numerical classes 0 to 9.
# ============================================================


# -------------------- INSTALLATION --------------------------

# Windows:
# python -m pip install scikit-learn matplotlib numpy
# python practical_svm.py

# Ubuntu/Linux:
# pip install scikit-learn matplotlib numpy
# python3 -m pip install scikit-learn matplotlib numpy
# python3 practical_svm.py


# -------------------- IMPORT LIBRARIES ----------------------

# import matplotlib.pyplot as plt
# Imports Matplotlib for displaying handwritten digit images
# and visualization graphs.

# from sklearn.datasets import load_digits
# Imports the built-in handwritten digits dataset from Scikit-learn.

# from sklearn.model_selection import train_test_split
# Imports the function used to divide the dataset into training
# and testing datasets.

# from sklearn.preprocessing import StandardScaler
# Imports StandardScaler to standardize the pixel values.

# from sklearn.pipeline import make_pipeline
# Imports make_pipeline to combine preprocessing and the SVM model.

# from sklearn.svm import SVC
# Imports Support Vector Classifier, which implements the SVM algorithm.

# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Imports evaluation metrics to check the performance of the model.


# -------------------- LOAD DATASET --------------------------

# digits = load_digits()
# Loads the built-in handwritten digits dataset.
# It contains images of handwritten digits from 0 to 9.
# Each image is an 8 x 8 grayscale image.

# X = digits.data
# Stores the input features of the dataset.
# Each image is converted into a one-dimensional array.
# Each image contains 64 pixel values because 8 x 8 = 64.
# Therefore, each row represents one handwritten digit image.

# y = digits.target
# Stores the actual output labels of the images.
# The labels represent the numerical digits from 0 to 9.

# print("Handwritten Digits Dataset")
# Prints the heading of the dataset information.

# print("Total Images:", len(X))
# Displays the total number of handwritten digit images.
# The built-in digits dataset contains 1797 images.

# print("Number of Features:", X.shape[1])
# Displays the number of pixel features in each image.
# Each image has 64 features.

# print("Digit Classes:", digits.target_names)
# Displays all possible digit classes.
# The classes are 0, 1, 2, 3, 4, 5, 6, 7, 8 and 9.


# -------------------- DISPLAY SAMPLE IMAGES ----------------

# fig, axes = plt.subplots(2, 5, figsize=(10, 5))
# Creates a figure containing 2 rows and 5 columns.
# It provides 10 separate plotting areas to display sample images.
# figsize=(10, 5) sets the width and height of the figure.

# for ax, image, label in zip(
#     axes.ravel(),
#     digits.images[:10],
#     digits.target[:10]
# ):
# Repeats the loop for the first 10 images.
# axes.ravel() converts the 2D plotting areas into a 1D array.
# digits.images[:10] selects the first 10 original images.
# digits.target[:10] selects the corresponding actual labels.
# zip() combines each plotting area, image and label.

# ax.imshow(image, cmap="gray")
# Displays the handwritten digit image.
# cmap="gray" displays the image using grayscale shades.

# ax.set_title("Digit: " + str(label))
# Displays the actual digit label above each image.
# str(label) converts the numerical label into a string.

# ax.axis("off")
# Hides the x-axis and y-axis markings.

# plt.tight_layout()
# Automatically adjusts the spacing between the images.
# Prevents image titles and plots from overlapping.

# plt.show()
# Displays the first 10 handwritten digit images.


# -------------------- SPLIT DATASET -------------------------

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )
# Divides the complete dataset into training and testing data.
# X contains input pixel features.
# y contains the actual digit labels.
# test_size=0.2 means 20% of the data is used for testing.
# The remaining 80% is used for training.
# random_state=42 ensures the same split when the program runs again.
# stratify=y maintains the proportion of each digit class
# in both training and testing datasets.

# X_train
# Contains the input images used to train the model.

# X_test
# Contains the input images used to test the model.

# y_train
# Contains the actual digit labels for training images.

# y_test
# Contains the actual digit labels for testing images.


# -------------------- CREATE SVM MODEL ----------------------

# model = make_pipeline(
#     StandardScaler(),
#     SVC(kernel="rbf", C=10, gamma="scale")
# )
# Creates a machine learning pipeline.
# The pipeline performs feature scaling first and classification next.

# StandardScaler()
# Standardizes the pixel features.
# It transforms the features so they have approximately
# zero mean and unit variance.
# Scaling is important because SVM is sensitive to feature scales.

# SVC()
# Creates a Support Vector Classifier.
# It is used to classify handwritten digit images.

# kernel="rbf"
# Selects the Radial Basis Function kernel.
# The RBF kernel helps SVM classify data that is not
# linearly separable.
# It can create nonlinear decision boundaries.

# C=10
# Controls the penalty for classification errors.
# A higher C value tries to classify training samples correctly
# but may increase the risk of overfitting.
# A lower C value allows more training errors and stronger regularization.

# gamma="scale"
# Automatically calculates the gamma value based on
# the number of features and their variance.
# Gamma controls how far the influence of a training sample extends.


# -------------------- TRAIN THE MODEL -----------------------

# model.fit(X_train, y_train)
# Trains the SVM classifier using the training dataset.
# The model learns patterns from the pixel values
# and their corresponding digit labels.


# -------------------- PREDICT DIGIT CLASSES -----------------

# y_pred = model.predict(X_test)
# Uses the trained SVM model to predict the digit classes
# for the testing images.
# Stores the predicted labels in y_pred.


# -------------------- CALCULATE ACCURACY --------------------

# accuracy = accuracy_score(y_test, y_pred)
# Compares the actual testing labels with predicted labels.
# Calculates the proportion of correctly classified images.

# accuracy * 100
# Converts the accuracy value into a percentage.

# print("\nSVM Classification Results")
# Prints the heading for the classification results.

# print("Accuracy:", accuracy * 100, "%")
# Displays the classification accuracy in percentage.
# Higher accuracy generally indicates better classification performance.


# -------------------- CONFUSION MATRIX ----------------------

# print("\nConfusion Matrix:")
# Prints the confusion matrix heading.

# print(confusion_matrix(y_test, y_pred))
# Displays the confusion matrix.
# Rows represent the actual digit classes.
# Columns represent the predicted digit classes.
# Diagonal values represent correctly classified images.
# Values outside the diagonal represent incorrect predictions.
# It helps identify which digits are confused with other digits.


# -------------------- CLASSIFICATION REPORT -----------------

# print("\nClassification Report:")
# Prints the classification report heading.

# print(classification_report(y_test, y_pred))
# Displays the performance measures for each digit class.
# It includes precision, recall, F1-score and support.

# Precision
# Measures how many images predicted as a particular digit
# actually belong to that digit.
# Formula: True Positive / (True Positive + False Positive).

# Recall
# Measures how many actual images of a digit
# are correctly identified by the model.
# Formula: True Positive / (True Positive + False Negative).

# F1-score
# Represents the harmonic mean of precision and recall.
# It balances precision and recall.
# Formula: 2 * Precision * Recall / (Precision + Recall).

# Support
# Represents the number of actual testing samples
# belonging to each digit class.

# Macro Average
# Calculates the average metric across all classes
# by giving equal importance to every digit class.

# Weighted Average
# Calculates the average metric across all classes
# by considering the number of samples in each class.


# -------------------- ACTUAL AND PREDICTED DIGITS ------------

# print("\nActual Digits:", y_test[:20])
# Displays the first 20 actual digit labels from the testing dataset.

# print("Predicted Digits:", y_pred[:20])
# Displays the first 20 digit labels predicted by the SVM model.
# Comparing both arrays helps us understand the predictions.


# -------------------- DISPLAY ONE TEST IMAGE ----------------

# plt.figure(figsize=(4, 4))
# Creates a figure with a width of 4 inches and height of 4 inches.

# plt.imshow(X_test[0].reshape(8, 8), cmap="gray")
# Selects the first testing image.
# reshape(8, 8) converts its 64 pixel values back into
# the original 8 x 8 image format.
# cmap="gray" displays the image in grayscale.

# plt.title("Predicted Digit: " + str(y_pred[0]))
# Displays the digit predicted by the SVM model as the title.

# plt.axis("off")
# Hides the x-axis and y-axis markings.

# plt.show()
# Displays the selected test image with its predicted digit.


# ============================================================
# IMPORTANT CONCEPTS FOR PRACTICAL AND VIVA
# ============================================================

# 1. What is SVM?
# SVM stands for Support Vector Machine.
# It is a supervised machine learning algorithm.
# It is used for classification and regression.
# It finds the best decision boundary, called a hyperplane,
# to separate different classes.

# 2. What is a hyperplane?
# A hyperplane is a decision boundary that separates
# different classes in the feature space.

# 3. What are support vectors?
# Support vectors are the data points closest to the decision boundary.
# They play an important role in determining the position
# of the optimal hyperplane.

# 4. What is a margin?
# Margin is the distance between the decision boundary
# and the nearest data points from different classes.
# SVM tries to maximize this margin.

# 5. What is a kernel?
# A kernel helps SVM handle data that cannot be separated
# using a straight-line decision boundary.
# It allows SVM to create nonlinear decision boundaries.

# 6. What is the RBF kernel?
# RBF stands for Radial Basis Function.
# It is a popular nonlinear kernel used in SVM.
# It measures similarity between data points.

# 7. Why is StandardScaler used?
# StandardScaler brings features to a comparable scale.
# It helps SVM perform more effectively because
# SVM is sensitive to feature magnitudes.

# 8. Why is the dataset divided into training and testing data?
# Training data is used to teach the model.
# Testing data is used to evaluate the model
# on images it did not use during training.

# 9. Why is stratify=y used?
# It maintains the proportion of each digit class
# in both training and testing datasets.

# 10. What is classification accuracy?
# Accuracy measures the proportion of correct predictions.
# Formula:
# Accuracy = Correct Predictions / Total Predictions.

# 11. What is a confusion matrix?
# It is a table that compares actual classes
# with predicted classes.
# It shows correct and incorrect classifications.

# 12. What is the difference between classification and regression?
# Classification predicts categories or class labels.
# Regression predicts continuous numerical values.

# 13. Is SVM supervised or unsupervised?
# SVM is a supervised machine learning algorithm.
# It learns using input data and known output labels.

# 14. What is multiclass classification?
# Multiclass classification means predicting one class
# from more than two possible classes.
# This practical classifies images into 10 classes, from 0 to 9.

# 15. Why is this dataset suitable for SVM?
# Each handwritten digit image is represented by 64 pixel features.
# These features can be used by SVM to learn patterns
# and distinguish between different handwritten digits.


# ============================================================
# EXPECTED OUTPUT
# ============================================================

# 1. Dataset information:
# Handwritten Digits Dataset
# Total Images: 1797
# Number of Features: 64
# Digit Classes: [0 1 2 3 4 5 6 7 8 9]

# 2. First graph:
# Displays 10 sample handwritten digit images
# arranged in 2 rows and 5 columns.

# 3. Classification results:
# Displays the SVM classification accuracy.
# The exact accuracy may vary depending on the dataset split
# and library version.

# 4. Confusion matrix:
# Displays the correct and incorrect predictions
# for all 10 digit classes.

# 5. Classification report:
# Displays precision, recall, F1-score and support
# for each digit from 0 to 9.

# 6. Actual and predicted digits:
# Displays the first 20 actual labels and their predictions.

# 7. Final graph:
# Displays one testing image with the digit predicted by SVM.


# ============================================================
# CONCLUSION
# ============================================================

# In this practical, the Support Vector Machine algorithm
# is implemented to classify handwritten digit images
# into numerical classes from 0 to 9.
#
# The built-in digits dataset is loaded and divided into
# training and testing datasets.
#
# StandardScaler is used to standardize the input features.
# An SVM classifier with an RBF kernel is trained
# using the training dataset.
#
# The trained model predicts the digit classes of testing images.
# Its performance is evaluated using accuracy, confusion matrix
# and classification report.
#
# Thus, handwritten digit classification is successfully
# performed using the Support Vector Machine algorithm.
# ============================================================
