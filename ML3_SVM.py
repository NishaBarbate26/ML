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
# AIM:
# Implement Support Vector Machine (SVM) to Recognize
# Handwritten Digits Using the Digits Dataset.
# ============================================================


# ============================================================
# 1. INSTALLATION AND EXECUTION COMMANDS
# ============================================================

# Windows:
# python -m pip install matplotlib scikit-learn
# python practical_svm.py

# Ubuntu/Linux:
# pip install matplotlib scikit-learn
# python3 -m pip install matplotlib scikit-learn
# python3 practical_svm.py


# ============================================================
# 2. IMPORTING LIBRARIES
# ============================================================

# import matplotlib.pyplot as plt
# Imports Matplotlib for creating graphs and displaying images.
# plt is the short name used for pyplot.

# from sklearn.datasets import load_digits
# Imports the load_digits dataset from Scikit-learn.
# It contains images of handwritten digits from 0 to 9.
# Each image is an 8 x 8 pixel image.

# from sklearn.model_selection import train_test_split
# Imports train_test_split.
# It divides the dataset into training and testing data.

# from sklearn.preprocessing import StandardScaler
# Imports StandardScaler.
# It standardizes the input features.
# Standardization brings features to a similar scale.

# from sklearn.pipeline import make_pipeline
# Imports make_pipeline.
# It combines multiple machine learning steps into one pipeline.
# Here, it combines scaling and SVM classification.

# from sklearn.svm import SVC
# Imports Support Vector Classifier.
# SVC is used to perform classification using the SVM algorithm.

# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Imports three evaluation functions.
# accuracy_score calculates the accuracy of the model.
# confusion_matrix displays correct and incorrect predictions.
# classification_report displays precision, recall, and F1-score.


# ============================================================
# 3. LOADING THE HANDWRITTEN DIGITS DATASET
# ============================================================

# digits = load_digits()
# Loads the built-in handwritten digits dataset.
# No external CSV file is required.
# The dataset contains handwritten digit images from 0 to 9.
# Each image has 8 rows and 8 columns of pixels.
# Pixel values generally range from 0 to 16.

# X = digits.data
# Stores the input features.
# Each image is converted into a one-dimensional array.
# An 8 x 8 image contains 64 pixels.
# Therefore, every image has 64 input features.

# y = digits.target
# Stores the correct digit labels.
# These are the output values that the model must predict.
# The target values range from 0 to 9.


# ============================================================
# 4. DISPLAYING DATASET INFORMATION
# ============================================================

# print("Handwritten Digits Dataset")
# Displays the dataset heading.

# print("Total Images:", len(X))
# len(X) calculates the total number of handwritten images.
# The digits dataset contains 1797 images.

# print("Number of Features:", X.shape[1])
# X.shape gives the dimensions of the input dataset.
# X.shape[1] gives the number of features in each image.
# Every image contains 64 pixel features.

# print("Digit Classes:", digits.target_names)
# Displays all possible digit classes.
# The classes are 0, 1, 2, 3, 4, 5, 6, 7, 8, and 9.


# ============================================================
# 5. DISPLAYING SAMPLE HANDWRITTEN DIGITS
# ============================================================

# fig, axes = plt.subplots(2, 5, figsize=(10, 5))
# Creates a figure containing 2 rows and 5 columns of subplots.
# A total of 10 sample images can be displayed.
# figsize=(10, 5) sets the width and height of the figure.
# fig represents the complete figure.
# axes represents the individual graph areas.

# for ax, image, label in zip(
#     axes.ravel(),
#     digits.images[:10],
#     digits.target[:10]
# ):
# The for loop processes the first 10 handwritten images.
# axes.ravel() converts the 2D axes array into a one-dimensional array.
# digits.images[:10] selects the first 10 original 8 x 8 images.
# digits.target[:10] selects their corresponding correct labels.
# zip() combines the axes, images, and labels together.
# ax represents the current subplot.
# image represents the current handwritten image.
# label represents the correct digit.

#     ax.imshow(image, cmap="gray")
# Displays the handwritten image in the current subplot.
# imshow() is used to display image data.
# cmap="gray" displays the image in grayscale.

#     ax.set_title("Digit: " + str(label))
# Displays the actual digit label above the image.
# str(label) converts the numerical label into text.

#     ax.axis("off")
# Hides the x-axis and y-axis markings.
# This makes the handwritten images easier to view.

# plt.tight_layout()
# Adjusts the spacing between the subplots.
# It prevents images and titles from overlapping.

# plt.show()
# Displays all 10 handwritten digit images.


# ============================================================
# 6. SPLITTING DATASET INTO TRAINING AND TESTING DATA
# ============================================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# train_test_split() divides the dataset into two parts.

# X_train contains input images used for training.
# X_test contains input images used for testing.
# y_train contains the correct labels for training images.
# y_test contains the correct labels for testing images.

# test_size=0.2 means 20% of the dataset is used for testing.
# The remaining 80% is used for training.

# random_state=42 ensures that the same split is obtained every time.
# It makes the experiment reproducible.

# stratify=y maintains the proportion of each digit class.
# It helps ensure that all digit classes are represented
# in both training and testing datasets.


# ============================================================
# 7. CREATING THE SVM MODEL
# ============================================================

# model = make_pipeline(
#     StandardScaler(),
#     SVC(kernel="rbf", C=10, gamma="scale")
# )

# make_pipeline() combines multiple steps into one model pipeline.

# StandardScaler()
# Standardizes the input features before classification.
# It transforms features so they have approximately
# zero mean and unit variance.

# SVC()
# Creates a Support Vector Classifier.
# It implements the Support Vector Machine algorithm.

# kernel="rbf"
# Specifies the Radial Basis Function kernel.
# It helps SVM classify data that cannot be separated
# easily using a straight line or flat boundary.
# It can create non-linear decision boundaries.

# C=10
# C is the regularization parameter.
# It controls the balance between classification errors
# and the complexity of the decision boundary.
# A higher C gives more importance to correctly classifying
# training examples, but may increase overfitting.

# gamma="scale"
# Gamma controls how far the influence of a training point reaches.
# scale automatically calculates a suitable gamma value
# using the number of features and their variance.

# The pipeline first standardizes the data.
# Then the SVM classifier uses the standardized data to learn.


# ============================================================
# 8. TRAINING THE SVM MODEL
# ============================================================

# model.fit(X_train, y_train)
# fit() trains the SVM model using the training dataset.
# The model learns patterns from the handwritten digit images.
# It learns how to distinguish digits from 0 to 9.

# During training, SVM identifies important data points
# called support vectors.
# These points help determine the decision boundary.


# ============================================================
# 9. PREDICTING DIGIT CLASSES
# ============================================================

# y_pred = model.predict(X_test)
# predict() uses the trained SVM model to classify test images.
# X_test contains images that were not used for training.
# y_pred stores the predicted digit labels.
# Each prediction is a digit from 0 to 9.


# ============================================================
# 10. CALCULATING ACCURACY
# ============================================================

# accuracy = accuracy_score(y_test, y_pred)
# Compares actual digit labels with predicted digit labels.
# Calculates the proportion of correct predictions.

# print("\nSVM Classification Results")
# Displays the classification results heading.

# print("Accuracy:", accuracy * 100, "%")
# Displays the accuracy as a percentage.
# accuracy ranges from 0 to 1.
# Multiplying by 100 converts it into a percentage.
# For example, 0.98 means 98% accuracy.
# Higher accuracy indicates more correct predictions.


# ============================================================
# 11. DISPLAYING THE CONFUSION MATRIX
# ============================================================

# print("\nConfusion Matrix:")
# Displays the confusion matrix heading.

# print(confusion_matrix(y_test, y_pred))
# Creates and displays the confusion matrix.
# It compares actual digit labels with predicted digit labels.

# Rows represent actual digit classes.
# Columns represent predicted digit classes.
# Diagonal values represent correctly classified images.
# Values outside the diagonal represent incorrect predictions.

# For example:
# A value in row 3 and column 3 represents correctly predicted digit 3.
# A value in row 3 and column 5 represents actual digit 3
# incorrectly predicted as digit 5.

# Higher diagonal values generally indicate better classification.


# ============================================================
# 12. DISPLAYING THE CLASSIFICATION REPORT
# ============================================================

# print("\nClassification Report:")
# Displays the classification report heading.

# print(classification_report(y_test, y_pred))
# Displays the performance of the model for each digit class.

# Precision:
# Measures how many images predicted as a particular digit
# actually belong to that digit.

# Recall:
# Measures how many actual images of a digit
# are correctly identified by the model.

# F1-score:
# It is the harmonic mean of precision and recall.
# It balances precision and recall.

# Support:
# Represents the number of actual test images belonging to each digit.

# Macro Average:
# Calculates the average metric across all classes
# by giving equal importance to each digit class.

# Weighted Average:
# Calculates the average metric by considering
# the number of samples in each class.


# ============================================================
# 13. DISPLAYING ACTUAL AND PREDICTED DIGITS
# ============================================================

# print("\nActual Digits:", y_test[:20])
# Displays the first 20 actual digit labels from the testing dataset.
# [:20] selects the first 20 values.

# print("Predicted Digits:", y_pred[:20])
# Displays the first 20 predicted digit labels.
# These predictions can be compared with the actual digits.
# Matching values indicate correct predictions.


# ============================================================
# 14. DISPLAYING ONE TEST IMAGE WITH ITS PREDICTION
# ============================================================

# plt.figure(figsize=(4, 4))
# Creates a new figure with width 4 inches and height 4 inches.

# plt.imshow(X_test[0].reshape(8, 8), cmap="gray")
# X_test[0] selects the first test image.
# reshape(8, 8) converts its 64 pixel values
# back into an 8 x 8 image.
# cmap="gray" displays the image in grayscale.

# plt.title("Predicted Digit: " + str(y_pred[0]))
# Displays the digit predicted by the SVM model.
# y_pred[0] selects the prediction for the first test image.
# str() converts the prediction into text.

# plt.axis("off")
# Hides the x-axis and y-axis markings.

# plt.show()
# Displays the test image with its predicted digit.


# ============================================================
# 15. UNDERSTANDING THE COMPLETE OUTPUT
# ============================================================

# Handwritten Digits Dataset:
# Displays the dataset heading.

# Total Images:
# Displays the total number of images.
# The dataset contains 1797 handwritten digit images.

# Number of Features:
# Displays the number of pixel features in each image.
# Every image contains 64 features.

# Digit Classes:
# Displays the ten possible classes from 0 to 9.

# Sample Handwritten Digits:
# Displays the first 10 handwritten digit images.
# Each image has its actual digit label.

# SVM Classification Results:
# Displays the model's classification accuracy.

# Confusion Matrix:
# Displays correct and incorrect predictions for every digit.

# Classification Report:
# Displays precision, recall, F1-score, and support.

# Actual Digits:
# Displays the actual labels of the first 20 test images.

# Predicted Digits:
# Displays the labels predicted by the trained SVM model.

# Final Image:
# Displays one test image and the digit predicted by the model.


# ============================================================
# 16. IMPORTANT SVM CONCEPTS FOR VIVA
# ============================================================

# Q1. What is SVM?
# SVM stands for Support Vector Machine.
# It is a supervised machine learning algorithm.
# It is used for classification and regression.
# It finds a decision boundary that separates different classes.

# Q2. What is SVC?
# SVC stands for Support Vector Classifier.
# It is the Scikit-learn implementation of SVM for classification.

# Q3. What is a hyperplane?
# A hyperplane is a decision boundary used to separate classes.
# In two dimensions, it is a line.
# In three dimensions, it is a plane.
# In higher dimensions, it is called a hyperplane.

# Q4. What are support vectors?
# Support vectors are the data points closest to the decision boundary.
# They help determine the position of the hyperplane.

# Q5. What is a margin?
# Margin is the distance between the decision boundary
# and the closest data points from the classes.
# SVM tries to find a suitable boundary with a large margin.

# Q6. What is a kernel in SVM?
# A kernel helps SVM handle data that cannot be separated
# using a simple straight-line decision boundary.
# It allows SVM to create non-linear decision boundaries.

# Q7. What is the RBF kernel?
# RBF stands for Radial Basis Function.
# It is a non-linear kernel.
# It is useful when the relationship between data points is complex.

# Q8. What is the C parameter?
# C is the regularization parameter.
# It controls the penalty for classification errors.
# A high C tries to classify training points more correctly.
# A low C allows more training errors to obtain a simpler boundary.

# Q9. What is gamma?
# Gamma controls the influence of individual training points.
# A high gamma creates a more complex decision boundary.
# A low gamma creates a smoother decision boundary.

# Q10. Why is StandardScaler used?
# StandardScaler brings all input features to a similar scale.
# It is important because SVM is sensitive to feature scales.

# Q11. What is a pipeline?
# A pipeline combines multiple machine learning steps.
# In this practical, it combines feature scaling and SVM classification.

# Q12. Why is the dataset divided into training and testing data?
# Training data is used to teach the model.
# Testing data is used to evaluate its performance on unseen images.

# Q13. What is classification?
# Classification is a supervised learning task.
# It predicts a category or class for given input data.

# Q14. What is accuracy?
# Accuracy is the ratio of correct predictions to total predictions.
# Accuracy = Correct Predictions / Total Predictions.

# Q15. What is a confusion matrix?
# A confusion matrix compares actual and predicted classes.
# It shows the number of correct and incorrect predictions.

# Q16. What is precision?
# Precision measures how many predicted positive cases are correct.

# Q17. What is recall?
# Recall measures how many actual positive cases are correctly identified.

# Q18. What is F1-score?
# F1-score is the harmonic mean of precision and recall.

# Q19. What is the size of each handwritten image?
# Each image is 8 x 8 pixels.

# Q20. How many features are present in each image?
# Each image contains 64 features because 8 x 8 = 64.

# Q21. How many digit classes are available?
# There are 10 digit classes, from 0 to 9.

# Q22. What are the applications of SVM?
# Handwritten digit recognition.
# Face recognition.
# Text classification.
# Image classification.
# Spam email detection.
# Medical diagnosis.

# Q23. What are the advantages of SVM?
# It works well with high-dimensional data.
# It can handle linear and non-linear classification.
# It can perform well on relatively small and medium-sized datasets.

# Q24. What are the limitations of SVM?
# Training can be slow for very large datasets.
# Choosing the correct kernel and parameters can be difficult.
# It may require feature scaling for good performance.


# ============================================================
# 17. CONCLUSION
# ============================================================

# In this practical, Support Vector Machine is implemented
# to recognize handwritten digits using the built-in Digits dataset.

# The dataset contains 1797 handwritten images.
# Each image contains 64 pixel features and belongs to a digit class
# from 0 to 9.

# The dataset is divided into 80% training data and 20% testing data.
# StandardScaler is used to standardize the input features.
# The SVM model is created using the RBF kernel.
# The model is trained and used to predict handwritten digits.

# Finally, its performance is evaluated using accuracy,
# confusion matrix, and classification report.

# Sample handwritten images and one predicted test image
# are also displayed.

# Therefore, SVM can be used for handwritten digit recognition.
