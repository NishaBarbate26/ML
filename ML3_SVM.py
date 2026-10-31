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
# EXPLANATION LINE BY LINE
# ============================================================

# import matplotlib.pyplot as plt
# Imports Matplotlib to display handwritten digit images.

# load_digits
# Loads the built-in handwritten digits dataset.

# train_test_split
# Divides the dataset into training and testing data.

# StandardScaler
# Standardizes the input features.

# make_pipeline
# Combines data preprocessing and the SVM model.

# SVC
# Imports the Support Vector Classifier algorithm.

# accuracy_score
# Calculates the classification accuracy.

# confusion_matrix
# Displays actual and predicted classifications.

# classification_report
# Displays precision, recall and F1-score.

# digits = load_digits()
# Loads the handwritten digits dataset.

# X = digits.data
# Stores the pixel values of all digit images.
# Each image contains 64 pixel features.

# y = digits.target
# Stores the actual digit labels from 0 to 9.

# len(X)
# Displays the total number of digit images.

# X.shape[1]
# Displays the number of features in each image.

# digits.target_names
# Displays all digit classes from 0 to 9.

# plt.subplots(2, 5)
# Creates a grid containing two rows and five columns.

# axes.ravel()
# Converts the axes into a one-dimensional array.

# zip(...)
# Combines each image, its label and its plotting area.

# ax.imshow(image, cmap="gray")
# Displays the handwritten digit in grayscale.

# ax.set_title(...)
# Displays the actual digit label above the image.

# ax.axis("off")
# Hides the graph axes.

# plt.tight_layout()
# Adjusts the spacing between images.

# plt.show()
# Displays the sample handwritten digit images.

# train_test_split(...)
# Divides the dataset into 80% training and 20% testing data.
# random_state=42 gives repeatable results.
# stratify=y maintains class proportions in both sets.

# make_pipeline(...)
# Creates a pipeline for scaling and classification.

# StandardScaler()
# Standardizes the pixel features before classification.

# SVC(kernel="rbf", C=10, gamma="scale")
# Creates an SVM classifier using the RBF kernel.
# C controls the penalty for classification errors.
# gamma controls the influence of individual training samples.

# model.fit(X_train, y_train)
# Trains the SVM model using training images and labels.

# model.predict(X_test)
# Predicts the digit labels for testing images.

# accuracy_score(y_test, y_pred)
# Calculates the percentage of correctly classified digits.

# confusion_matrix(...)
# Displays the number of correct and incorrect predictions
# for each digit class.

# classification_report(...)
# Displays precision, recall and F1-score for digits 0 to 9.

# y_test[:20]
# Displays the first 20 actual digit labels.

# y_pred[:20]
# Displays the first 20 predicted digit labels.

# X_test[0].reshape(8, 8)
# Converts the first test image into its original 8 by 8 shape.

# plt.imshow(...)
# Displays the selected test image.

# plt.title(...)
# Displays the predicted digit above the image.

# plt.show()
# Displays the final digit classification result.