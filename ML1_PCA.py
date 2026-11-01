# AIM: To use PCA Algorithm for Dimensionality Reduction
# Using the Wine Dataset.

# Installation commands:
# Windows:
# python -m pip install pandas scikit-learn matplotlib
# python practical1_pca.py
#
# Ubuntu/Linux:
# pip install pandas scikit-learn matplotlib
# python3 -m pip install pandas scikit-learn matplotlib
# python3 practical1_pca.py


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("Wine.csv")

print("Original Dataset:")
print(df.head())

target_column = None

for column in ["Customer_Segment", "Wine", "target", "Class"]:
    if column in df.columns:
        target_column = column
        break

if target_column is not None:
    y = df[target_column]
    X = df.drop(columns=[target_column])
else:
    y = None
    X = df.copy()

X = X.select_dtypes(include=["number"])

X = X.fillna(X.mean())

print("\nOriginal Number of Features:", X.shape[1])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=["Principal Component 1", "Principal Component 2"]
)

if y is not None:
    pca_df["Wine Class"] = y.values

print("\nTransformed Dataset:")
print(pca_df.head())

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print(
    "\nTotal Variance Captured:",
    round(sum(pca.explained_variance_ratio_) * 100, 2),
    "%"
)

plt.figure(figsize=(8, 6))

if y is not None:
    for wine_class in sorted(y.unique()):
        data = pca_df[pca_df["Wine Class"] == wine_class]

        plt.scatter(
            data["Principal Component 1"],
            data["Principal Component 2"],
            label=str(wine_class)
        )

    plt.legend(title="Wine Class")

else:
    plt.scatter(
        pca_df["Principal Component 1"],
        pca_df["Principal Component 2"]
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Wine Dataset")
plt.grid(True)
plt.show()

# ============================================================
# AIM:
# Implement Principal Component Analysis (PCA) on the Wine Dataset
# to Reduce the Number of Features and Visualize the Data.
# ============================================================


# ============================================================
# 1. INSTALLATION AND EXECUTION COMMANDS
# ============================================================

# Windows:
# python -m pip install pandas matplotlib scikit-learn
# python practical_pca.py

# Ubuntu/Linux:
# pip install pandas matplotlib scikit-learn
# python3 -m pip install pandas matplotlib scikit-learn
# python3 practical_pca.py


# ============================================================
# 2. IMPORTING LIBRARIES
# ============================================================

# import pandas as pd
# Imports the Pandas library.
# Pandas is used to read, store, and process datasets.
# pd is the short name used for Pandas.

# import matplotlib.pyplot as plt
# Imports Matplotlib for creating graphs and visualizations.
# plt is the short name used for pyplot.

# from sklearn.preprocessing import StandardScaler
# Imports StandardScaler.
# It standardizes numerical features.
# It brings features to a similar scale.

# from sklearn.decomposition import PCA
# Imports PCA from Scikit-learn.
# PCA stands for Principal Component Analysis.
# It is used to reduce the number of input features.
# It converts the original features into new principal components.


# ============================================================
# 3. LOADING THE WINE DATASET
# ============================================================

# df = pd.read_csv("Wine.csv")
# Reads the Wine.csv dataset using Pandas.
# The dataset is stored in a DataFrame named df.
# A DataFrame is a table containing rows and columns.
# This practical uses the local CSV file.

# print("Original Dataset:")
# Displays the original dataset heading.

# print(df.head())
# Displays the first five rows of the dataset.
# It helps us understand the dataset structure.
# We can see the column names and sample values.


# ============================================================
# 4. IDENTIFYING THE TARGET COLUMN
# ============================================================

# target_column = None
# Creates a variable to store the target column name.
# None means that no target column has been selected yet.

# for column in ["Customer_Segment", "Wine", "target", "Class"]:
# The for loop checks the dataset for possible target column names.
# It checks these names one by one:
# Customer_Segment, Wine, target, and Class.

#     if column in df.columns:
# Checks whether the current column name exists in the dataset.
# df.columns contains all column names.

#         target_column = column
# Stores the matching column name in target_column.

#         break
# Stops the loop after finding the first matching column.
# It avoids checking unnecessary column names.


# ============================================================
# 5. SEPARATING INPUT FEATURES AND TARGET
# ============================================================

# if target_column is not None:
# Checks whether a target column was found.

#     y = df[target_column]
# Stores the target column in y.
# y contains the class or category labels.

#     X = df.drop(columns=[target_column])
# Creates the input feature dataset.
# drop() removes the target column from the input data.
# The remaining columns are used for PCA.

# else:
# Runs when none of the specified target columns is found.

#     y = None
# Sets y to None because no target column is available.

#     X = df.copy()
# Copies the complete dataset into X.
# All columns are considered input features.


# ============================================================
# 6. SELECTING NUMERICAL FEATURES
# ============================================================

# X = X.select_dtypes(include=["number"])
# Selects only numerical columns from the input dataset.
# PCA works with numerical data.
# Text or categorical columns are excluded.
# The selected numerical columns are stored in X.


# ============================================================
# 7. HANDLING MISSING VALUES
# ============================================================

# X = X.fillna(X.mean())
# fillna() replaces missing values in numerical columns.
# X.mean() calculates the average of each numerical column.
# Each missing value is replaced with the mean of its column.
# This helps prevent errors during PCA.


# ============================================================
# 8. DISPLAYING THE ORIGINAL NUMBER OF FEATURES
# ============================================================

# print("\nOriginal Number of Features:", X.shape[1])
# X.shape gives the dimensions of the dataset.
# X.shape[1] gives the number of columns or features.
# It displays how many numerical features were originally present.
# \n moves the output to a new line.


# ============================================================
# 9. STANDARDIZING THE DATA
# ============================================================

# scaler = StandardScaler()
# Creates a StandardScaler object.
# It is used to standardize the input features.

# X_scaled = scaler.fit_transform(X)
# fit() calculates the mean and standard deviation of each feature.
# transform() standardizes the feature values.
# fit_transform() performs both operations together.

# Standardization makes the features have approximately:
# Mean = 0
# Standard Deviation = 1

# It is important because different Wine features
# may have different measurement scales.
# Standardization prevents large-scale features
# from dominating the PCA calculation.


# ============================================================
# 10. CREATING THE PCA MODEL
# ============================================================

# pca = PCA(n_components=2)
# Creates a PCA model.
# n_components=2 means that PCA will retain two principal components.
# These components are new features created from the original features.
# They are called Principal Component 1 and Principal Component 2.

# PCA selects directions that capture the maximum possible variance.
# The first principal component captures the highest variance.
# The second principal component captures the next highest variance.
# The second component is perpendicular to the first component.


# ============================================================
# 11. APPLYING PCA
# ============================================================

# X_pca = pca.fit_transform(X_scaled)
# fit() learns the principal components from standardized data.
# transform() converts the original features into principal components.
# fit_transform() performs both operations together.
# X_pca stores the reduced dataset.
# Each original record is now represented using two values.
# The number of features is reduced to two.


# ============================================================
# 12. CREATING THE TRANSFORMED DATAFRAME
# ============================================================

# pca_df = pd.DataFrame(
#     X_pca,
#     columns=["Principal Component 1", "Principal Component 2"]
# )

# pd.DataFrame() creates a new DataFrame.
# X_pca contains the transformed PCA values.

# columns assigns names to the two new features.
# Principal Component 1 represents the first new feature.
# Principal Component 2 represents the second new feature.

# pca_df stores the reduced dataset in table format.


# ============================================================
# 13. ADDING THE TARGET COLUMN IF AVAILABLE
# ============================================================

# if y is not None:
# Checks whether a target column was found in the original dataset.

#     pca_df["Wine Class"] = y.values
# Adds the original target labels to the transformed dataset.
# y.values extracts the target values as an array.
# Wine Class stores the class name for each wine record.
# This column is used to display different classes in the graph.


# ============================================================
# 14. DISPLAYING THE TRANSFORMED DATASET
# ============================================================

# print("\nTransformed Dataset:")
# Displays the transformed dataset heading.

# print(pca_df.head())
# Displays the first five rows of the reduced dataset.
# It shows Principal Component 1 and Principal Component 2.
# If a target column is available, Wine Class is also displayed.


# ============================================================
# 15. DISPLAYING EXPLAINED VARIANCE RATIO
# ============================================================

# print("\nExplained Variance Ratio:")
# Displays the explained variance ratio heading.

# print(pca.explained_variance_ratio_)
# Displays the variance captured by each principal component.
# It gives the proportion of total variance represented
# by each principal component.

# For example:
# [0.36, 0.19]
# The first component captures 36% of the variance.
# The second component captures 19% of the variance.
# These are only example values; actual values depend on the dataset.


# ============================================================
# 16. CALCULATING TOTAL VARIANCE CAPTURED
# ============================================================

# print(
#     "\nTotal Variance Captured:",
#     round(sum(pca.explained_variance_ratio_) * 100, 2),
#     "%"
# )

# pca.explained_variance_ratio_
# Returns the variance proportion captured by each component.

# sum()
# Adds the variance proportions of both components.

# Multiplying by 100 converts the value into a percentage.

# round(value, 2)
# Rounds the percentage to two decimal places.

# The final result shows how much information
# is retained by the two principal components.

# For example:
# If the result is 55.00%, the two components retain
# 55% of the total variance in the original features.


# ============================================================
# 17. CREATING THE PCA GRAPH
# ============================================================

# plt.figure(figsize=(8, 6))
# Creates a new graph.
# figsize=(8, 6) sets its width to 8 inches and height to 6 inches.


# ============================================================
# 18. PLOTTING DATA WHEN TARGET CLASSES ARE AVAILABLE
# ============================================================

# if y is not None:
# Checks whether the original dataset contains a target column.

#     for wine_class in sorted(y.unique()):
# The loop processes each unique wine class.
# y.unique() returns all different class labels.
# sorted() arranges the class labels in sorted order.

#         data = pca_df[pca_df["Wine Class"] == wine_class]
# Selects only the rows belonging to the current wine class.
# This allows each class to be plotted separately.

#         plt.scatter(
#             data["Principal Component 1"],
#             data["Principal Component 2"],
#             label=str(wine_class)
#         )

# plt.scatter() creates a scatter plot.
# The horizontal axis represents Principal Component 1.
# The vertical axis represents Principal Component 2.
# Each point represents one wine record.
# label=str(wine_class) assigns the class name to the plotted points.


# ============================================================
# 19. DISPLAYING THE GRAPH LEGEND
# ============================================================

#     plt.legend(title="Wine Class")
# Displays the legend on the graph.
# The legend identifies the different wine classes.
# Different groups of points can be distinguished using the legend.


# ============================================================
# 20. PLOTTING DATA WHEN NO TARGET CLASS IS AVAILABLE
# ============================================================

# else:
# Executes when the dataset does not contain a recognized target column.

#     plt.scatter(
#         pca_df["Principal Component 1"],
#         pca_df["Principal Component 2"]
#     )

# Displays all transformed records in one scatter plot.
# Each point represents one wine record.
# No class-based grouping is performed because no target is available.


# ============================================================
# 21. ADDING GRAPH LABELS AND TITLE
# ============================================================

# plt.xlabel("Principal Component 1")
# Sets the horizontal axis label.

# plt.ylabel("Principal Component 2")
# Sets the vertical axis label.

# plt.title("PCA - Wine Dataset")
# Sets the title of the graph.

# plt.grid(True)
# Displays grid lines in the graph.
# Grid lines make the graph easier to read.

# plt.show()
# Displays the final PCA scatter plot.


# ============================================================
# 22. UNDERSTANDING THE COMPLETE OUTPUT
# ============================================================

# Original Dataset:
# Displays the first five records of the original Wine dataset.

# Original Number of Features:
# Displays the number of numerical input features before PCA.

# Transformed Dataset:
# Displays the reduced dataset containing two principal components.

# Explained Variance Ratio:
# Shows how much variance is captured by each principal component.

# Total Variance Captured:
# Shows the combined percentage of variance retained by both components.

# PCA Graph:
# Displays the wine records using two principal components.

# Each point represents one wine record.
# Points that are close together have similar PCA coordinates.
# Different groups may indicate differences between wine classes.
# Overlapping groups indicate that the classes are not completely
# separated using only these two principal components.


# ============================================================
# 23. IMPORTANT PCA CONCEPTS FOR VIVA
# ============================================================

# Q1. What is PCA?
# PCA stands for Principal Component Analysis.
# It is an unsupervised dimensionality reduction technique.
# It reduces the number of features while retaining
# as much variation in the data as possible.

# Q2. What is dimensionality reduction?
# Dimensionality reduction means reducing the number of input features.
# It makes complex datasets easier to visualize and analyze.

# Q3. Why is PCA used?
# PCA is used to reduce the number of features.
# It helps visualize high-dimensional data.
# It can reduce storage and computation requirements.
# It may also remove some redundant information.

# Q4. What is a principal component?
# A principal component is a new feature created
# by combining the original input features.
# It represents a direction of maximum variance in the data.

# Q5. What is the first principal component?
# It is the direction that captures the maximum variance
# in the original standardized dataset.

# Q6. What is the second principal component?
# It captures the next highest possible variance.
# It is perpendicular to the first principal component.

# Q7. Why do we use StandardScaler before PCA?
# Different features may have different measurement scales.
# StandardScaler brings them to a similar scale.
# This prevents features with large numerical values
# from dominating the PCA calculation.

# Q8. What is explained variance?
# Explained variance tells us how much variation
# in the original dataset is represented by a principal component.

# Q9. What is explained_variance_ratio_?
# It returns the proportion of total variance
# captured by each principal component.

# Q10. What does n_components=2 mean?
# It means that PCA reduces the dataset to two principal components.
# These two components can be displayed in a 2D graph.

# Q11. Is PCA supervised or unsupervised?
# PCA is an unsupervised learning technique.
# It does not use target labels when calculating principal components.

# Q12. What is the difference between original features and principal components?
# Original features are the columns in the original dataset.
# Principal components are new features created from combinations
# of the original numerical features.

# Q13. Are principal components independent?
# Principal components are uncorrelated with one another
# in the transformed dataset.

# Q14. What is variance?
# Variance measures how spread out the data values are
# from their mean.

# Q15. What does the PCA scatter plot show?
# It displays the dataset using two principal components.
# Each point represents one record.
# It helps us understand patterns and possible groups in the data.

# Q16. What is a scatter plot?
# A scatter plot displays data points using horizontal
# and vertical coordinates.
# It helps identify patterns and relationships.

# Q17. What happens if we increase n_components?
# More principal components are retained.
# More variance can generally be preserved.
# However, the dimensionality reduction becomes smaller.

# Q18. What are the advantages of PCA?
# It reduces the number of features.
# It helps visualize high-dimensional data.
# It can reduce redundant information.
# It may improve the efficiency of some machine learning models.

# Q19. What are the limitations of PCA?
# Some information may be lost during dimensionality reduction.
# Principal components may be difficult to interpret.
# PCA mainly captures linear patterns in the data.

# Q20. What are the applications of PCA?
# Image compression.
# Face recognition.
# Data visualization.
# Pattern recognition.
# Feature extraction.
# Data preprocessing in machine learning.


# ============================================================
# 24. CONCLUSION
# ============================================================

# In this practical, Principal Component Analysis is implemented
# on the Wine dataset.

# The dataset is loaded and numerical features are selected.
# Missing values are replaced using column means.
# StandardScaler is used to standardize the input features.
# PCA reduces the original features to two principal components.

# The explained variance ratio is calculated to understand
# how much information is retained by each component.

# Finally, a scatter plot is created to visualize the reduced data.

# Therefore, PCA helps reduce dataset dimensionality
# and makes high-dimensional data easier to visualize.

