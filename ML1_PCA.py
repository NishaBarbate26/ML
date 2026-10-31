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
# EXPLANATION LINE BY LINE
# ============================================================

# import pandas as pd
# Imports Pandas to load and process the CSV dataset.

# import matplotlib.pyplot as plt
# Imports Matplotlib to display the PCA graph.

# from sklearn.preprocessing import StandardScaler
# Imports StandardScaler to standardize the features.

# from sklearn.decomposition import PCA
# Imports PCA to reduce the number of features.

# df = pd.read_csv("Wine.csv")
# Loads the Wine dataset from the local CSV file.

# print(df.head())
# Displays the first five rows of the original dataset.

# target_column = None
# Creates a variable to store the name of the class column.

# for column in ["Customer_Segment", "Wine", "target", "Class"]:
# Checks common names used for the class column in the dataset.

# if column in df.columns:
# Checks whether the column exists in the dataset.

# target_column = column
# Stores the class column name when it is found.

# break
# Stops checking after finding the class column.

# if target_column is not None:
# Checks whether a class column was found.

# y = df[target_column]
# Stores the class labels separately.

# X = df.drop(columns=[target_column])
# Removes the class column from the input features.

# else:
# Runs when the dataset does not contain a recognized class column.

# y = None
# Indicates that class labels are not available.

# X = df.copy()
# Uses all dataset columns as input features.

# X = X.select_dtypes(include=["number"])
# Selects only numeric columns for PCA.

# X = X.fillna(X.mean())
# Replaces missing numeric values with their column averages.

# print(X.shape[1])
# Displays the original number of numeric features.

# scaler = StandardScaler()
# Creates an object to standardize the dataset.

# X_scaled = scaler.fit_transform(X)
# Converts features to a common scale.

# pca = PCA(n_components=2)
# Creates a PCA model that reduces the data to two components.

# X_pca = pca.fit_transform(X_scaled)
# Applies PCA and transforms the original features into two components.

# pca_df = pd.DataFrame(...)
# Stores the transformed data in a new DataFrame.

# if y is not None:
# Checks whether class labels are available.

# pca_df["Wine Class"] = y.values
# Adds the original class labels to the transformed dataset.

# print(pca_df.head())
# Displays the first five rows of the transformed dataset.

# print(pca.explained_variance_ratio_)
# Displays the percentage of variance explained by each component.

# sum(pca.explained_variance_ratio_) * 100
# Calculates the total percentage of variance captured by two components.

# plt.figure(figsize=(8, 6))
# Creates a figure for the PCA scatter plot.

# for wine_class in sorted(y.unique()):
# Iterates through the available wine classes.

# data = pca_df[pca_df["Wine Class"] == wine_class]
# Selects records belonging to the current class.

# plt.scatter(...)
# Plots the two principal components for that class.

# label=str(wine_class)
# Assigns the class name to the graph legend.

# plt.legend(title="Wine Class")
# Displays the legend showing the different classes.

# else:
# Runs when no class labels are available.

# plt.scatter(...)
# Displays the PCA points without class grouping.

# plt.xlabel(...)
# Sets the horizontal axis label.

# plt.ylabel(...)
# Sets the vertical axis label.

# plt.title("PCA - Wine Dataset")
# Sets the graph title.

# plt.grid(True)
# Displays grid lines on the graph.

# plt.show()
# Displays the final PCA graph.