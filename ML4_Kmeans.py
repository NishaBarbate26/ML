# AIM: Implement K-Means Clustering on Iris Dataset and
# Determine the Number of Clusters Using the Elbow Method.

# Installation commands:
# Windows:
# python -m pip install pandas numpy matplotlib scikit-learn
# python practical_kmeans.py
#
# Ubuntu/Linux:
# pip install pandas numpy matplotlib scikit-learn
# python3 -m pip install pandas numpy matplotlib scikit-learn
# python3 practical_kmeans.py


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load dataset

df = pd.read_csv("Iris.csv")

print("Original Iris Dataset:")
print(df.head())

# Remove unnecessary columns

df = df.drop(columns=["Id"], errors="ignore")

# Select numerical features

X = df.drop(columns=["Species"], errors="ignore")

# Handle missing values

X = X.fillna(X.mean())

# Standardize the data

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Apply Elbow Method

wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

# Plot Elbow Graph

plt.figure(figsize=(8, 5))

plt.plot(range(1, 11), wcss, marker="o")

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.grid(True)

plt.show()

# Apply K-Means with 3 clusters

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nK-Means Clustering Results:")

print(df[["Cluster"]].head(10))

print("\nNumber of Data Points in Each Cluster:")

print(df["Cluster"].value_counts().sort_index())

# Calculate Silhouette Score

score = silhouette_score(X_scaled, df["Cluster"])

print("\nSilhouette Score:", score)

# Reduce dimensions to 2D for visualization

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# Plot clusters

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["Cluster"],
    cmap="rainbow"
)

plt.title("K-Means Clustering on Iris Dataset")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.show()


# ============================================================
# EXPLANATION LINE BY LINE
# ============================================================

# import pandas as pd
# Imports Pandas to load and process the dataset.

# import matplotlib.pyplot as plt
# Imports Matplotlib to display graphs.

# KMeans
# Imports the K-Means clustering algorithm.

# StandardScaler
# Imports StandardScaler to standardize the features.

# PCA
# Imports Principal Component Analysis to reduce dimensions for plotting.

# silhouette_score
# Imports a metric to measure clustering quality.

# df = pd.read_csv("Iris.csv")
# Loads the Iris dataset from the local CSV file.

# df.head()
# Displays the first five records.

# df.drop(columns=["Id"], errors="ignore")
# Removes the unnecessary Id column if it exists.

# X = df.drop(columns=["Species"], errors="ignore")
# Selects numerical flower measurements for clustering.
# Species is excluded because K-Means is an unsupervised algorithm.

# X.fillna(X.mean())
# Replaces missing numerical values with column averages.

# scaler = StandardScaler()
# Creates an object for standardizing the features.

# scaler.fit_transform(X)
# Standardizes all numerical features.

# wcss = []
# Creates an empty list to store WCSS values.

# for i in range(1, 11):
# Tests cluster counts from 1 to 10.

# KMeans(n_clusters=i, random_state=42, n_init=10)
# Creates a K-Means model with i clusters.

# kmeans.fit(X_scaled)
# Trains the K-Means clustering model.

# kmeans.inertia_
# Calculates Within-Cluster Sum of Squares (WCSS).

# wcss.append(kmeans.inertia_)
# Stores the WCSS value for each cluster count.

# plt.plot(...)
# Draws the Elbow Method graph.

# plt.xlabel(...)
# Sets the horizontal axis label.

# plt.ylabel(...)
# Sets the vertical axis label.

# plt.show()
# Displays the Elbow Method graph.

# KMeans(n_clusters=3, ...)
# Creates the final K-Means model with three clusters.
# The Iris dataset contains three known species.

# kmeans.fit_predict(X_scaled)
# Trains the model and assigns a cluster number to every flower.

# df["Cluster"]
# Stores the assigned cluster number in the dataset.

# value_counts()
# Counts the number of flowers in each cluster.

# silhouette_score(...)
# Measures how well the data points fit their assigned clusters.
# A higher score generally indicates better-separated clusters.

# PCA(n_components=2)
# Creates a PCA model that reduces four features to two components.

# pca.fit_transform(X_scaled)
# Transforms the standardized data into two dimensions.

# plt.scatter(...)
# Displays the clusters using different colours.

# c=df["Cluster"]
# Assigns colours based on the cluster numbers.

# plt.title(...)
# Sets the graph title.

# plt.xlabel(...)
# Sets the horizontal axis label.

# plt.ylabel(...)
# Sets the vertical axis label.

# plt.show()
# Displays the final K-Means clustering graph.