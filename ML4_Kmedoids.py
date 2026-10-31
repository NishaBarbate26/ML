
# AIM: Implement K-Medoids Clustering on Credit Card Dataset
# and Determine the Number of Clusters Using Silhouette Method.

# Installation commands:
# Windows:
# python -m pip install pandas numpy matplotlib scikit-learn
# python ML4_Kmedoids.py
#
# Ubuntu/Linux:
# python3 -m pip install pandas numpy matplotlib scikit-learn
# python3 ML4_Kmedoids.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.decomposition import PCA


# K-Medoids implementation using NumPy and scikit-learn

def kmedoids_fit(X, n_clusters, max_iter=100, random_state=42):

    rng = np.random.default_rng(random_state)

    n_samples = len(X)

    medoids = rng.choice(n_samples, n_clusters, replace=False)

    distance_matrix = pairwise_distances(X)

    for iteration in range(max_iter):

        distances = distance_matrix[:, medoids]

        labels = np.argmin(distances, axis=1)

        new_medoids = medoids.copy()

        for cluster in range(n_clusters):

            members = np.where(labels == cluster)[0]

            if len(members) > 0:

                cluster_distances = distance_matrix[
                    np.ix_(members, members)
                ]

                costs = cluster_distances.sum(axis=1)

                new_medoids[cluster] = members[np.argmin(costs)]

            else:

                nearest_distances = distances.min(axis=1)

                used = set(new_medoids)

                candidates = np.argsort(nearest_distances)[::-1]

                for candidate in candidates:

                    if candidate not in used:
                        new_medoids[cluster] = candidate
                        break

        if np.array_equal(medoids, new_medoids):
            break

        medoids = new_medoids

    return X[medoids]


# Load dataset

df = pd.read_csv("CC GENERAL.csv")

print("Original Credit Card Dataset:")
print(df.head())

# Remove unnecessary customer ID column

df = df.drop(columns=["CUST_ID"], errors="ignore")

# Convert all columns into numeric values

df = df.apply(pd.to_numeric, errors="coerce")

# Replace missing values with column median

df = df.fillna(df.median())

# Standardize the dataset

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)

# Select a sample for K-Medoids training

sample_size = min(2000, len(X_scaled))

rng = np.random.default_rng(42)

sample_indices = rng.choice(
    len(X_scaled),
    sample_size,
    replace=False
)

X_sample = X_scaled[sample_indices]

# Apply Silhouette Method

silhouette_scores = []

K = range(2, 8)

for i in K:

    medoid_points = kmedoids_fit(
        X_sample,
        n_clusters=i,
        random_state=42
    )

    distances = pairwise_distances(
        X_sample,
        medoid_points
    )

    labels = np.argmin(distances, axis=1)

    score = silhouette_score(
        X_sample,
        labels
    )

    silhouette_scores.append(score)

    print("Clusters:", i, "Silhouette Score:", score)

# Plot Silhouette Scores

plt.figure(figsize=(8, 5))

plt.plot(
    list(K),
    silhouette_scores,
    marker="o"
)

plt.title("Silhouette Method")

plt.xlabel("Number of Clusters")

plt.ylabel("Silhouette Score")

plt.grid(True)

plt.show()

# Find the best number of clusters

best_k = list(K)[silhouette_scores.index(max(silhouette_scores))]

print("\nBest Number of Clusters:", best_k)

# Apply final K-Medoids model

final_medoids = kmedoids_fit(
    X_sample,
    n_clusters=best_k,
    random_state=42
)

# Assign all customers to their nearest medoid

final_distances = pairwise_distances(
    X_scaled,
    final_medoids
)

df["Cluster"] = np.argmin(final_distances, axis=1)

print("\nK-Medoids Clustering Results:")

print(df[["Cluster"]].head(10))

print("\nNumber of Customers in Each Cluster:")

print(df["Cluster"].value_counts().sort_index())

# Calculate final Silhouette Score

final_score = silhouette_score(
    X_scaled,
    df["Cluster"],
    sample_size=min(2000, len(X_scaled)),
    random_state=42
)

print("\nFinal Silhouette Score:", final_score)

# Reduce dimensions for visualization

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# Plot clusters

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["Cluster"],
    cmap="rainbow",
    s=10
)

plt.title("K-Medoids Clustering on Credit Card Dataset")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.show()


# ============================================================
# EXPLANATION LINE BY LINE
# ============================================================

# import pandas as pd
# Imports Pandas to load and process the credit card dataset.

# import numpy as np
# Imports NumPy for numerical calculations.

# import matplotlib.pyplot as plt
# Imports Matplotlib to display graphs.

# StandardScaler
# Standardizes numerical features.

# silhouette_score
# Evaluates the quality of clustering.

# pairwise_distances
# Calculates distances between data points.

# PCA
# Reduces dataset dimensions for visualization.

# kmedoids_fit()
# Implements the K-Medoids clustering algorithm.

# np.random.default_rng()
# Generates random numbers for selecting initial medoids.

# rng.choice()
# Selects initial medoids randomly.

# pairwise_distances()
# Calculates distances between all sample data points.

# np.argmin()
# Finds the nearest medoid for each data point.

# np.where()
# Finds the data points belonging to each cluster.

# distance_matrix
# Stores distances between data points.

# cluster_distances.sum()
# Calculates the total distance of each possible medoid
# from all points in its cluster.

# new_medoids
# Stores the updated representative points.

# return X[medoids]
# Returns the selected medoid points.

# pd.read_csv()
# Loads the credit card dataset.

# df.drop()
# Removes the customer ID column.

# pd.to_numeric()
# Converts columns into numeric values.

# df.fillna()
# Replaces missing values with column medians.

# StandardScaler()
# Creates the standardization object.

# fit_transform()
# Standardizes the dataset.

# sample_indices
# Selects a maximum of 2000 records for clustering.

# K = range(2, 8)
# Tests cluster counts from 2 to 7.

# silhouette_scores
# Stores Silhouette Scores for different cluster counts.

# silhouette_score()
# Measures how well-separated the clusters are.

# best_k
# Stores the cluster count with the highest Silhouette Score.

# final_medoids
# Stores the medoids of the final clustering model.

# final_distances
# Calculates distances between all customers and final medoids.

# df["Cluster"]
# Stores the cluster assigned to each customer.

# value_counts()
# Counts the number of customers in each cluster.

# PCA(n_components=2)
# Reduces the dataset to two principal components.

# pca.fit_transform()
# Transforms the dataset into two dimensions.

# plt.scatter()
# Displays the clusters using a scatter plot.

# plt.show()
# Displays the graph.
