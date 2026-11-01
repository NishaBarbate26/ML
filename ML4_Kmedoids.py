
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
# AIM:
# Implement K-Medoids Clustering Algorithm on Credit Card Data
# and Find the Optimal Number of Clusters Using Silhouette Score.
# ============================================================


# ============================================================
# 1. INSTALLATION AND EXECUTION COMMANDS
# ============================================================

# Windows:
# python -m pip install pandas numpy matplotlib scikit-learn
# python practical_kmedoids.py

# Ubuntu/Linux:
# pip install pandas numpy matplotlib scikit-learn
# python3 -m pip install pandas numpy matplotlib scikit-learn
# python3 practical_kmedoids.py


# ============================================================
# 2. IMPORTING LIBRARIES
# ============================================================

# import pandas as pd
# Imports Pandas for loading, storing, and processing datasets.
# pd is the short name used for Pandas.

# import numpy as np
# Imports NumPy for numerical calculations and array operations.
# np is the short name used for NumPy.

# import matplotlib.pyplot as plt
# Imports Matplotlib for creating graphs and visualizations.
# plt is the short name used for pyplot.

# from sklearn.preprocessing import StandardScaler
# Imports StandardScaler to standardize the dataset.
# It brings numerical features to a comparable scale.

# from sklearn.metrics import silhouette_score, pairwise_distances
# silhouette_score() evaluates the quality of clustering.
# pairwise_distances() calculates distances between data points.

# from sklearn.decomposition import PCA
# Imports PCA (Principal Component Analysis).
# PCA reduces the number of dimensions for visualization.


# ============================================================
# 3. IMPLEMENTING THE K-MEDOIDS ALGORITHM
# ============================================================

# def kmedoids_fit(X, n_clusters, max_iter=100, random_state=42):
# Defines a function named kmedoids_fit().
# This function implements the K-Medoids clustering algorithm.
#
# X:
# Input dataset containing numerical features.
#
# n_clusters:
# Number of clusters to create.
#
# max_iter=100:
# Maximum number of iterations allowed.
#
# random_state=42:
# Controls random initialization so results are reproducible.


# ------------------------------------------------------------
# 3.1 INITIALIZING THE RANDOM NUMBER GENERATOR
# ------------------------------------------------------------

# rng = np.random.default_rng(random_state)
# Creates a NumPy random number generator.
# It uses random_state=42 to make random selection reproducible.


# ------------------------------------------------------------
# 3.2 FINDING THE NUMBER OF DATA POINTS
# ------------------------------------------------------------

# n_samples = len(X)
# Counts the total number of data points in the dataset.
# len(X) returns the number of rows in X.


# ------------------------------------------------------------
# 3.3 SELECTING INITIAL MEDOIDS
# ------------------------------------------------------------

# medoids = rng.choice(n_samples, n_clusters, replace=False)
# Randomly selects data point indices as initial medoids.
#
# A medoid is an actual data point from the dataset.
# It represents the center of its cluster.
#
# n_samples specifies the available data points.
# n_clusters specifies how many medoids to select.
# replace=False ensures that the same point is not selected twice.


# ------------------------------------------------------------
# 3.4 CALCULATING THE DISTANCE MATRIX
# ------------------------------------------------------------

# distance_matrix = pairwise_distances(X)
# Calculates the distance between every pair of data points.
# Stores all distances in a distance matrix.
#
# Example:
# If the dataset contains 100 rows,
# the distance matrix contains 100 x 100 distance values.
#
# The distance matrix helps the algorithm identify
# which data points are closest to each other.


# ------------------------------------------------------------
# 3.5 STARTING THE ITERATION LOOP
# ------------------------------------------------------------

# for iteration in range(max_iter):
# Starts the main K-Medoids iteration loop.
# The algorithm can run a maximum of 100 iterations.
# It updates the medoids until they stop changing
# or the maximum iteration limit is reached.


# ------------------------------------------------------------
# 3.6 CALCULATING DISTANCES FROM MEDOIDS
# ------------------------------------------------------------

# distances = distance_matrix[:, medoids]
# Selects the distances between every data point
# and the currently selected medoids.
#
# The colon selects all rows.
# medoids selects the columns corresponding to medoid indices.
#
# The resulting matrix contains the distance of each point
# from every medoid.


# ------------------------------------------------------------
# 3.7 ASSIGNING DATA POINTS TO CLUSTERS
# ------------------------------------------------------------

# labels = np.argmin(distances, axis=1)
# Assigns every data point to its nearest medoid.
#
# argmin() finds the position of the minimum value.
# axis=1 performs this operation row by row.
#
# Each data point receives a cluster label.
#
# Example:
# Label 0 means the point belongs to Cluster 0.
# Label 1 means the point belongs to Cluster 1.
# Label 2 means the point belongs to Cluster 2.


# ------------------------------------------------------------
# 3.8 CREATING A COPY OF THE CURRENT MEDOIDS
# ------------------------------------------------------------

# new_medoids = medoids.copy()
# Creates a copy of the existing medoids.
# The new medoid positions will be stored in this array.


# ------------------------------------------------------------
# 3.9 UPDATING EACH CLUSTER'S MEDOID
# ------------------------------------------------------------

# for cluster in range(n_clusters):
# Loops through every cluster.
# Each cluster's medoid is updated separately.


# ------------------------------------------------------------
# 3.10 FINDING THE MEMBERS OF A CLUSTER
# ------------------------------------------------------------

# members = np.where(labels == cluster)[0]
# Finds the indices of all data points assigned to the current cluster.
#
# labels == cluster checks which points belong to that cluster.
# np.where() returns the positions of matching points.
# [0] extracts the array of row indices.


# ------------------------------------------------------------
# 3.11 CHECKING WHETHER THE CLUSTER HAS DATA POINTS
# ------------------------------------------------------------

# if len(members) > 0:
# Checks whether the current cluster contains any data points.
# If it contains points, the algorithm finds a new medoid.


# ------------------------------------------------------------
# 3.12 CALCULATING DISTANCES WITHIN THE CLUSTER
# ------------------------------------------------------------

# cluster_distances = distance_matrix[
#     np.ix_(members, members)
# ]

# Extracts distances between all pairs of points
# belonging to the current cluster.
#
# np.ix_() selects the required rows and columns
# using the member indices.
#
# This creates a smaller distance matrix for that cluster.


# ------------------------------------------------------------
# 3.13 CALCULATING THE TOTAL DISTANCE COST
# ------------------------------------------------------------

# costs = cluster_distances.sum(axis=1)
# Calculates the total distance from each cluster member
# to all other members of the same cluster.
#
# sum(axis=1) adds the distances row by row.
#
# A smaller total distance means that the point is
# closer to the other points in its cluster.


# ------------------------------------------------------------
# 3.14 SELECTING THE NEW MEDOID
# ------------------------------------------------------------

# new_medoids[cluster] = members[np.argmin(costs)]
# Finds the point with the minimum total distance cost.
# That point becomes the new medoid of the cluster.
#
# Unlike K-Means, K-Medoids selects an actual data point
# as the cluster representative.


# ------------------------------------------------------------
# 3.15 HANDLING AN EMPTY CLUSTER
# ------------------------------------------------------------

# else:
# Executes when the current cluster has no assigned data points.
# Such a cluster is called an empty cluster.


# nearest_distances = distances.min(axis=1)
# Finds the minimum distance of each data point
# from its currently assigned nearest medoid.


# used = set(new_medoids)
# Creates a set containing the medoid indices already selected.
# It helps prevent selecting the same medoid again.


# candidates = np.argsort(nearest_distances)[::-1]
# Sorts data point indices according to their nearest-medoid distance.
#
# np.argsort() returns indices in ascending order.
# [::-1] reverses the order.
# Therefore, points with the largest nearest-medoid distance
# are considered first.


# for candidate in candidates:
# Checks the candidate data points one by one.


# if candidate not in used:
# Checks whether the candidate is not already a selected medoid.


# new_medoids[cluster] = candidate
# Assigns the candidate as the new medoid for the empty cluster.


# break
# Stops the candidate-selection loop after finding an unused point.


# ------------------------------------------------------------
# 3.16 CHECKING CONVERGENCE
# ------------------------------------------------------------

# if np.array_equal(medoids, new_medoids):
# Checks whether the old medoids and new medoids are identical.
# If they are identical, the algorithm has converged.
# This means the medoids are no longer changing.


# break
# Stops the main iteration loop when the medoids stop changing.


# medoids = new_medoids
# Updates the current medoids with the newly calculated medoids.
# The next iteration uses these updated medoids.


# ------------------------------------------------------------
# 3.17 RETURNING THE FINAL MEDOIDS
# ------------------------------------------------------------

# return X[medoids]
# Returns the actual data points selected as final medoids.
# X[medoids] selects the rows corresponding to the medoid indices.
#
# The function returns medoid points, not cluster labels.


# ============================================================
# 4. LOADING THE CREDIT CARD DATASET
# ============================================================

# df = pd.read_csv("CC GENERAL.csv")
# Loads the credit card dataset from the CSV file.
# The data is stored in the DataFrame named df.

# print("Original Credit Card Dataset:")
# Displays the heading for the original dataset.

# print(df.head())
# Displays the first five records of the dataset.
# It helps inspect the dataset before processing.


# ============================================================
# 5. REMOVING THE CUSTOMER ID COLUMN
# ============================================================

# df = df.drop(columns=["CUST_ID"], errors="ignore")
# Removes the CUST_ID column from the dataset.
# CUST_ID is only a customer identification number.
# It does not represent useful customer behavior information.
# errors="ignore" prevents an error if the column is absent.


# ============================================================
# 6. CONVERTING ALL COLUMNS INTO NUMERIC VALUES
# ============================================================

# df = df.apply(pd.to_numeric, errors="coerce")
# Converts all dataset columns into numeric values.
#
# apply() applies the conversion to every column.
# pd.to_numeric() converts values into numbers.
# errors="coerce" converts invalid values into NaN.
#
# NaN means Not a Number.
# It represents a missing or invalid numerical value.


# ============================================================
# 7. HANDLING MISSING VALUES
# ============================================================

# df = df.fillna(df.median())
# Replaces missing values with the median of their respective columns.
#
# df.median() calculates the median for each numerical column.
# fillna() replaces missing values using those median values.
#
# Median is the middle value after sorting the data.
# It is less affected by extreme values than the mean.


# ============================================================
# 8. STANDARDIZING THE DATASET
# ============================================================

# scaler = StandardScaler()
# Creates a StandardScaler object.

# X_scaled = scaler.fit_transform(df)
# Calculates the mean and standard deviation of each feature.
# Then transforms the dataset into standardized values.
#
# Standardization formula:
# Z = (X - Mean) / Standard Deviation
#
# After standardization:
# Each feature has approximately zero mean
# and unit standard deviation.
#
# Standardization is important because K-Medoids
# uses distance calculations.
#
# Without standardization, features with large values
# may dominate the distance calculation.


# ============================================================
# 9. SELECTING A SAMPLE FOR K-MEDOIDS TRAINING
# ============================================================

# sample_size = min(2000, len(X_scaled))
# Selects the sample size for clustering.
#
# If the dataset contains more than 2000 rows,
# the sample size will be 2000.
#
# If the dataset contains fewer than 2000 rows,
# all available rows will be selected.
#
# This reduces computation time.


# rng = np.random.default_rng(42)
# Creates a random number generator.
# random_state=42 makes the random selection reproducible.


# sample_indices = rng.choice(
#     len(X_scaled),
#     sample_size,
#     replace=False
# )

# Randomly selects row indices from the standardized dataset.
# len(X_scaled) represents the total number of rows.
# sample_size represents the number of rows to select.
# replace=False prevents selecting the same row more than once.


# X_sample = X_scaled[sample_indices]
# Selects the rows using the randomly generated indices.
# Stores the selected sample in X_sample.
#
# K-Medoids is applied to this sample to reduce computation time.


# ============================================================
# 10. APPLYING THE SILHOUETTE METHOD
# ============================================================

# silhouette_scores = []
# Creates an empty list to store silhouette scores
# for different numbers of clusters.


# K = range(2, 8)
# Defines the number of clusters to test.
# The algorithm checks K values 2, 3, 4, 5, 6, and 7.
#
# At least two clusters are required to calculate
# a meaningful silhouette score.


# for i in K:
# Starts a loop to test each possible cluster count.
# i represents the current number of clusters.


# medoid_points = kmedoids_fit(
#     X_sample,
#     n_clusters=i,
#     random_state=42
# )

# Calls the K-Medoids function for the current number of clusters.
# X_sample is the sample dataset.
# n_clusters=i specifies the current cluster count.
# The returned medoid points are stored in medoid_points.


# distances = pairwise_distances(
#     X_sample,
#     medoid_points
# )

# Calculates the distance between every sample point
# and each selected medoid.
# The result contains the distance of each point
# from every cluster representative.


# labels = np.argmin(distances, axis=1)
# Assigns each sample point to its nearest medoid.
# The result is a cluster label for each sample point.


# score = silhouette_score(
#     X_sample,
#     labels
# )

# Calculates the Silhouette Score for the current clustering.
#
# The Silhouette Score measures how well data points
# fit within their own clusters compared with other clusters.
#
# Its value generally ranges from -1 to +1.
#
# A score close to +1 means clusters are well separated.
# A score close to 0 means clusters may overlap.
# A negative score may indicate incorrect cluster assignments.


# silhouette_scores.append(score)
# Adds the current silhouette score to the list.


# print("Clusters:", i, "Silhouette Score:", score)
# Displays the current number of clusters and its score.


# ============================================================
# 11. PLOTTING SILHOUETTE SCORES
# ============================================================

# plt.figure(figsize=(8, 5))
# Creates a graph with width 8 inches and height 5 inches.


# plt.plot(
#     list(K),
#     silhouette_scores,
#     marker="o"
# )

# Plots the number of clusters against their silhouette scores.
# list(K) contains the cluster counts from 2 to 7.
# silhouette_scores contains the corresponding scores.
# marker="o" displays a circular marker at each data point.


# plt.title("Silhouette Method")
# Sets the title of the graph.


# plt.xlabel("Number of Clusters")
# Labels the horizontal axis.
# It represents the number of clusters.


# plt.ylabel("Silhouette Score")
# Labels the vertical axis.
# It represents the clustering quality score.


# plt.grid(True)
# Displays grid lines to make the graph easier to read.


# plt.show()
# Displays the silhouette score graph.


# ============================================================
# 12. FINDING THE BEST NUMBER OF CLUSTERS
# ============================================================

# best_k = list(K)[silhouette_scores.index(max(silhouette_scores))]
# Finds the cluster count with the highest silhouette score.
#
# max(silhouette_scores) finds the maximum score.
# index() finds the position of that score in the list.
# list(K) converts the range into a list of cluster counts.
# The cluster count at that position is stored in best_k.
#
# The highest silhouette score is selected
# as the best cluster count among the tested values.


# print("\nBest Number of Clusters:", best_k)
# Displays the selected number of clusters.


# ============================================================
# 13. APPLYING THE FINAL K-MEDOIDS MODEL
# ============================================================

# final_medoids = kmedoids_fit(
#     X_sample,
#     n_clusters=best_k,
#     random_state=42
# )

# Runs K-Medoids again using the selected best number of clusters.
# X_sample is used as the training sample.
# best_k is the selected number of clusters.
# The final medoid points are stored in final_medoids.


# ============================================================
# 14. ASSIGNING ALL CUSTOMERS TO THEIR NEAREST MEDOID
# ============================================================

# final_distances = pairwise_distances(
#     X_scaled,
#     final_medoids
# )

# Calculates distances between every customer
# in the complete dataset and the final medoids.
#
# Unlike the earlier sample-based calculations,
# this step uses the complete standardized dataset.


# df["Cluster"] = np.argmin(final_distances, axis=1)
# Assigns each customer to the nearest final medoid.
# argmin() finds the position of the minimum distance.
# axis=1 performs the operation for every customer.
#
# The resulting cluster numbers are stored
# in a new Cluster column in the DataFrame.


# ============================================================
# 15. DISPLAYING CLUSTERING RESULTS
# ============================================================

# print("\nK-Medoids Clustering Results:")
# Displays the clustering results heading.


# print(df[["Cluster"]].head(10))
# Displays the cluster numbers of the first ten customers.
# Each customer is assigned to one cluster.


# ============================================================
# 16. COUNTING CUSTOMERS IN EACH CLUSTER
# ============================================================

# print("\nNumber of Customers in Each Cluster:")
# Displays the cluster count heading.


# print(df["Cluster"].value_counts().sort_index())
# Counts the number of customers assigned to each cluster.
#
# value_counts() counts the records in every cluster.
# sort_index() displays the clusters in numerical order.
#
# This helps understand how customers are distributed
# among the clusters.


# ============================================================
# 17. CALCULATING THE FINAL SILHOUETTE SCORE
# ============================================================

# final_score = silhouette_score(
#     X_scaled,
#     df["Cluster"],
#     sample_size=min(2000, len(X_scaled)),
#     random_state=42
# )

# Calculates the final silhouette score for the clustering.
#
# X_scaled contains the complete standardized dataset.
# df["Cluster"] contains the cluster labels for all customers.
#
# sample_size limits the number of samples used
# for silhouette score calculation.
#
# min(2000, len(X_scaled)) ensures that the sample size
# does not exceed the number of available records.
#
# random_state=42 makes the internal sample selection reproducible.


# print("\nFinal Silhouette Score:", final_score)
# Displays the final silhouette score.
# A higher score generally indicates better-separated clusters.


# ============================================================
# 18. REDUCING DIMENSIONS USING PCA
# ============================================================

# pca = PCA(n_components=2)
# Creates a PCA object.
# n_components=2 means the dataset will be reduced to two dimensions.
#
# PCA stands for Principal Component Analysis.
# It is a dimensionality reduction technique.
#
# It converts the original features into new features
# called principal components.


# X_pca = pca.fit_transform(X_scaled)
# Fits PCA on the standardized dataset.
# Transforms the original features into two principal components.
#
# X_pca[:, 0] contains the first principal component.
# X_pca[:, 1] contains the second principal component.
#
# These two components are used to visualize
# high-dimensional customer data in a two-dimensional graph.
#
# PCA is used here for visualization.
# It does not perform the K-Medoids clustering itself.


# ============================================================
# 19. VISUALIZING THE CLUSTERS
# ============================================================

# plt.figure(figsize=(8, 6))
# Creates a new graph with width 8 inches and height 6 inches.


# plt.scatter(
#     X_pca[:, 0],
#     X_pca[:, 1],
#     c=df["Cluster"],
#     cmap="rainbow",
#     s=10
# )

# Creates a scatter plot of the customer clusters.
#
# X_pca[:, 0] represents the first principal component.
# X_pca[:, 1] represents the second principal component.
#
# c=df["Cluster"] assigns colors according to cluster labels.
# cmap="rainbow" uses different colors for different clusters.
# s=10 sets the size of each plotted point.
#
# Each point represents one customer.
# Points with the same color belong to the same cluster.


# plt.title("K-Medoids Clustering on Credit Card Dataset")
# Sets the graph title.


# plt.xlabel("Principal Component 1")
# Labels the horizontal axis as the first principal component.


# plt.ylabel("Principal Component 2")
# Labels the vertical axis as the second principal component.


# plt.show()
# Displays the final customer clustering graph.


# ============================================================
# 20. UNDERSTANDING THE COMPLETE OUTPUT
# ============================================================

# Original Credit Card Dataset:
# Displays the first five rows of the original dataset.

# Silhouette Scores:
# Displays the clustering quality for K values from 2 to 7.

# Silhouette Method Graph:
# Displays how the silhouette score changes
# with different numbers of clusters.

# Best Number of Clusters:
# Displays the K value with the highest silhouette score
# among the tested values.

# K-Medoids Clustering Results:
# Displays the cluster assigned to the first ten customers.

# Number of Customers in Each Cluster:
# Displays how many customers belong to each cluster.

# Final Silhouette Score:
# Displays the clustering quality calculated on a sample
# from the complete dataset.

# PCA Clustering Graph:
# Displays customers as points in two dimensions.
# Different colors represent different clusters.


# ============================================================
# 21. IMPORTANT K-MEDOIDS CONCEPTS FOR VIVA
# ============================================================

# Q1. What is K-Medoids?
# K-Medoids is an unsupervised machine learning clustering algorithm.
# It divides data points into K clusters using representative
# data points called medoids.

# Q2. What is a medoid?
# A medoid is an actual data point selected to represent a cluster.
# It is generally the point with the minimum total distance
# to the other points in that cluster.

# Q3. What is clustering?
# Clustering is the process of grouping similar data points together.
# It is an unsupervised learning technique.

# Q4. What is the value of K in this practical?
# The program tests K values from 2 to 7.
# It selects the value with the highest silhouette score.

# Q5. What is the difference between K-Means and K-Medoids?
# K-Means uses the mean of data points as its cluster center.
# K-Medoids uses an actual data point as its cluster center.
# K-Medoids is generally more resistant to outliers.

# Q6. Why is K-Medoids called an unsupervised algorithm?
# Because it groups data without using predefined class labels.

# Q7. What is a distance matrix?
# A distance matrix stores distances between pairs of data points.

# Q8. Why is StandardScaler used?
# It standardizes features to a comparable scale.
# This is important because clustering uses distance calculations.

# Q9. What is the Silhouette Score?
# It measures how well a data point fits into its own cluster
# compared with other clusters.

# Q10. What is the range of the Silhouette Score?
# It ranges from -1 to +1.

# Q11. What does a score close to +1 indicate?
# It indicates that clusters are well separated
# and data points are close to their own clusters.

# Q12. What does a score close to 0 indicate?
# It indicates that clusters may overlap.

# Q13. What does a negative silhouette score indicate?
# It may indicate that a data point is closer to another cluster
# than the cluster to which it has been assigned.

# Q14. Why is the Silhouette Method used?
# It helps select a suitable number of clusters.

# Q15. What is PCA?
# PCA stands for Principal Component Analysis.
# It reduces the number of dimensions in a dataset.

# Q16. Why is PCA used in this practical?
# It converts the dataset into two dimensions
# so that clusters can be displayed in a scatter plot.

# Q17. Does PCA perform clustering?
# No. PCA only reduces dimensions.
# K-Medoids performs the clustering.

# Q18. Why is CUST_ID removed?
# It is an identification column.
# It does not represent customer spending or behavior.

# Q19. Why are missing values replaced with the median?
# Median provides a central value and is less affected
# by extreme values than the mean.

# Q20. Why is a sample of 2000 records selected?
# K-Medoids requires many distance calculations.
# Sampling reduces computation time and memory usage.

# Q21. What does random_state=42 do?
# It makes random selection reproducible.

# Q22. What does max_iter=100 mean?
# It allows the algorithm to run a maximum of 100 iterations.

# Q23. What is convergence?
# Convergence occurs when the medoids stop changing
# between iterations.

# Q24. What is an empty cluster?
# It is a cluster that does not contain any assigned data points.

# Q25. What are the applications of K-Medoids?
# Customer segmentation.
# Credit card customer analysis.
# Market research.
# Grouping similar products.
# Document clustering.
# Medical data analysis.


# ============================================================
# 22. CONCLUSION
# ============================================================

# In this practical, the K-Medoids clustering algorithm
# is implemented using the credit card customer dataset.

# First, the dataset is loaded and the customer ID column is removed.
# Missing values are replaced with column medians.
# The data is converted into numerical form and standardized.

# A sample of up to 2000 records is selected to reduce computation.
# K-Medoids is tested with different numbers of clusters.

# The Silhouette Method is used to select the best cluster count.
# The final K-Medoids model assigns customers to clusters.

# The number of customers in each cluster is displayed.
# The final silhouette score evaluates clustering quality.

# Finally, PCA reduces the dataset to two dimensions
# and a scatter plot displays the customer clusters.

# Therefore, K-Medoids can be used for customer segmentation
# by grouping customers with similar credit card behavior.

