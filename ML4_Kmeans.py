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
# AIM:
# Implement K-Means Clustering on Iris Dataset and
# Determine the Number of Clusters Using the Elbow Method.
# ============================================================


# -------------------- INSTALLATION COMMANDS -----------------

# Windows:
# python -m pip install pandas numpy matplotlib scikit-learn
# python practical_kmeans.py

# Ubuntu/Linux:
# pip install pandas numpy matplotlib scikit-learn
# python3 -m pip install pandas numpy matplotlib scikit-learn


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

# import pandas as pd
# Imports the Pandas library.
# Pandas is used to read, store and manipulate tabular data.
# It helps us work with rows and columns of a dataset.
# pd is the short name used for Pandas.

# import matplotlib.pyplot as plt
# Imports the Pyplot module from Matplotlib.
# It is used to create graphs and visualizations.
# plt is the short name used for pyplot.

# from sklearn.cluster import KMeans
# Imports the K-Means clustering algorithm.
# K-Means is an unsupervised machine learning algorithm.
# It groups similar data points into clusters.

# from sklearn.preprocessing import StandardScaler
# Imports StandardScaler.
# It is used to standardize numerical features.
# It brings different features to a comparable scale.

# from sklearn.decomposition import PCA
# Imports Principal Component Analysis.
# PCA reduces the number of features in a dataset.
# Here, it reduces four features into two dimensions
# so that the clusters can be displayed on a 2D graph.

# from sklearn.metrics import silhouette_score
# Imports Silhouette Score.
# It measures how well the data points are grouped
# into their assigned clusters.


# ============================================================
# 2. LOAD THE IRIS DATASET
# ============================================================

# df = pd.read_csv("Iris.csv")
# Reads the Iris dataset from the local CSV file.
# The dataset is stored in a Pandas DataFrame named df.
# A DataFrame is a table containing rows and columns.
# Iris.csv should be present in the same folder as the Python file.

# print("Original Iris Dataset:")
# Displays the heading before showing the dataset.

# print(df.head())
# Displays the first five rows of the Iris dataset.
# head() is used to view the first five records by default.
# Each row represents one flower.
# Each column represents a feature or information about the flower.


# -------------------- IRIS DATASET INFORMATION ---------------

# The Iris dataset contains measurements of iris flowers.
# It contains 150 flower records.
# Each flower belongs to one of three species.

# The three species are:
# 1. Iris-setosa
# 2. Iris-versicolor
# 3. Iris-virginica

# The dataset generally contains the following columns:

# Id
# Represents the identification number of each flower.

# SepalLengthCm
# Represents the length of the sepal in centimeters.

# SepalWidthCm
# Represents the width of the sepal in centimeters.

# PetalLengthCm
# Represents the length of the petal in centimeters.

# PetalWidthCm
# Represents the width of the petal in centimeters.

# Species
# Represents the actual flower species.
# It contains the names Setosa, Versicolor and Virginica.

# Sepal
# The green, leaf-like part that protects the flower bud.

# Petal
# The colourful part of the flower.


# ============================================================
# 3. REMOVE UNNECESSARY COLUMNS
# ============================================================

# df = df.drop(columns=["Id"], errors="ignore")
# Removes the Id column from the dataset.
# Id is not useful for clustering because it is only an identification number.
# It does not describe the actual characteristics of a flower.

# columns=["Id"]
# Specifies the column that needs to be removed.

# errors="ignore"
# Prevents an error if the Id column does not exist.
# If the column is missing, the program continues normally.


# ============================================================
# 4. SELECT NUMERICAL FEATURES
# ============================================================

# X = df.drop(columns=["Species"], errors="ignore")
# Selects the numerical flower measurements for clustering.
# Removes the Species column from the input features.
# X contains the four numerical features:
# SepalLengthCm, SepalWidthCm, PetalLengthCm and PetalWidthCm.

# Why do we remove Species?
# K-Means is an unsupervised learning algorithm.
# Unsupervised learning works without using known output labels.
# Therefore, the actual Species column is not given to K-Means.
# The algorithm creates groups based on similarities
# between the numerical flower measurements.

# Important:
# Removing Species does not mean that the actual species
# information is permanently deleted from the original dataset.
# It is only excluded from the clustering input features.


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

# X = X.fillna(X.mean())
# Checks for missing values in the numerical features.
# Replaces missing values with the mean of their respective columns.

# fillna()
# fillna() is a Pandas function used to replace missing values.

# X.mean()
# Calculates the average value of each numerical column.

# Example:
# Suppose the petal lengths are 1, 2, missing, 4 and 5.
# Their average is 3.
# The missing value is replaced with 3.

# Why do we handle missing values?
# Machine learning algorithms may not work properly
# when the input dataset contains missing values.
# Replacing missing values helps prepare clean data
# for the clustering algorithm.


# ============================================================
# 6. STANDARDIZE THE DATA
# ============================================================

# scaler = StandardScaler()
# Creates an object of the StandardScaler class.
# This object is used to standardize the numerical features.

# X_scaled = scaler.fit_transform(X)
# Standardizes all four numerical features.
# First, fit() calculates the mean and standard deviation
# of each feature.
# Then, transform() uses these values to standardize the data.
# fit_transform() performs both operations together.

# Standardization formula:
# Z = (X - Mean) / Standard Deviation

# X
# Represents the original feature value.

# Mean
# Represents the average value of that feature.

# Standard Deviation
# Measures how much the values are spread around the mean.

# Z
# Represents the standardized feature value.

# After standardization:
# Each feature has approximately zero mean.
# Each feature has approximately unit standard deviation.

# Why is standardization important?
# The four Iris features may have different numerical ranges.
# K-Means uses distances to group similar data points.
# Features with larger numerical values may influence
# the distance calculation more than smaller-valued features.
# Standardization gives the features a comparable scale.

# Example:
# Sepal length may have values around 5.
# Petal width may have values around 1.
# Standardization prevents differences in their scales
# from unfairly affecting the clustering process.

# Important:
# X_scaled contains the standardized numerical data.
# K-Means uses X_scaled instead of the original X.


# ============================================================
# 7. APPLY THE ELBOW METHOD
# ============================================================

# wcss = []
# Creates an empty list named wcss.
# It stores the WCSS value for each tested number of clusters.

# What is WCSS?
# WCSS stands for Within-Cluster Sum of Squares.
# It measures how close the data points are to the
# center of their assigned clusters.
# The center of a cluster is called its centroid.

# Lower WCSS means that the data points are closer
# to their respective cluster centroids.
# WCSS generally decreases when the number of clusters increases.


# -------------------- LOOP FROM 1 TO 10 ----------------------

# for i in range(1, 11):
# Starts a loop to test different numbers of clusters.
# range(1, 11) generates the values 1, 2, 3, ..., 10.
# Therefore, the algorithm tests 10 different cluster counts.

# Why do we test different cluster counts?
# We do not initially know the best number of clusters.
# So, we test different values and compare their WCSS.

# Example:
# First iteration: 1 cluster.
# Second iteration: 2 clusters.
# Third iteration: 3 clusters.
# This continues until 10 clusters.


# -------------------- CREATE K-MEANS MODEL -------------------

# kmeans = KMeans(
#     n_clusters=i,
#     random_state=42,
#     n_init=10
# )
# Creates a K-Means clustering model for the current iteration.

# n_clusters=i
# Specifies the number of clusters to create.
# Its value changes in every loop iteration.
# For example, it becomes 1, then 2, then 3 and so on.

# random_state=42
# Controls the random initialization of cluster centroids.
# It helps produce repeatable results when the program runs again.

# n_init=10
# Runs the K-Means algorithm 10 times using different
# initial centroid positions.
# The algorithm selects the result with the lowest WCSS.
# This helps reduce the effect of poor random initialization.


# -------------------- TRAIN THE MODEL -----------------------

# kmeans.fit(X_scaled)
# Trains the K-Means algorithm using standardized data.
# The algorithm groups similar data points into clusters.
# It calculates the cluster centroids.
# It assigns each data point to its nearest centroid.

# What happens inside K-Means?
# Step 1: Select the number of clusters, K.
# Step 2: Initialize K cluster centroids.
# Step 3: Calculate the distance between each data point
#         and the cluster centroids.
# Step 4: Assign each data point to its nearest centroid.
# Step 5: Recalculate the centroid of every cluster.
# Step 6: Repeat the assignment and centroid calculation.
# Step 7: Stop when the clusters no longer change significantly
#         or the algorithm reaches its stopping condition.

# What is a centroid?
# A centroid is the center point of a cluster.
# It is calculated using the mean of all data points
# assigned to that cluster.

# What is Euclidean distance?
# It is the straight-line distance between two points.
# K-Means commonly uses Euclidean distance to assign
# data points to their nearest centroids.

# Euclidean distance formula:
# Distance = Square Root of the Sum of Squared Differences.


# -------------------- STORE WCSS VALUES ---------------------

# kmeans.inertia_
# Returns the WCSS value of the trained K-Means model.
# It calculates the sum of squared distances
# between each data point and its assigned centroid.

# wcss.append(kmeans.inertia_)
# Adds the current WCSS value to the wcss list.
# One WCSS value is stored for every tested cluster count.

# At the end of the loop:
# wcss contains 10 values.
# Each value represents the WCSS for 1 to 10 clusters.

# Example:
# wcss[0] represents WCSS for 1 cluster.
# wcss[1] represents WCSS for 2 clusters.
# wcss[2] represents WCSS for 3 clusters.
# This continues until wcss[9], which represents 10 clusters.


# ============================================================
# 8. PLOT THE ELBOW GRAPH
# ============================================================

# plt.figure(figsize=(8, 5))
# Creates a new graph with width 8 inches and height 5 inches.

# plt.plot(range(1, 11), wcss, marker="o")
# Draws the Elbow Method graph.
# The X-axis contains the number of clusters from 1 to 10.
# The Y-axis contains the corresponding WCSS values.
# marker="o" displays a circular marker at each data point.

# plt.title("Elbow Method")
# Sets the title of the graph.

# plt.xlabel("Number of Clusters")
# Sets the horizontal axis label.
# The horizontal axis represents the number of clusters.

# plt.ylabel("WCSS")
# Sets the vertical axis label.
# The vertical axis represents Within-Cluster Sum of Squares.

# plt.grid(True)
# Displays grid lines on the graph.
# Grid lines make it easier to read the graph.

# plt.show()
# Displays the Elbow Method graph.


# -------------------- UNDERSTANDING ELBOW GRAPH --------------

# What is the Elbow Method?
# The Elbow Method is a technique used to determine
# the suitable number of clusters in K-Means clustering.

# How does it work?
# 1. Run K-Means with different numbers of clusters.
# 2. Calculate WCSS for every cluster count.
# 3. Plot the number of clusters against WCSS.
# 4. Observe how WCSS decreases as clusters increase.
# 5. Find the point where the decrease becomes slower.
# 6. This point looks like an elbow in the graph.

# What is the elbow point?
# The elbow point is the point where adding more clusters
# does not produce a large improvement in WCSS.
# It helps us choose a suitable number of clusters.

# Why does WCSS decrease?
# When we increase the number of clusters,
# data points can be grouped into smaller and closer groups.
# Therefore, their distances from their centroids decrease.

# Why not always choose 10 clusters?
# More clusters generally produce lower WCSS.
# However, too many clusters make the grouping unnecessarily complex.
# The Elbow Method helps balance cluster quality and simplicity.

# Important for this practical:
# The Iris dataset contains three known flower species.
# We use three clusters in the final model.
# The Elbow Graph helps us examine whether three clusters
# are a reasonable choice for this dataset.

# Note:
# The elbow may not always be perfectly clear.
# The suitable number of clusters is selected by observing
# the shape of the graph and considering the problem.


# ============================================================
# 9. APPLY FINAL K-MEANS WITH 3 CLUSTERS
# ============================================================

# kmeans = KMeans(
#     n_clusters=3,
#     random_state=42,
#     n_init=10
# )
# Creates the final K-Means model with three clusters.

# n_clusters=3
# Specifies that the dataset should be divided into three groups.
# This is chosen because the Iris dataset has three known species
# and three clusters are a reasonable choice for this practical.

# random_state=42
# Helps make the random initialization repeatable.

# n_init=10
# Runs the algorithm 10 times with different initial centroids
# and selects the best result based on WCSS.


# -------------------- PERFORM CLUSTERING ---------------------

# df["Cluster"] = kmeans.fit_predict(X_scaled)
# Trains the final K-Means model and assigns a cluster number
# to every flower in the dataset.
# fit_predict() performs training and prediction together.
# The predicted cluster numbers are stored in a new column
# called Cluster in the DataFrame.

# What are cluster labels?
# K-Means assigns numerical labels to its clusters.
# For three clusters, the labels are generally:
# 0, 1 and 2.

# Important:
# Cluster 0 does not automatically mean Iris-setosa.
# Cluster 1 does not automatically mean Iris-versicolor.
# Cluster 2 does not automatically mean Iris-virginica.
# The numbers are only group identifiers assigned by the algorithm.
# Their order may change between different model initializations.

# Also:
# K-Means does not know the actual Species names.
# It groups flowers based only on their numerical measurements.


# -------------------- DISPLAY CLUSTER RESULTS ----------------

# print("\nK-Means Clustering Results:")
# Prints the heading for the clustering results.

# print(df[["Cluster"]].head(10))
# Displays the cluster assignments of the first 10 flowers.
# The output shows which cluster each flower belongs to.
# head(10) displays only the first 10 records.

# Example output format:
#    Cluster
# 0        1
# 1        1
# 2        1
# 3        1
# 4        1
# 5        1
# 6        1
# 7        1
# 8        1
# 9        1

# Note:
# These cluster numbers are only an example.
# Actual cluster assignments can differ.


# -------------------- COUNT DATA POINTS IN EACH CLUSTER ------

# print("\nNumber of Data Points in Each Cluster:")
# Prints the heading for the cluster counts.

# print(df["Cluster"].value_counts().sort_index())
# Counts how many flower records belong to each cluster.
# value_counts() counts the number of occurrences of each cluster label.
# sort_index() arranges the cluster labels in ascending order.

# Example output format:
# Cluster
# 0    50
# 1    50
# 2    50

# Explanation:
# Cluster 0 contains 50 flowers.
# Cluster 1 contains 50 flowers.
# Cluster 2 contains 50 flowers.

# Important:
# The example above is only an illustration.
# K-Means does not guarantee exactly 50 flowers in every cluster.
# The actual number of flowers in each cluster depends
# on the clustering results.

# Total number of flowers:
# The sum of all cluster counts should be 150
# for the standard Iris dataset.


# ============================================================
# 10. CALCULATE SILHOUETTE SCORE
# ============================================================

# score = silhouette_score(X_scaled, df["Cluster"])
# Calculates the Silhouette Score of the clustering result.
# It measures how well each data point fits its own cluster
# compared with other clusters.

# X_scaled
# Contains the standardized numerical flower features.

# df["Cluster"]
# Contains the cluster labels assigned by K-Means.

# What is Silhouette Score?
# Silhouette Score is a clustering evaluation measure.
# It checks whether data points are close to their own cluster
# and far away from other clusters.

# Silhouette Score ranges from -1 to +1.

# Score close to +1:
# Data points are well matched to their own clusters.
# They are generally far away from other clusters.
# This indicates good clustering.

# Score close to 0:
# Data points may lie near the boundary between clusters.
# The clusters may overlap.

# Score close to -1:
# Some data points may have been assigned to the wrong clusters.
# This indicates poor clustering.

# print("\nSilhouette Score:", score)
# Displays the calculated Silhouette Score.

# Important:
# A higher Silhouette Score generally indicates better-separated
# and more clearly formed clusters.
# It should be considered along with the problem requirements
# and other clustering evaluation methods.


# ============================================================
# 11. REDUCE DIMENSIONS USING PCA
# ============================================================

# pca = PCA(n_components=2)
# Creates a PCA model.
# n_components=2 means that the data will be reduced
# to two principal components.

# Why is PCA required?
# The Iris dataset contains four numerical features.
# A normal 2D scatter plot can display only two axes.
# PCA reduces the four features into two new components.
# These two components can be used to visualize the clusters.

# What is PCA?
# PCA stands for Principal Component Analysis.
# It is a dimensionality reduction technique.
# It transforms the original features into new features
# called principal components.

# What is dimensionality reduction?
# Dimensionality reduction means reducing the number of features
# while trying to preserve as much useful information as possible.

# What are principal components?
# Principal components are new combinations of the original features.
# The first principal component captures the maximum possible variation.
# The second principal component captures the next highest variation
# while being uncorrelated with the first component.

# What does n_components=2 mean?
# It keeps only two principal components.
# The original four features become two transformed features.

# X_pca = pca.fit_transform(X_scaled)
# Fits PCA on the standardized data and transforms it.
# The result is stored in X_pca.
# X_pca contains two columns:
# Principal Component 1 and Principal Component 2.

# Important:
# PCA is used here only to visualize the clusters in two dimensions.
# The final K-Means clustering was performed using all four
# standardized original numerical features.


# ============================================================
# 12. PLOT THE FINAL CLUSTERS
# ============================================================

# plt.figure(figsize=(8, 6))
# Creates a new figure with width 8 inches and height 6 inches.

# plt.scatter(
#     X_pca[:, 0],
#     X_pca[:, 1],
#     c=df["Cluster"],
#     cmap="rainbow"
# )
# Creates a scatter plot to display the clusters.

# X_pca[:, 0]
# Selects the first principal component.
# It is used for the horizontal X-axis.

# X_pca[:, 1]
# Selects the second principal component.
# It is used for the vertical Y-axis.

# c=df["Cluster"]
# Assigns a colour to each point based on its cluster number.
# Flowers belonging to the same cluster receive the same colour.

# cmap="rainbow"
# Uses the rainbow colour map to display different clusters
# in different colours.

# What is a scatter plot?
# A scatter plot displays data points as individual dots.
# It helps us understand the distribution of data points
# and the separation between different clusters.

# plt.title("K-Means Clustering on Iris Dataset")
# Sets the title of the final clustering graph.

# plt.xlabel("Principal Component 1")
# Labels the horizontal axis as Principal Component 1.

# plt.ylabel("Principal Component 2")
# Labels the vertical axis as Principal Component 2.

# plt.show()
# Displays the final K-Means clustering graph.


# -------------------- UNDERSTANDING FINAL GRAPH ---------------

# Each dot represents one flower from the Iris dataset.

# The horizontal position represents the first principal component.

# The vertical position represents the second principal component.

# Different colours represent different clusters.

# Dots with the same colour belong to the same cluster.

# Dots that are close to one another have similar
# measurements in the reduced PCA representation.

# Dots that are far away from one another have more
# different measurements in the reduced PCA representation.

# If different coloured groups are clearly separated,
# it indicates that the clusters are visually distinguishable.

# If different colours overlap,
# it indicates that some clusters are not clearly separated
# in the two-dimensional PCA visualization.

# Important:
# The PCA graph is a reduced 2D representation.
# It may not show every separation present in the original
# four-dimensional feature space.


# ============================================================
# 13. UNDERSTAND THE COMPLETE K-MEANS ALGORITHM
# ============================================================

# K-Means is an unsupervised machine learning algorithm.

# It is mainly used for clustering similar data points.

# The letter K represents the number of clusters.

# The word Means refers to calculating the mean position
# of the data points to determine the centroid of each cluster.

# WORKING OF K-MEANS:

# Step 1:
# Select the number of clusters, K.

# Step 2:
# Randomly initialize K centroids.

# Step 3:
# Calculate the distance between every data point
# and each centroid.

# Step 4:
# Assign every data point to its nearest centroid.

# Step 5:
# Calculate the new centroid of each cluster
# by finding the mean of the points assigned to it.

# Step 6:
# Repeat the distance calculation and reassignment process.

# Step 7:
# Stop when the cluster assignments or centroid positions
# stop changing significantly, or a stopping condition is reached.

# FINAL RESULT:
# The dataset is divided into K clusters
# containing similar data points.


# ============================================================
# 14. IMPORTANT DIFFERENCES FOR VIVA
# ============================================================

# SUPERVISED LEARNING:
# Uses labelled data.
# The correct output is already known during training.
# Example: Classification and Regression.

# UNSUPERVISED LEARNING:
# Uses unlabelled data.
# The algorithm discovers patterns or groups in the data.
# Example: K-Means Clustering.

# CLASSIFICATION:
# Predicts predefined categories.
# Example: Predicting whether an email is spam or not spam.

# CLUSTERING:
# Groups similar data points without predefined class labels.
# Example: Grouping customers based on purchasing behaviour.

# K-MEANS:
# A clustering algorithm that uses centroids
# to divide data into K groups.

# ELBOW METHOD:
# Helps select a suitable number of clusters
# by observing the change in WCSS.

# WCSS:
# Measures the sum of squared distances between data points
# and their assigned cluster centroids.

# SILHOUETTE SCORE:
# Measures how well data points fit into their assigned clusters
# compared with other clusters.

# PCA:
# Reduces the number of features while retaining
# as much variation as possible.

# STANDARDIZATION:
# Converts numerical features to a comparable scale.

# CENTROID:
# The center of a cluster, calculated using the mean
# of the data points belonging to that cluster.


# ============================================================
# 15. IMPORTANT VIVA QUESTIONS AND ANSWERS
# ============================================================

# Q1. What is K-Means Clustering?
# Answer:
# K-Means is an unsupervised machine learning algorithm
# used to divide data into K clusters based on similarity.

# Q2. Why is K-Means called an unsupervised algorithm?
# Answer:
# Because it does not require predefined output labels.
# It discovers groups from the input data itself.

# Q3. What is the value of K in this practical?
# Answer:
# K is 3 in the final clustering model.

# Q4. Why are there three clusters?
# Answer:
# The Iris dataset contains three known flower species.
# Therefore, three clusters are used in this practical.

# Q5. Does K-Means know the actual species names?
# Answer:
# No. K-Means does not use the Species column.
# It assigns cluster numbers based on numerical features.

# Q6. What is the Elbow Method?
# Answer:
# It is a technique used to select a suitable number of clusters
# by plotting WCSS against the number of clusters.

# Q7. What is WCSS?
# Answer:
# WCSS stands for Within-Cluster Sum of Squares.
# It measures the squared distances between data points
# and their assigned cluster centroids.

# Q8. What does the elbow point indicate?
# Answer:
# It indicates a suitable number of clusters where
# adding more clusters gives a smaller improvement in WCSS.

# Q9. Why does WCSS decrease when K increases?
# Answer:
# Because increasing the number of clusters allows
# data points to be grouped closer to their centroids.

# Q10. What is a centroid?
# Answer:
# A centroid is the center of a cluster.
# It is calculated using the mean of the data points
# assigned to that cluster.

# Q11. Why is StandardScaler used?
# Answer:
# It standardizes the numerical features so that
# features with larger scales do not dominate distance calculations.

# Q12. Why is Species removed?
# Answer:
# Because K-Means is unsupervised and should form clusters
# without using the actual species labels.

# Q13. What is Silhouette Score?
# Answer:
# It measures how well data points fit into their own clusters
# compared with other clusters.

# Q14. What is the range of Silhouette Score?
# Answer:
# It ranges from -1 to +1.

# Q15. What does a Silhouette Score close to +1 indicate?
# Answer:
# It indicates that the clusters are generally well separated.

# Q16. What does a Silhouette Score close to 0 indicate?
# Answer:
# It indicates that some data points may lie near
# the boundaries between clusters.

# Q17. What does a negative Silhouette Score indicate?
# Answer:
# It may indicate that some data points are assigned
# to an inappropriate cluster.

# Q18. What is PCA?
# Answer:
# PCA stands for Principal Component Analysis.
# It is a dimensionality reduction technique.

# Q19. Why is PCA used in this practical?
# Answer:
# The dataset has four numerical features.
# PCA reduces them to two components so that the clusters
# can be displayed on a 2D graph.

# Q20. What is the difference between K-Means and PCA?
# Answer:
# K-Means is used to group similar data points.
# PCA is used to reduce the number of features.

# Q21. What is random_state=42?
# Answer:
# It controls random initialization and helps produce
# repeatable results.

# Q22. What is n_init=10?
# Answer:
# It runs K-Means 10 times with different initial centroids
# and selects the result with the lowest WCSS.

# Q23. What is the use of fit_predict()?
# Answer:
# It trains the clustering model and returns
# the cluster label assigned to each data point.

# Q24. What are the applications of K-Means?
# Answer:
# 1. Customer segmentation.
# 2. Image segmentation.
# 3. Document grouping.
# 4. Market analysis.
# 5. Pattern recognition.
# 6. Grouping similar products.

# Q25. What are the limitations of K-Means?
# Answer:
# 1. The number of clusters K must be selected.
# 2. It is sensitive to initial centroid positions.
# 3. It is sensitive to feature scaling.
# 4. It can be affected by outliers.
# 5. It may not work well for clusters with irregular shapes.


# ============================================================
# 16. EXPECTED OUTPUT EXPLANATION
# ============================================================

# OUTPUT 1: ORIGINAL IRIS DATASET

# The program displays:
# Original Iris Dataset:

# Then it displays the first five rows of Iris.csv.

# You can see columns such as:
# Id, SepalLengthCm, SepalWidthCm,
# PetalLengthCm, PetalWidthCm and Species.

# This confirms that the dataset has been loaded successfully.


# OUTPUT 2: ELBOW METHOD GRAPH

# The first graph displays:
# X-axis: Number of Clusters.
# Y-axis: WCSS.

# The graph generally decreases as the number of clusters increases.

# Observe the graph for a point where the decrease
# in WCSS becomes slower.

# That point is called the elbow point.

# In this practical, three clusters are used in the final model.


# OUTPUT 3: K-MEANS CLUSTERING RESULTS

# The program displays the cluster numbers
# assigned to the first 10 flowers.

# The cluster labels are numerical values such as:
# 0, 1 and 2.

# Each number represents a different group.


# OUTPUT 4: NUMBER OF DATA POINTS IN EACH CLUSTER

# The program displays how many flowers belong to each cluster.

# The total number of records across all clusters
# should be 150 for the standard Iris dataset.

# The individual cluster sizes may be different.


# OUTPUT 5: SILHOUETTE SCORE

# The program displays a numerical score between -1 and +1.

# A score closer to +1 generally indicates better-separated clusters.

# A score close to 0 indicates overlapping cluster boundaries.

# A negative score may indicate incorrectly assigned data points.

# The exact score depends on the clustering result.


# OUTPUT 6: FINAL CLUSTERING GRAPH

# The final graph displays all 150 flower records as dots.

# The dots are shown using different colours
# according to their assigned cluster.

# The X-axis represents Principal Component 1.

# The Y-axis represents Principal Component 2.

# The graph helps us visually understand the grouping
# of similar flowers.


# ============================================================
# 17. CONCLUSION
# ============================================================

# In this practical, K-Means Clustering is implemented
# on the Iris dataset using Python and Scikit-learn.

# First, the Iris dataset is loaded and unnecessary columns
# are removed.

# Missing values are handled and numerical features
# are standardized using StandardScaler.

# The Elbow Method is applied by calculating WCSS
# for different numbers of clusters.

# Based on the selected number of clusters,
# the final K-Means model is created with three clusters.

# The model groups the flowers based on their numerical features.

# The clustering performance is evaluated using Silhouette Score.

# Finally, PCA is used to reduce the four features
# into two principal components for visualization.

# Thus, K-Means Clustering is successfully implemented
# on the Iris dataset and the clusters are visualized.
# ============================================================tering graph.
