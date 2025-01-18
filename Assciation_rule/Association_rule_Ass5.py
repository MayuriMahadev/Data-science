# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:25:34 2025

@author: Admin
"""
'''
1.Problem Statement:
    Perform clustering on mixed data. Convert the
    categorical variables to numeric by using 
    dummies or label encoding and perform
    normalization techniques. The dataset has the
    details of customers related to their auto 
    insurance. Refer to Autoinsurance.csv dataset.
    
1.1.Business Objective
The objective of performing clustering on the
 Auto Insurance dataset is to:

1.Group customers into segments based on their
 behaviors, demographics, and claim history.
2.Identify patterns and trends in customer
 behavior to predict future risks and potential
 claims.
3.Develop personalized marketing strategies for
 each cluster to improve retention and satisfaction.
4.Optimize policy pricing by identifying high-risk
 and low-risk customer groups.
5.Design targeted interventions such as 
cross-selling or premium service offerings to
 appropriate segments.

1.2.Constraints
1.Data Quality and Missing Values:
Incomplete or inaccurate data can affect clustering performance. Missing values need to be handled properly.
2.Mixed Data Types:

The dataset contains both categorical and numerical data, requiring careful pre-processing through encoding and scaling.
3.Selection of Optimal Clusters:

Identifying the ideal number of clusters is subjective and may require multiple methods (e.g., Elbow method, Silhouette score) for validation.
4.Interpretability of Clusters:

Clusters must be interpretable for business actions; meaningless or overly complex clusters may limit insights.
5.Scalability:

The clustering model needs to be scalable to handle increasing data volumes over time.
6.Timely Business Actions:

Insights derived from clusters need to be acted upon quickly to provide measurable business value.
7.Cost-Effectiveness:

Implementing strategies based on cluster insights must align with the business budget and available resources.

'''
# Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Dataset
# Update the file path as needed
df = pd.read_csv("D:/Assciation_rule/AutoInsurance.csv")

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Get basic info about the dataset
print("\nDataset Information:")
print(df.info())

# Data Pre-processing

# Handle Missing Values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Drop rows with missing values or impute
df = df.dropna()  # Optionally, replace this with imputation techniques if needed
print("\nData Shape After Dropping Missing Values:", df.shape)

# Encode Categorical Variables
# Label Encoding for binary variables
binary_cols = ['Gender', 'Married']  # Adjust column names as per your dataset
for col in binary_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])

# One-Hot Encoding for multi-category variables
df = pd.get_dummies(df, drop_first=True)
print("\nData After Encoding:")
print(df.head())

# Scale Numerical Features
# Identify numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Standardize the numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
print("\nData After Scaling:")
print(df.head())

# Determine the Optimal Number of Clusters
# Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Silhouette Score
print("\nSilhouette Scores for Different Cluster Sizes:")
for n_clusters in range(2, 6):
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    preds = clusterer.fit_predict(df)
    score = silhouette_score(df, preds)
    print(f"Silhouette Score for {n_clusters} clusters: {score}")

# Apply K-means Clustering
# Assuming the optimal cluster count from the Elbow method or Silhouette Score
optimal_clusters = 4  # Replace this with the optimal number based on analysis
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df)

# Check the distribution of clusters
print("\nCluster Distribution:")
print(df['Cluster'].value_counts())

# Visualize the Clusters Using PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df.drop('Cluster', axis=1))

# Plot the clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=df['Cluster'], palette='viridis')
plt.title('Customer Clusters - PCA Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()
