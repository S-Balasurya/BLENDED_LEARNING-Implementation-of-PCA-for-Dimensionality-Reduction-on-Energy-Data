# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Dataset Loading – Load the Height and Weight dataset using pandas and select the required features for analysis.
2.Feature Scaling – Standardize the selected features using StandardScaler to bring them to the same scale.
3.Principal Component Analysis (PCA) – Apply PCA to reduce the dimensionality of the dataset and extract principal components.
4.Visualization and Analysis – Analyze the explained variance and visualize the principal components using a scatter plot.

## Program:
```
/*
Program to implement Principal Component Analysis (PCA) for dimensionality reduction on the energy data.
Developed by: Balasurya S
RegisterNumber: 212225100003 
*/

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset from a local file
data = pd.read_csv("HeightWeight.csv")

# Step 2: Explore the data
print(data.head())
print(data.columns)

# Step 3: Preprocess the data (Feature Scaling)
X = data[['Height(Inches)', 'Weight(Pounds)']]  # Use the appropriate column names

# Standardize the features to bring them to the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 5: Analyze the explained variance
explained_variance = pca.explained_variance_ratio_
print("\nName: Balasurya S")
print("Reg No: 212225100003\n")
print("Explained Variance Ratio for each Principal Component:", explained_variance)
print("Total Explained Variance:", sum(explained_variance))

# Step 6: Visualize the principal components
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# Plot the first two principal components
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Heights and Weights Dataset")
plt.show()
```

## Output:

<Figure size 800x600 with 1 Axes><img width="689" height="545" alt="image" src="https://github.com/user-attachments/assets/5a6fd934-67a7-4143-9371-48f3b7b49765" />


## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
