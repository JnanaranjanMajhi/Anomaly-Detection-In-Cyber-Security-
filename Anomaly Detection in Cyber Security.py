# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

# Load the dataset
file_path = "data/RT_IOT2022.csv" # Change this path to your dataset location
data = pd.read_csv(file_path)

# Initial exploration
print("Initial Data Shape:", data.shape)
print("Data Info:")
print(data.info())
print("Sample Data:\n", data.head())

# Drop irrelevant columns
columns_to_drop = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
data = data.drop(columns=columns_to_drop, errors='ignore')

# Identify and print the target column
target_column = 'Attack_type'
for col in ['Label', 'Attack', 'target']:
    if col in data.columns:
        target_column = col
        break

if target_column is None:
    raise ValueError("No target column found. Please verify the column name for the labels.")

# Encode the target column: BENIGN -> 0, Attack -> 1
label_encoder = LabelEncoder()
data[target_column] = label_encoder.fit_transform(data[target_column])

# Handle missing values by filling with the median value of each column
numeric_columns = data.select_dtypes(include=np.number).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

# Identify categorical columns (assuming they are of type object)
categorical_columns = data.select_dtypes(include=['object']).columns

# Encode categorical columns using LabelEncoder
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# After encoding, check if all columns are numeric
print("Data types after encoding:", data.dtypes)

# Feature Selection based on Mutual Information
X = data.drop(target_column, axis=1)
y = data[target_column]

# Calculate mutual information scores
mi_scores = mutual_info_classif(X, y)

# Display mutual information scores
mi_scores_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print("Top features based on mutual information:\n", mi_scores_series.head(20))

# Selecting the top features based on MI
top_features = mi_scores_series.head(20).index.tolist()

# Filter dataset with top features
X = X[top_features]

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Exploratory Data Analysis (EDA)
print("Plotting histogram")
plt.figure(figsize=(10, 6))
sns.histplot(data[target_column], kde=False)
plt.title(f"Distribution of {target_column} (0: BENIGN, 1: Attack)")
plt.tight_layout()
plt.show()

# Correlation heatmap
print("Plotting correlation heatmap")
plt.figure(figsize=(12, 10))
sns.heatmap(pd.DataFrame(X_scaled, columns=top_features).corr(), cmap='viridis', annot=False)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Determine optimal number of clusters using the Elbow method
inertia = []
cluster_range = range(1, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(cluster_range, inertia, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.tight_layout()
plt.show()

# Using k=2 for anomaly detection (normal vs. anomalous traffic)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

# Predict the clusters
predicted_labels = kmeans.labels_

# Evaluate clustering with silhouette score
silhouette_avg = silhouette_score(X_scaled, predicted_labels)
print(f"Silhouette Score: {silhouette_avg:.2f}")

# Map predicted clusters to actual labels
unique_labels, counts = np.unique(predicted_labels, return_counts=True)
label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1} if counts[0] > counts[1] else {unique_labels[0]: 1, unique_labels[1]: 0}
predicted = [label_mapping[label] for label in predicted_labels]

# Confusion Matrix and Classification Report
class_names = ['BENIGN', 'Attack']
print("Confusion Matrix:")
print(confusion_matrix(y, predicted))

print("\nClassification Report:")
print(classification_report(y, predicted, target_names=class_names))

# PCA for 2D Visualization
print("Plotting PCA")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=predicted, palette="coolwarm", style=y)
plt.title("PCA of K-Means Clustering (2 Clusters)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(['Predicted BENIGN', 'Predicted Attack'])
plt.tight_layout()
plt.show()
