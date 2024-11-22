# Importing necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# File path of the dataset
file_to_load = r"C:\Users\jnana\Downloads\ML\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# Check if the file exists
if not os.path.exists(file_to_load):
    raise FileNotFoundError(f"File not found: {file_to_load}. Please verify the file path and try again.")

# Load the dataset into a pandas DataFrame
data = pd.read_csv(file_to_load)

# Sample the data to reduce its size for faster processing (10% of the data)
data = data.sample(frac=0.1, random_state=42)

# Drop unnecessary columns from the dataset
columns_to_drop = ['Timestamp', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol']
data = data.drop(columns=columns_to_drop, errors='ignore')

# Handle infinite values and replace them with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill missing values with 0 (for now)
data.fillna(0, inplace=True)

# Convert categorical columns to numerical using Label Encoding
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Get the numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns

# Set extreme values in numeric columns (greater than 1e10) as NaN and fill with 0
data[numeric_columns][data[numeric_columns] > 1e10] = np.nan
data.fillna(0, inplace=True)

# Apply logarithmic transformation to skewed numeric columns to reduce skewness
for col in data.columns:
    if data[col].skew() > 1:
        data[col] = np.log1p(data[col])

# Check for any remaining NaN or infinite values and handle them
if data[numeric_columns].isnull().values.any():
    print("Warning: There are still NaN values in the dataset.")
    data.fillna(0, inplace=True)
if np.isinf(data[numeric_columns].values).any():
    print("Warning: There are still infinite values in the dataset.")
    data.replace([np.inf, -np.inf], 0, inplace=True)

# Standardize the numeric data (zero mean, unit variance)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[numeric_columns])

# Function to determine the optimal number of clusters for KMeans using silhouette score
def optimal_kmeans_clusters(data, max_k=10):
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(data)
        score = silhouette_score(data, clusters)
        silhouette_scores.append(score)
        print(f"Silhouette Score for k={k}: {score:.2f}")
    
    # Plot the silhouette scores for different k values
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different k values')
    plt.show()

# Call the function to find the optimal k value
optimal_kmeans_clusters(data_scaled)

# Set the best k value based on the silhouette score (here we choose 7)
best_k = 7

# Perform KMeans clustering with the selected number of clusters
kmeans = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans.fit_predict(data_scaled)
data['Cluster'] = clusters

# Evaluate the clustering performance using silhouette score
silhouette_avg = silhouette_score(data_scaled, clusters)
print(f"Silhouette Score for KMeans with {best_k} clusters: {silhouette_avg:.2f}")

# Anomaly detection using Isolation Forest
iso_forest = IsolationForest(contamination=0.1, n_estimators=200, max_samples=0.8, random_state=42)
iso_anomalies = iso_forest.fit_predict(data_scaled)
data['Iso_Anomaly'] = iso_anomalies == -1

# Anomaly detection using Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
lof_anomalies = lof.fit_predict(data_scaled)
data['LOF_Anomaly'] = lof_anomalies == -1

# Anomaly detection using One-Class SVM
oc_svm = OneClassSVM(nu=0.1, kernel="rbf", gamma='scale')
svm_anomalies = oc_svm.fit_predict(data_scaled)
data['SVM_Anomaly'] = svm_anomalies == -1

# Create an ensemble anomaly detection by combining the results of all three methods
ensemble_votes = (
    data['Iso_Anomaly'].astype(int) + 
    data['LOF_Anomaly'].astype(int) + 
    data['SVM_Anomaly'].astype(int)
)

# Mark an anomaly if two or more methods agree
data['Ensemble_Anomaly'] = ensemble_votes >= 2

# Apply PCA (Principal Component Analysis) for visualization
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Visualize the KMeans clustering results in 2D using PCA components
plt.figure(figsize=(12, 8))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=data['Cluster'], cmap='viridis', alpha=0.6, edgecolors='w', s=50)
plt.title('K-means Clustering Visualization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# Highlight anomalous data points in red
anomalous_data_pca = data_pca[data['Ensemble_Anomaly']]
plt.scatter(anomalous_data_pca[:, 0], anomalous_data_pca[:, 1], color='red', label='Ensemble Anomalies', edgecolors='black', s=100)
plt.legend()
plt.show()

# Evaluate the performance of the anomaly detection using precision, recall, and F1 score
y_true = data['Ensemble_Anomaly'].astype(int)
y_pred = data['Iso_Anomaly'].astype(int)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Print the evaluation metrics
print(f"Improved Precision: {precision:.2f}")
print(f"Improved Recall: {recall:.2f}")
print(f"Improved F1 Score: {f1:.2f}")

# Display the anomalies detected by the ensemble method
anomalies_detected = data[data['Ensemble_Anomaly']]
print(f"Total anomalies detected by ensemble: {len(anomalies_detected)}")
print("Sample anomalies data:")
print(anomalies_detected.head())

# Save the detected anomalies to a CSV file
anomalies_detected.to_csv("Detected_Ensemble_Anomalies.csv", index=False)
print("Ensemble anomalies saved to 'Detected_Ensemble_Anomalies.csv'.")

# Visualize the distribution of anomalies across clusters
plt.figure(figsize=(8, 6))
anomalies_detected['Cluster'].value_counts().plot(kind='bar', color='coral')
plt.title("Anomalies per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Anomalies")
plt.show()
