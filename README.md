# Anomaly Detection Using Clustering and Ensemble Methods

This project focuses on detecting anomalies in network traffic data using unsupervised learning techniques, including clustering and ensemble anomaly detection methods. The dataset used is from the **ISCX 2017 dataset**, specifically the `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` file. 

The project involves:
- Preprocessing the data (handling missing values, encoding categorical data, and scaling features).
- Applying clustering techniques (KMeans) and anomaly detection algorithms (Isolation Forest, Local Outlier Factor, and One-Class SVM).
- Using ensemble voting to combine the results from the individual anomaly detection methods.
- Visualizing clustering and anomaly detection results using PCA for dimensionality reduction.

## File Path
The dataset is located at `data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`. Make sure the dataset is available in the specified path before running the program.

### Preprocessing:
1. Remove unnecessary columns (e.g., Timestamp, Flow ID, IP addresses, and Ports).
2. Handle missing, infinite, and extreme values.
3. Apply logarithmic transformation for skewed numeric columns.
4. Standardize the dataset to zero mean and unit variance.

### Anomaly Detection:
- **KMeans Clustering**: KMeans is used to cluster the data and determine the optimal number of clusters using silhouette score.
- **Isolation Forest**: Anomaly detection based on the isolation principle.
- **Local Outlier Factor (LOF)**: Detects outliers based on the density of data points.
- **One-Class SVM**: A method based on support vector machines for detecting outliers.

### Ensemble Method:
An ensemble of the above anomaly detection methods is used by combining their results. A data point is considered an anomaly if two or more methods agree.

### Evaluation:
The performance of anomaly detection is evaluated using **Precision**, **Recall**, and **F1 Score**.

### Visualization:
- **PCA** is used to reduce the dimensionality of the data for visualization.
- A scatter plot visualizes the KMeans clustering results, highlighting the anomalies in red.

### Output:
- The detected anomalies are saved to `Detected_Ensemble_Anomalies.csv`.
- A bar plot showing the distribution of anomalies across clusters is displayed.

## Dependencies
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib

### To install the required dependencies, use the following:

```bash
pip install pandas numpy scikit-learn matplotlib
```
## Usage
Download or clone this repository.
- Place the Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv file in the data directory.
- Run the Python script that processes the dataset and performs the anomaly detection.
```bash
python anomaly_detection.py
```
## Results
The anomalies detected by the ensemble method will be saved in Detected_Ensemble_Anomalies.csv.
A visualization of anomalies in the clusters will be shown.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
