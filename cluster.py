import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.preprocessing import LabelEncoder
import os

# load the data
folder_path = '/embeddings/poliovirus/28mnv'
files = os.listdir(folder_path)

all_data = []
all_labels = []

for file in files:
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        all_data.append(df.values)
        label = file.split('.')[0]
        all_labels.extend([label] * len(df))

data = np.vstack(all_data)

scaler = StandardScaler()
X = scaler.fit_transform(data)

y = np.array(all_labels)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print(f"Data shape: {X.shape}, Labels shape: {y.shape}")

# clustering
dbscan = DBSCAN(eps=3.4, min_samples=5)
cluster_labels = dbscan.fit_predict(X)

# evaluation
ari = adjusted_rand_score(y, cluster_labels) if len(set(cluster_labels)) > 1 else -1
nmi = normalized_mutual_info_score(y, cluster_labels) if len(set(cluster_labels)) > 1 else -1
fmi = fowlkes_mallows_score(y, cluster_labels) if len(set(cluster_labels)) > 1 else -1

print('\nResults：')
print(f"Silhouette Score: {ss:.4f}")
print(f"Calinski-Harabasz Score: {chs:.4f}")
print(f"Davies-Bouldin Score: {dbs:.4f}")
print(f"Adjusted Rand Index: {ari:.4f}")
print(f"Normalized Mutual Information: {nmi:.4f}")
print(f"Fowlkes-Mallows Score: {fmi:.4f}")

# calculate the noise ratio
noise_ratio = np.sum(cluster_labels == -1) / len(cluster_labels)
print(f"Noise ratio: {noise_ratio:.4f}")
