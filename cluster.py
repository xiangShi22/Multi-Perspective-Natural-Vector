import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_index, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler 
import os

# Data loading and processing
folder_path = '/poliovirus/3mer'
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

# Data standardization
scaler = StandardScaler()
X = scaler.fit_transform(data)

# Label encoding
y = np.array(all_labels)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print(f"Original X shape: {X.shape}, y shape: {y.shape}")

undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=10)
X_resampled, y_resampled = undersampler.fit_resample(X, y)
print(f"Resampled X shape: {X_resampled.shape}, y_resampled shape: {y_resampled.shape}")

# Verify consistent sample count
assert X_resampled.shape[0] == y_resampled.shape[0], "Sample count mismatch"

# Initialize DBSCAN clustering
dbscan = DBSCAN(eps=3.8, min_samples=8)
# Initialize accumulators for evaluation metrics
ari_total = 0
nmi_total = 0
fmi_total = 0
folds = 0

# Perform cross-validation using Stratified K-Fold
skf = StratifiedKFold(n_splits=5)
for train_idx, test_idx in skf.split(X_resampled, y_resampled):
    X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
    y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]
    
    # Cluster on training set
    dbscan.fit(X_train)  # Train DBSCAN model
    communities_train = dbscan.labels_  # Get cluster labels for training set
    
    # Cluster test set directly since DBSCAN can't predict new samples
    communities_test = dbscan.fit_predict(X_test)  # Cluster test data
    
    # Compute evaluation metrics for current fold
    if len(set(communities_test)) > 1:
        ari = adjusted_rand_score(y_test, communities_test)
        nmi = normalized_mutual_info_score(y_test, communities_test)
        fmi = fowlkes_mallows_score(y_test, communities_test)
    else:
        # Assign default values for invalid cases
        ari = nmi = fmi = -1
    
    dbs = davies_bouldin_score(X_test, communities_test)  
    
    # Accumulate metrics across folds
    ari_total += ari
    nmi_total += nmi
    fmi_total += fmi

    folds += 1  

# Calculate average metrics
ari_avg = ari_total / folds
nmi_avg = nmi_total / folds
fmi_avg = fmi_total / folds

# Print average results
print('\nAverage clustering performance:')
print(f"Adjusted Rand Index: {ari_avg:.4f}")
print(f"Normalized Mutual Information: {nmi_avg:.4f}")
print(f"Fowlkes-Mallows Score: {fmi_avg:.4f}")

# Calculate noise ratio
noise_ratio = np.sum(dbscan.labels_ == -1) / len(dbscan.labels_)
print(f"Noise ratio: {noise_ratio:.4f}")
