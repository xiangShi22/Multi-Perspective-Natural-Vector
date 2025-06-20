import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score, jaccard_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE  # 导入SMOTE方法
import os

# 数据加载与处理
folder_path = '/home/shixiang/shixiang/multiview_review/poliovirus/3mer'
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

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(data)

# 标签编码
y = np.array(all_labels)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 打印SMOTE前后数据的维度
print(f"Original X shape: {X.shape}, y shape: {y.shape}")

# 处理数据不平衡：应用SMOTE过采样方法
# smote = SMOTE(sampling_strategy='auto', random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)
X_resampled, y_resampled = X, y
# 打印SMOTE后的数据维度，确认是否一致
print(f"Resampled X shape: {X_resampled.shape}, y_resampled shape: {y_resampled.shape}")

# 确保X_resampled和y_resampled的样本数一致
assert X_resampled.shape[0] == y_resampled.shape[0], "样本数不一致"

# 使用DBSCAN进行聚类
dbscan = DBSCAN(eps=3.8, min_samples=8)
# 初始化用于存储各项评估指标的累积值
ss_total = 0
chs_total = 0
dbs_total = 0
ari_total = 0
nmi_total = 0
fmi_total = 0
jss_total = 0
folds = 0

# 使用交叉验证来评估模型（StratifiedKFold）
skf = StratifiedKFold(n_splits=5)
for train_idx, test_idx in skf.split(X_resampled, y_resampled):
    X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
    y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]
    
    # 在训练集上进行聚类
    dbscan.fit(X_train)  # 训练DBSCAN
    communities_train = dbscan.labels_  # 获取训练集的聚类标签
    
    # 对于测试集，DBSCAN无法直接预测标签，但可以直接对测试集进行聚类
    communities_test = dbscan.fit_predict(X_test)  # 使用训练后的DBSCAN对测试集进行聚类
    
    # 计算每个fold的评估指标
    ss = silhouette_score(X_test, communities_test) if len(set(communities_test)) > 1 else -1
    chs = calinski_harabasz_score(X_test, communities_test) if len(set(communities_test)) > 1 else -1
    dbs = davies_bouldin_score(X_test, communities_test)
    
    ari = adjusted_rand_score(y_test, communities_test) if len(set(communities_test)) > 1 else -1
    nmi = normalized_mutual_info_score(y_test, communities_test) if len(set(communities_test)) > 1 else -1
    fmi = fowlkes_mallows_score(y_test, communities_test) if len(set(communities_test)) > 1 else -1
    jss = jaccard_score(y_test, communities_test, average='weighted') if len(set(communities_test)) > 1 else -1
    
    # 将每个fold的指标累加
    ss_total += ss
    chs_total += chs
    dbs_total += dbs
    ari_total += ari
    nmi_total += nmi
    fmi_total += fmi
    jss_total += jss
    
    folds += 1  # 累加fold数量

# 计算平均值
ss_avg = ss_total / folds
chs_avg = chs_total / folds
dbs_avg = dbs_total / folds
ari_avg = ari_total / folds
nmi_avg = nmi_total / folds
fmi_avg = fmi_total / folds
jss_avg = jss_total / folds

# 打印平均结果
print('\n平均聚类性能：')
print(f"Silhouette Score: {ss_avg:.4f}")
print(f"Calinski-Harabasz Score: {chs_avg:.4f}")
print(f"Davies-Bouldin Score: {dbs_avg:.4f}")
print(f"Adjusted Rand Index: {ari_avg:.4f}")
print(f"Normalized Mutual Information: {nmi_avg:.4f}")
print(f"Fowlkes-Mallows Score: {fmi_avg:.4f}")
print(f"Jaccard Score: {jss_avg:.4f}")

# 计算噪声比例
noise_ratio = np.sum(dbscan.labels_ == -1) / len(dbscan.labels_)
print(f"Noise ratio: {noise_ratio:.4f}")
