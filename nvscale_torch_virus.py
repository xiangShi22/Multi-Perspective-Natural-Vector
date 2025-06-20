import os
from Bio import SeqIO
import pandas as pd
from time import time
from typing import *
import torch
from torch import Tensor
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

s = time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 指定输入和输出
input_file = '/home/shixiang/shixiang/multiview/viral.1.genomic.gbff'  # 替换为你的GBFF文件路径
output_folder = '/home/shixiang/shixiang/multiview/virus/mnvscale_torch_10'  # 替换为输出文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 存储不同Family的特征向量
family_vectors = {}

@torch.no_grad()
def calculate_counts(seq: Tensor):
    L = seq.shape[0]
    one_hot = torch.nn.functional.one_hot(seq, num_classes=4).float()
    counts = one_hot.sum(dim=0)
    positions = torch.arange(1, L+1, dtype=torch.float32, device=seq.device)
    avg_pos = (one_hot.T @ positions) / counts.clamp(min=1e-7)
    
    # 归一化位置到[0, 2π]
    normalized_pos = positions / L * 2 * torch.pi
    cos_vals = torch.cos(normalized_pos)
    sin_vals = torch.sin(normalized_pos)
    
    avg_cos = (one_hot.T @ cos_vals) / counts.clamp(min=1e-7)
    avg_sin = (one_hot.T @ sin_vals) / counts.clamp(min=1e-7)
    return counts, avg_pos, avg_cos, avg_sin, one_hot, positions, normalized_pos

@torch.no_grad()
def calculate_moments(
    seq: Tensor, 
    counts: Tensor, 
    avg_pos: Tensor,
    avg_cos: Tensor,
    avg_sin: Tensor,
    one_hot: Tensor,
    positions: Tensor,
    normalized_pos: Tensor,
    k_values: List[int] = [2,3,4]
) -> Tuple[Tensor, Tensor, Tensor]:
    device = seq.device
    L = seq.shape[0]
    k_tensor = torch.tensor(k_values, device=device, dtype=torch.float32)
    
    
    # === 位置归一化 ===
    # 将位置映射到 [0, 2π] 范围
    pos_diff = positions[:, None] - avg_pos[None, :]

    denominator = L * counts.clamp(min=1e-7)  # [4] # 防止除零

    # 计算 (diff / denominator) 的幂次 [L,4,K]
    scaled_diff = (pos_diff.unsqueeze(-1) / denominator[None, :, None])  # [L,4,1] / [4,1] -> [L,4,K]
    power_terms = scaled_diff ** (k_tensor - 1)  # [L,4,K]

    # 计算完整项：diff * (diff/denominator)^(k-1) [L,4,K]
    full_terms = pos_diff.unsqueeze(-1) * power_terms  # 广播乘法 [L,4,K]

    # 应用核苷酸掩码并求和 [L,4,K] -> [4,K]
    masked_terms = full_terms * one_hot.unsqueeze(-1)  # [L,4,K]
    raw_moments = masked_terms.sum(dim=0)  

    # === 计算三角矩 ===
    # 计算cos和sin位置 [L]
    cos_pos = torch.cos(normalized_pos)
    sin_pos = torch.sin(normalized_pos)
    
    # 计算cos差值 [L,4]
    cos_diff = cos_pos[:, None, None] - avg_cos[None, :, None]  # [L,4,1]
    cos_terms = cos_diff ** k_tensor  # [L,4,K]
    cos_moments = (cos_terms * one_hot.unsqueeze(-1)).sum(dim=0)  # [4,K]
    cos_moments /= counts[:, None].clamp(min=1e-7)  # [4,K]
    
    # 计算sin差值 [L,4]
    sin_diff = sin_pos[:, None, None] - avg_sin[None, :, None]  # [L,4,1]
    sin_terms = sin_diff ** k_tensor  # [L,4,K]
    sin_moments = (sin_terms * one_hot.unsqueeze(-1)).sum(dim=0)  # [4,K]
    sin_moments /= counts[:, None].clamp(min=1e-7)  # [4,K]

    return raw_moments, cos_moments, sin_moments  # 转置为[4, len(k)]

def convert(sequence: str) -> Tensor:
    # 新增严格模式
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    valid_indices = []
    
    for c in sequence:
        if (idx := mapping.get(c.upper(), -1)) == -1:
            raise ValueError(f"发现无效字符 '{c}'，拒绝处理整个序列")
        valid_indices.append(idx)
        
    return torch.tensor(valid_indices, device=device, dtype=torch.long)


max_k = 9  # 最大阶数

# 预生成列名
columns = [
    'A_count', 'C_count', 'G_count', 'T_count',
    'A_avg_pos', 'C_avg_pos', 'G_avg_pos', 'T_avg_pos',
    'A_avg_pos_cos', 'C_avg_pos_cos', 'G_avg_pos_cos', 'T_avg_pos_cos',
    'A_avg_pos_sin', 'C_avg_pos_sin', 'G_avg_pos_sin', 'T_avg_pos_sin'
]
for k in range(2, max_k + 1):
    for moment_type in ['', 'cos_', 'sin_']:
        columns.extend([f'{nt}_{moment_type}D_{k}' for nt in ['A', 'C', 'G', 'T']])

# 解析GBFF文件
print(f"Processing {input_file}...")
with open(input_file, 'r') as handle:
    for record in SeqIO.parse(handle, 'genbank'):
        # 检查是否为完整基因组
        if 'complete' not in record.description.lower():
            continue
        try:
            seq_tensor = convert(str(record.seq))
        except ValueError as e:
            # print(f"Skipping record {record.id} due to error: {e}")
            continue        
        # 提取分类信息
        taxonomy = record.annotations.get('taxonomy', [])
        family = next((tax for tax in taxonomy if tax.lower().endswith('viridae')), 'Unknown')
        
        if family == 'Unknown':
            continue  # 跳过没有"viridae"分类的条目
        
        # 计算统计量
        count, avg_pos, avg_cos, avg_sin, one_hot, positions, normalized_pos = calculate_counts(seq_tensor)
        
        # 计算各阶矩
        raw_m, cos_m, sin_m = calculate_moments(seq_tensor, count, avg_pos, avg_cos, avg_sin, one_hot, positions, normalized_pos, k_values=list(range(2, max_k+1)))
        
        # 拼接所有特征
        # 修改特征向量拼接方式（关键修复）
        feature_vector = np.concatenate([
            count.cpu().numpy().flatten(),
            avg_pos.cpu().numpy().flatten(),
            avg_cos.cpu().numpy().flatten(),
            avg_sin.cpu().numpy().flatten(),
            raw_m.T.cpu().numpy().flatten(),
            cos_m.T.cpu().numpy().flatten(),
            sin_m.T.cpu().numpy().flatten()
        ])
        
        
        # 按Family存储特征向量
        if family not in family_vectors:
            family_vectors[family] = []
        family_vectors[family].append(feature_vector)

# 为每个Family保存结果
for family, vectors in family_vectors.items():
    # 清理family名称用于文件名
    safe_family = "".join(c if c.isalnum() else "_" for c in family)
    output_csv = os.path.join(output_folder, f"{safe_family}_features.csv")
    
    # 创建DataFrame并保存
    df = pd.DataFrame(vectors, columns=columns)
    df.to_csv(output_csv, index=False)
    # print(f"Saved {len(vectors)} sequences for family {family}")

e = time()
print("Total time:", e - s)