import os
import hashlib
from time import time
from typing import List, Tuple

import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
from torch import Tensor

start_time = time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

input_folder = '/data/covid'
output_folder = '/embedding/covid/52mnv'
os.makedirs(output_folder, exist_ok=True)

@torch.no_grad()
def calculate_counts(seq: Tensor):
    L = seq.shape[0]
    one_hot = torch.nn.functional.one_hot(seq, num_classes=4).float()
    counts = one_hot.sum(dim=0)
    positions = torch.arange(1, L + 1, device=seq.device, dtype=torch.float32)

    avg_pos = (one_hot.T @ positions) / counts.clamp(min=1e-7)

    norm_pos = positions / L * 2 * torch.pi
    cos_vals, sin_vals = torch.cos(norm_pos), torch.sin(norm_pos)
    avg_cos = (one_hot.T @ cos_vals) / counts.clamp(min=1e-7)
    avg_sin = (one_hot.T @ sin_vals) / counts.clamp(min=1e-7)

    return counts, avg_pos, avg_cos, avg_sin, one_hot, positions, norm_pos

@torch.no_grad()
def calculate_moments(
    seq: Tensor,
    counts: Tensor,
    avg_pos: Tensor,
    avg_cos: Tensor,
    avg_sin: Tensor,
    one_hot: Tensor,
    positions: Tensor,
    norm_pos: Tensor,
    k_values: List[int] = [2, 3, 4]
) -> Tuple[Tensor, Tensor, Tensor]:
    L = seq.shape[0]
    k_tensor = torch.tensor(k_values, device=seq.device, dtype=torch.float32)
    pos_diff = positions[:, None] - avg_pos[None, :]
    denominator = L * counts.clamp(min=1e-7)

    scaled_diff = pos_diff.unsqueeze(-1) / denominator[None, :, None]
    power_terms = scaled_diff ** (k_tensor - 1)
    full_terms = pos_diff.unsqueeze(-1) * power_terms
    masked_terms = full_terms * one_hot.unsqueeze(-1)
    raw_moments = masked_terms.sum(dim=0)

    cos_diff = torch.cos(norm_pos)[:, None, None] - avg_cos[None, :, None]
    sin_diff = torch.sin(norm_pos)[:, None, None] - avg_sin[None, :, None]

    cos_moments = (cos_diff ** k_tensor * one_hot.unsqueeze(-1)).sum(dim=0)
    sin_moments = (sin_diff ** k_tensor * one_hot.unsqueeze(-1)).sum(dim=0)

    cos_moments /= counts[:, None].clamp(min=1e-7)
    sin_moments /= counts[:, None].clamp(min=1e-7)

    return raw_moments, cos_moments, sin_moments

def convert(sequence: str) -> Tensor:
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    indices = [mapping[c.upper()] if c.upper() in mapping else -1 for c in sequence]
    if -1 in indices:
        raise ValueError("Invalid nucleotide found.")
    return torch.tensor(indices, device=device, dtype=torch.long)

max_k = 4
columns = [
    'A_count', 'C_count', 'G_count', 'T_count',
    'A_avg_pos', 'C_avg_pos', 'G_avg_pos', 'T_avg_pos',
    'A_avg_pos_cos', 'C_avg_pos_cos', 'G_avg_pos_cos', 'T_avg_pos_cos',
    'A_avg_pos_sin', 'C_avg_pos_sin', 'G_avg_pos_sin', 'T_avg_pos_sin'
]

for k in range(2, max_k + 1):
    for prefix in ['', 'cos_', 'sin_']:
        columns.extend([f'{nt}_{prefix}D_{k}' for nt in ['A', 'C', 'G', 'T']])

for filename in os.listdir(input_folder):
    if not filename.endswith(".fasta"):
        continue

    print(f"Processing {filename}...")
    fasta_path = os.path.join(input_folder, filename)
    sequences, names, seen_hashes = [], [], set()

    for record in SeqIO.parse(fasta_path, "fasta"):
        try:
            seq_tensor = convert(str(record.seq))
        except ValueError:
            continue

        seq_hash = hashlib.sha256(seq_tensor.cpu().numpy().tobytes()).hexdigest()
        if seq_hash not in seen_hashes:
            seen_hashes.add(seq_hash)
            sequences.append(seq_tensor)
            names.append(record.id)

    if not sequences:
        continue

    features = []
    for seq in sequences:
        c, avg_p, avg_cos, avg_sin, one_hot, pos, norm_pos = calculate_counts(seq)
        raw_m, cos_m, sin_m = calculate_moments(seq, c, avg_p, avg_cos, avg_sin, one_hot, pos, norm_pos, list(range(2, max_k + 1)))
        feat = [
            c.cpu().numpy(), avg_p.cpu().numpy(),
            avg_cos.cpu().numpy(), avg_sin.cpu().numpy(),
            raw_m.T.flatten().cpu().numpy(),
            cos_m.T.flatten().cpu().numpy(),
            sin_m.T.flatten().cpu().numpy()
        ]
        features.append(np.concatenate(feat))

    df = pd.DataFrame(features, columns=columns)
    df.to_csv(os.path.join(output_folder, filename.replace('.fasta', '.csv')), index=False)

    with open(os.path.join(output_folder, "unique_sequences.txt"), "a") as f:
        f.write(f"File: {filename}\n")
        f.writelines([f"{name}\n" for name in names])

print("Total time:", time() - start_time)
