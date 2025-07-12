import torch
import numpy as np
from torch import Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def compute_feature_vector(sequence: str, max_k: int = 4) -> np.ndarray:
    """Compute the feature vector for a single DNA sequence"""
    seq_tensor = convert(sequence)
    c, avg_p, avg_cos, avg_sin, one_hot, pos, norm_pos = calculate_counts(seq_tensor)
    raw_m, cos_m, sin_m = calculate_moments(
        seq_tensor, c, avg_p, avg_cos, avg_sin, 
        one_hot, pos, norm_pos, list(range(2, max_k + 1))
    
    return np.concatenate([
        c.cpu().numpy(), 
        avg_p.cpu().numpy(),
        avg_cos.cpu().numpy(), 
        avg_sin.cpu().numpy(),
        raw_m.T.flatten().cpu().numpy(),
        cos_m.T.flatten().cpu().numpy(),
        sin_m.T.flatten().cpu().numpy()
    ])
