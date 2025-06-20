import math
from Bio import SeqIO
import pandas as pd
from time import time
import os
from math import cos

# 定义计算各阶矩的函数
def calculate_moments(sequence, avg_positions, seq_len, nucleotide_counts, max_k):  
    moments = {k: [0, 0, 0, 0] for k in range(2, max_k + 1)}
    for i, nt in enumerate(sequence):
        for j, k in enumerate("ACGT"):
            if nt == k:
                for n in range(2, max_k + 1):
                    if nucleotide_counts[j] > 0:
                        moments[n][j] += ((i + 1) - avg_positions[j]) *  ((i + 1 - avg_positions[j])/(seq_len * nucleotide_counts[j])) ** (n - 1)  #
    return moments

# 开始计时
s = time()

# 指定输入和输出文件夹
input_folder = "/home/shixiang/shixiang/natural_vector/covid_ncbi"
output_folder = "/home/shixiang/shixiang/multiview_review/covid_ncbi/52nv"

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
def is_valid_sequence(seq: str) -> bool:
    """检查序列是否只包含ACGT字符（不区分大小写）"""
    valid_chars = set("ACGTacgt")
    return all(char in valid_chars for char in seq)

for filename in os.listdir(input_folder):
    # 检查文件是否以 '.fasta' 结尾
    if filename.endswith(".fasta"):
        # 拼接输入文件的完整路径
        fasta_file = os.path.join(input_folder, filename)

        # 读取 FASTA 文件
        sequences = []
        sequence_names = []

        for record in SeqIO.parse(fasta_file, "fasta"):
            seq_str = str(record.seq)
            
            if not is_valid_sequence(seq_str):
                continue
                
            # 仅保留不重复的序列
            if seq_str not in sequences:
                sequences.append(seq_str)
                sequence_names.append(record.id)

        with open(os.path.join(output_folder, "unique_sequences.txt"), "a") as f:
            f.write(f"File: {filename}\n")
            f.writelines([f"{name}\n" for name in sequence_names])

        # 存储每条序列的特征向量
        nucleotide_vectors = []

        # 设置最大阶数
        max_k = 12

        # 遍历每条序列
        for sequence in sequences:
            nucleotide_counts = [sequence.count("A"), sequence.count("C"), sequence.count("G"), sequence.count("T")]
            seq_len = len(sequence)
            avg_positions = []
            for nt in "ACGT":
                positions = [(i + 1) for i, base in enumerate(sequence) if base == nt]
                avg_positions.append(sum(positions) / len(positions) if positions else 0)
            
            # 计算各阶矩
            moments = calculate_moments(sequence, avg_positions, seq_len, nucleotide_counts, max_k)  

            nucleotide_vector = nucleotide_counts + avg_positions
            for n in range(2, max_k + 1):
                nucleotide_vector += moments[n]

            nucleotide_vectors.append(nucleotide_vector)

        # 定义列名
        columns = ['A_count', 'C_count', 'G_count', 'T_count', 'A_avg_position', 'C_avg_position', 'G_avg_position', 'T_avg_position']
        for n in range(2, max_k + 1):
            columns += [f'A_D_{n}', f'C_D_{n}', f'G_D_{n}', f'T_D_{n}']

        # 将结果存储在 pandas DataFrame 中
        df_nucleotide_vectors = pd.DataFrame(nucleotide_vectors, columns=columns)

        # 将 DataFrame 存储到 CSV 文件
        output_csv = os.path.join(output_folder, filename.split('.')[0] + ".csv")
        df_nucleotide_vectors.to_csv(output_csv, index=False)

# 结束计时
e = time()
print("Total time:", e - s)
