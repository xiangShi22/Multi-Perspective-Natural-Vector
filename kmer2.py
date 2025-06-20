import math
from Bio import SeqIO
import pandas as pd
from time import time
import os

# 定义计算Kmer的函数
def Kmer(sequence, K):
    m = 4**K
    na_vect = [0] * m
    n = len(sequence) - (K - 1)
    index_map = { 'a': 0, 'A': 0, 'c': 1, 'C': 1, 'g': 2, 'G': 2, 't': 3, 'T': 3 }
    
    for i in range(0, n):
        flag = 1
        for l in range(0, K):
            if sequence[i + l] not in index_map.keys():
                flag = 0
        if flag == 0:
            continue
        
        tem = index_map[sequence[i]]
        for l in range(1, K):
            tem = 4 * tem + index_map[sequence[i + l]]
        na_vect[tem] += 1
    
    return na_vect

# 开始计时
s = time()

# 指定输入和输出文件夹
input_folder = "/home/shixiang/shixiang/natural_vector/poliovirus"
output_folder = "/home/shixiang/shixiang/multiview_review/poliovirus/2mer"

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 存储不重复序列名称
unique_sequences_names = []

# 添加ACGT筛选函数
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

        # 存储每条序列的特征向量
        nucleotide_vectors = []

        # 遍历每条序列
        for sequence in sequences:
            nucleotide_vector = Kmer(sequence, 2)
            nucleotide_vectors.append(nucleotide_vector)

        # 将结果存储在 pandas DataFrame 中
        df_nucleotide_vectors = pd.DataFrame(nucleotide_vectors)
        
        # 创建列名：所有可能的3-mer组合
        nucleotides = ['A', 'C', 'G', 'T']
        kmer_columns = []
        for i in nucleotides:
            for j in nucleotides:
                for k in nucleotides:
                    kmer_columns.append(f"{i}{j}{k}")
        
        # 确保列数匹配
        if len(df_nucleotide_vectors.columns) == len(kmer_columns):
            df_nucleotide_vectors.columns = kmer_columns

        # 将 DataFrame 存储到 CSV 文件
        output_csv = os.path.join(output_folder, filename.split('.')[0] + ".csv")
        df_nucleotide_vectors.to_csv(output_csv, index=False)

# 结束计时
e = time()
print("Total time:", e - s)