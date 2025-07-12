# Function to calculate K-mer frequencies
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
