import numpy as np
from scipy.optimize import linprog
import os
import pandas as pd

def intersection(mutset0, mutset00):
    m, l1 = mutset0.shape
    n, l2 = mutset00.shape
    if l1 != l2:
        print("Error: Dimensions do not match!")
        return None
    
    l = l1
    c = np.ones(m + n)
    A0 = np.hstack((mutset0.T, -mutset00.T))
    a1 = np.concatenate((np.ones(m), np.zeros(n)))
    b1 = np.concatenate((np.zeros(m), np.ones(n)))
    Aeq = np.vstack((A0, a1, b1))
    beq = np.hstack((np.zeros(l), 1, 1))
    bounds = [(0, 1)] * (m + n)
    
    # Using linprog to find the intersection
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds)
    
    if res.success:
        return 0
    else:
        return 1

# Specify the folder path
path = '/covid/52mnv' 

# Initialize a list to store intersection results
results = []

# Initialize a counter for clusters with at least 3 sequences
cluster_count = 0

# Iterate over each file in the folder
files = os.listdir(path)
for i, filename1 in enumerate(files):
    if filename1.endswith('.csv'):
        file_path1 = os.path.join(path, filename1)
        
        # Read the first CSV file
        df1 = pd.read_csv(file_path1)
        
        # Check if the number of sequences is at least 3
        if len(df1) < 3:
            continue
        
        cluster_count += 1
        
        vectors1 = df1.values
        
        # Compare with other files
        for filename2 in files[i + 1:]:  # Avoid redundant comparisons
            if filename2.endswith('.csv'):
                file_path2 = os.path.join(path, filename2)
                
                # Read the second CSV file
                df2 = pd.read_csv(file_path2)
                
                # Check if the number of sequences is at least 3
                if len(df2) < 3:
                    continue
                
                vectors2 = df2.values
                
                # Check intersection
                result = intersection(vectors1, vectors2)
                
                # Store the result
                results.append(result)
            

# Calculate the ratio of successful intersections
total_intersections = sum(results)
total_comparisons = len(results)
disjoint_ratio = total_intersections / total_comparisons if total_comparisons > 0 else 0

print(f"Number of clusters with at least 3 sequences: {cluster_count}")
print(f"Disjoints: {total_intersections:.4f}")
print(f"Disjoint ratio: {disjoint_ratio:.4f}")
