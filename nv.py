import math

# Function to calculate higher-order moments
def calculate_moments(sequence, avg_positions, seq_len, nucleotide_counts, max_k):  
    # Initialize dictionary to store moments for orders 2 to max_k
    moments = {k: [0, 0, 0, 0] for k in range(2, max_k + 1)}
    
    # Iterate through each position in the sequence
    for i, nt in enumerate(sequence):
        # Check each nucleotide type
        for j, k in enumerate("ACGT"):
            if nt == k:
                # Calculate moments from order 2 to max_k
                for n in range(2, max_k + 1):
                    if nucleotide_counts[j] > 0:
                        # Compute the n-th moment component
                        moments[n][j] += ((i + 1) - avg_positions[j]) * (
                            (i + 1 - avg_positions[j])/(seq_len * nucleotide_counts[j])
                        ) ** (n - 1)
    return moments

def is_valid_sequence(seq: str) -> bool:
    """Check if sequence contains only valid ACGT characters (case-insensitive)"""
    valid_chars = set("ACGTacgt")
    return all(char in valid_chars for char in seq)

def calculate_nv_vector(sequence: str, max_k: int = 12):
    # Validate sequence contains only ACGT characters
    if not is_valid_sequence(sequence):
        raise ValueError("Sequence contains non-ACGT characters")
    
    # Convert to uppercase for consistent processing
    sequence = sequence.upper()
    seq_len = len(sequence)
    
    # Calculate nucleotide counts
    nucleotide_counts = [
        sequence.count("A"),
        sequence.count("C"),
        sequence.count("G"),
        sequence.count("T")
    ]
    
    # Calculate mean positions for each nucleotide
    avg_positions = []
    for nt in "ACGT":
        positions = [(i + 1) for i, base in enumerate(sequence) if base == nt]
        # Avoid division by zero for missing nucleotides
        avg_positions.append(sum(positions) / len(positions) if positions else 0)
    
    # Compute higher-order moments
    moments = calculate_moments(sequence, avg_positions, seq_len, nucleotide_counts, max_k)
    
    # Construct natural vector: [counts] + [mean positions] + [moments]
    nv_vector = nucleotide_counts + avg_positions
    for n in range(2, max_k + 1):
        nv_vector.extend(moments[n])
    
    return nv_vector
