import numpy as np

def create_sequences(data, seq_length=30):
    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        
        # Target variable is assumed to be at index 1
        y.append(data[i + seq_length][1])  

    return np.array(X), np.array(y)