import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
def load_txt_from_zip(zip_path):
    arrays = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        txt_files = sorted([f for f in zf.namelist() if f.lower().endswith('.txt')])
        print(f"Found {len(txt_files)} text files in {zip_path}")
        
        for f in txt_files:
            with zf.open(f) as file:
                try:
                    data = np.genfromtxt(file, dtype=float)
                    if data.size == 0:
                        print(f"Warning: {f} is empty")
                        continue
                    arrays.append(data)
                except Exception as e:
                    print(f"Failed to load {f}: {e}")
    return arrays

def prepare_binary_dataset(class0_sets, class1_sets):
    """
    Combine multiple sets into a binary dataset.
    class0_sets / class1_sets: lists of NumPy arrays
    Returns: X, y
    """
    # Flatten each sample if needed and stack
    X0 = [x.flatten() for s in class0_sets for x in s]
    X1 = [x.flatten() for s in class1_sets for x in s]
    
    X = np.array(X0 + X1)
    y = np.array([0]*len(X0) + [1]*len(X1))
    
    # Shuffle dataset
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    
    return X, y

# Example: load all sets
setA = load_txt_from_zip("Z.zip")
setB = load_txt_from_zip("O.zip")
setC = load_txt_from_zip("N.zip")
setD = load_txt_from_zip("F.zip")
setE = load_txt_from_zip("S.zip")

