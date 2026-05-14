import pandas as pd
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy
from math import factorial, log2

# --- Entropy and Complexity Implementations (Replacing 'ant') ---

def _embed(x, order=3, delay=1):
    """Time-delay embedding."""
    N = len(x)
    Y = np.empty((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay : i * delay + Y.shape[1]]
    return Y.T

def perm_entropy(x, order=3, delay=1, normalize=True):
    """Permutation Entropy."""
    x = np.array(x)
    ran_ki = np.array([np.argsort(x[i:i+order*delay:delay]) for i in range(len(x) - (order-1)*delay)])
    _, counts = np.unique(ran_ki, axis=0, return_counts=True)
    probs = counts / counts.sum()
    pe = -np.sum(probs * np.log2(probs))
    if normalize:
        return pe / log2(factorial(order))
    return pe

def spectral_entropy(x, sf, method='welch', nperseg=None, normalize=True):
    """Spectral Entropy."""
    _, psd = welch(x, sf, nperseg=nperseg)
    psd_norm = psd / psd.sum()
    se = -(psd_norm * np.log2(psd_norm + 1e-12)).sum()
    if normalize:
        return se / np.log2(psd_norm.size)
    return se

def svd_entropy(x, order=3, delay=1, normalize=True):
    """SVD Entropy."""
    embedded = _embed(x, order, delay)
    W = np.linalg.svd(embedded, compute_uv=False)
    W /= np.sum(W)
    svd_e = -np.sum(W * np.log2(W))
    if normalize:
        return svd_e / np.log2(order)
    return svd_e

def sample_entropy(x, order=2, metric='chebyshev'):
    """Sample Entropy (Simplified Implementation)."""
    x = np.array(x)
    def _phi(m):
        x_mat = _embed(x, m, 1)
        # Calculate distance matrix (Chebyshev)
        diff = np.abs(x_mat[:, None, :] - x_mat[None, :, :]).max(axis=2)
        r = 0.2 * np.std(x)
        count = (diff <= r).sum() - len(x_mat)
        return count / (len(x_mat) * (len(x_mat) - 1))
    
    phi_m = _phi(order)
    phi_m1 = _phi(order + 1)
    return -np.log(phi_m1 / phi_m) if phi_m1 != 0 else 0

def lziv_complexity(sequence, normalize=True):
    """Lempel-Ziv complexity for binary sequences."""
    if isinstance(sequence, np.ndarray):
        sequence = "".join(sequence.astype(str))
    
    i, n = 1, len(sequence)
    u, v, w = 0, 1, 1
    complexity = 1
    while i + v <= n:
        if sequence[i : i + v] == sequence[u : u + v]:
            v += 1
        else:
            w = max(v, w)
            complexity += 1
            i += v
            u, v = 0, 1
    
    if normalize:
        return (complexity * np.log2(n)) / n
    return complexity

# --- Statistical and Time Domain Placeholders ---
# (Added to ensure the calls in your notebook logic work)

def extract_statistical_features(signal):
    return {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'min': np.min(signal),
        'max': np.max(signal)
    }

def extract_time_domain_features(signal):
    return {
        'rms': np.sqrt(np.mean(signal**2)),
        'zero_crossing': ((signal[:-1] * signal[1:]) < 0).sum()
    }

# --- Main Logic from Notebook ---

def extract_entropy_features(signal):
    signal = np.array(signal, dtype=np.float64, order='C')
    fs = 100

    # Replacement for ant calls
    se_val = sample_entropy(signal)
    spec_val = spectral_entropy(signal, sf=fs, method='welch')
    perm_val = perm_entropy(signal, normalize=True)
    svd_val = svd_entropy(signal, order=3, normalize=True)

    binary_signal = (signal > np.median(signal)).astype(int)
    lziv_val = lziv_complexity(binary_signal, normalize=True)

    return {
        'Sample Entropy': se_val,
        'Spectral Entropy': spec_val,
        'Permutation Entropy': perm_val,
        'SVD Entropy': svd_val,
        'LZiv Complexity': lziv_val
    }

def main():
    # 1. Merge Data
    def merge_with_labels(file1, file2, label1, label2):
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        df2.columns = df1.columns
        df1['label'] = label1
        df2['label'] = label2
        merged_df = pd.concat([df1, df2], axis=0, ignore_index=True)
        return merged_df

    # Note: Replace with your actual file paths
    combined_data = merge_with_labels("GestureClassification/PaperData.csv", "GestureClassification/PaperData.csv", 0, 1)
    
    # Example Dummy Data for demonstration
    #combined_data = pd.DataFrame(np.random.rand(10, 1025), columns=[f'c{i}' for i in range(1024)] + ['label'])

    # 2. Split Features/Target
    X = combined_data.drop(columns=['label']).apply(pd.to_numeric, errors='coerce')
    y = combined_data['label']

    # 3. Feature Extraction
    print("Extracting features...")
    stat_features = [extract_statistical_features(row.values) for _, row in X.iterrows()]
    time_features = [extract_time_domain_features(row.values) for _, row in X.iterrows()]
    entropy_features = [extract_entropy_features(row.values) for _, row in X.iterrows()]

    # 4. Create Final DataFrame
    df_final = pd.concat([
        pd.DataFrame(stat_features),
        pd.DataFrame(time_features),
        pd.DataFrame(entropy_features),
        y.reset_index(drop=True)
    ], axis=1)

    print("Data Preparation Complete.")
    print(df_final.head())
    df_final.to_csv("training_data.csv", index=False)

if __name__ == "__main__":
    main()