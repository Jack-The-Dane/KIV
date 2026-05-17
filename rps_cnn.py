import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from scipy.signal import butter, filtfilt

from cnn_improved import run_config, grid_search, DEFAULT_PARAMS, hold_out_validation, export, kfold_cv

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "./GestureClassification"
SAMPLING_RATE = 256
SEGMENT_LENGTH = 768  # 256 + 512
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA1 = "PaperData.csv"
DATA2 = "RockData.csv"

# -----------------------------
# MODEL & COMPONENTS
# -----------------------------

class GestureDataset(Dataset):
    def __init__(self, X, y, is_train=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.is_train = is_train
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.is_train:
            x = x + torch.randn(x.size()) * 0.01
        return x.unsqueeze(0), self.y[idx]

# -----------------------------
# DATA LOADING (Simplified from your script)
# -----------------------------
def load_and_preprocess():
    df0 = pd.read_csv(os.path.join(DATA_DIR, DATA1), header=None)
    df1 = pd.read_csv(os.path.join(DATA_DIR, DATA2), header=None)
    df0['label'], df1['label'] = 0, 1
    df = pd.concat([df0, df1], axis=0)
    X = df.drop(columns=['label']).to_numpy()
    y = df['label'].to_numpy()
    # Normalize
    X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)
    return X, y

# -----------------------------
# MAIN VALIDATION PIPELINE
# -----------------------------
def main():
    X, y = load_and_preprocess()
    results = []

    # 1. HOLDOUT VALIDATION
    print("Running Holdout Validation...")
    ## Maybe do segmentation here as well?
    metrics = hold_out_validation(X, y, params=DEFAULT_PARAMS, DataClass=GestureDataset)
    export(metrics, "holdout_rps")

    # 2. CROSS-VALIDATION (5-Fold)
    print("Running 5-Fold Cross-Validation...")
    metrics = kfold_cv([DEFAULT_PARAMS], X, y, GestureDataset) # Probably do this on best parameters of EEG, once that is done
    export(metrics, "kfold_rps")

    # 3. GRID SEARCH VALIDATION
    print("Running Grid Search...")
    param_grid = {
        'lr': [1e-2, 1e-3, 1e-4],
        'hidden_size': [32, 64, 128],
        "kernel_size": [5, 11, 15]
    }
    grid = ParameterGrid(param_grid)
    best_f1 = -1
    best_params = {}
    
    for p in grid:
        m = train_model(X_train, y_train, X_test, y_test, p, epochs=20)
        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_params = p
            best_metrics = m

    best_metrics['method'] = f"Grid Search (Best: {best_params})"
    results.append(best_metrics)

    # SAVE RESULTS
    results_df = pd.DataFrame(results)
    results_df.to_csv("validation_results.csv", index=False)
    print("\nValidation Complete. Results saved to 'validation_results.csv'.")
    print(results_df)

if __name__ == "__main__":
    main()