import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from scipy.signal import butter, filtfilt

from cnn_improved import run_config, grid_search, DEFAULT_PARAMS, hold_out_validation, export, kfold_cv, PARAMETER_MAMES

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
    df0['label'] = 0
    df1['label'] = 1
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
    # print("Running Holdout Validation...")
    # top_params = pd.read_csv("top_model.csv")
    # params = top_params.to_dict("records")[0]
    # print(params)
    # metrics = hold_out_validation(X, y, params=params, DataClass=GestureDataset, epochs=50)
    # export([metrics], "holdout_rps")
    
    # 2. CROSS-VALIDATION (5-Fold)
    print("Running 5-Fold Cross-Validation...")
    params_df = pd.read_parquet("results/kfold-top5_eeg.parquet")
    params_df = params_df[PARAMETER_MAMES]
    params = params_df.to_dict("records")
    metrics = kfold_cv(params, X, y, GestureDataset, epochs=50) # Probably do this on best parameters of EEG, once that is done
    export(metrics, "kfold_top5_models_rps")
    
    # 3. GRID SEARCH VALIDATION

    #print("Running Grid Search...")
    # param_grid = {
    #     'lr': [1e-2, 1e-3, 1e-4],
    #     'hidden_size': [32, 64, 128],
    #     "kernel_size": [5, 11, 15]
    # }
    # test_params = pd.read_csv("test_params.csv")
    # params = {key: pd.unique(test_params[key]).tolist() for key in test_params.columns}
    # params["batch_size"] = [32, 64, 128]
    #print(params)
    #return
    # grid_res = grid_search(X, y, params, 50, GestureDataset)
    # export(grid_res, "batch_test")
    #print(grid_res)
    return
    # SAVE RESULTS
    results_df = pd.DataFrame(results)
    results_df.to_csv("validation_results.csv", index=False)
    print("\nValidation Complete. Results saved to 'validation_results.csv'.")
    print(results_df)

if __name__ == "__main__":
    main()