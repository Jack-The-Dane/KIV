import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from scipy.signal import butter, filtfilt

from cnn_improved import run_config, grid_search, DEFAULT_PARAMS

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
class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size=64, kernel_size=5):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

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
# TRAINING UTILITIES
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch.to(DEVICE))
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

def train_model(X_train, y_train, X_val, y_val, params, epochs=20):
    train_loader = DataLoader(GestureDataset(X_train, y_train, is_train=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(GestureDataset(X_val, y_val), batch_size=BATCH_SIZE)
    
    model = CNN_LSTM(hidden_size=params.get('hidden_size', 64), 
                     kernel_size=params.get('kernel_size', 5)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 1e-3))
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer)
    
    return evaluate_model(model, val_loader)

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    metrics = train_model(X_train, y_train, X_test, y_test, {'lr': 1e-3, "hidden_size": 64})
    metrics['method'] = 'Holdout'
    results.append(metrics)

    # 2. CROSS-VALIDATION (5-Fold)
    print("Running 5-Fold Cross-Validation...")
    kf = KFold(n_splits=5, shuffle=True)
    cv_metrics = []
    for train_idx, val_idx in kf.split(X):
        m = train_model(X[train_idx], y[train_idx], X[val_idx], y[val_idx], {'lr': 1e-3, "hidden_size": 64})
        cv_metrics.append(list(m.values()))
    
    avg_cv = np.mean(cv_metrics, axis=0)
    results.append({'accuracy': avg_cv[0], 'precision': avg_cv[1], 'recall': avg_cv[2], 'f1': avg_cv[3], 'method': 'k-Fold CV'})

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