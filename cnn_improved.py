"""
CNN + LSTM for Epileptic Seizure Detection (Bonn EEG Dataset)

Requirements:
- numpy
- scipy
- torch
- scikit-learn

Dataset structure (example):
data/
    A/  (healthy)
    B/  (healthy)
    C/  (interictal - optional)
    D/  (interictal - optional)
    E/  (seizure)

We will do binary classification:
- Healthy: A, B
- Seizure: E
"""

import os
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "./dataset"   # change to your dataset path
SAMPLING_RATE = 173.61  # Bonn dataset sampling rate (Hz)
SEGMENT_LENGTH = 178  # samples (~1 second)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# BANDPASS FILTER
# -----------------------------
def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=173.61, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(data_dir):
    X = []
    y = []

    # Healthy classes
    healthy_folders = ["A", "B"]
    seizure_folders = ["E"]

    for folder in healthy_folders:
        folder_path = os.path.join(data_dir, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                signal = np.loadtxt(os.path.join(folder_path, file))
                X.append(signal)
                y.append(0)

    for folder in seizure_folders:
        folder_path = os.path.join(data_dir, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                signal = np.loadtxt(os.path.join(folder_path, file))
                X.append(signal)
                y.append(1)

    return X, y

# -----------------------------
# PREPROCESSING
# -----------------------------
def preprocess_signals(signals):
    processed = []

    for sig in signals:
        # Bandpass filter
        sig = bandpass_filter(sig, fs=SAMPLING_RATE)

        # Normalize (z-score)
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

        processed.append(sig)

    return processed

# -----------------------------
# SEGMENTATION
# -----------------------------
def segment_signal(signal, segment_length, overlap=0.75):
    step_size = int(segment_length * (1 - overlap))
    segments = []
    for i in range(0, len(signal) - segment_length + 1, step_size):
        segments.append(signal[i:i+segment_length])
    return segments

def create_dataset(signals, labels):
    X_segments = []
    y_segments = []

    for sig, label in zip(signals, labels):
        segments = segment_signal(sig, SEGMENT_LENGTH)
        for seg in segments:
            X_segments.append(seg)
            y_segments.append(label)

    return np.array(X_segments), np.array(y_segments)

# -----------------------------
# DATASET CLASS
# -----------------------------
class EEGDataset(Dataset):
    def __init__(self, X, y, is_train = False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.is_train = is_train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.is_train:
            noise = torch.randn(x.size()) * 0.01 # 1% noise level
            x = x + noise
        return x.unsqueeze(0), self.y[idx]

# -----------------------------
# MODEL (CNN + LSTM)
# -----------------------------
class CNN_LSTM(nn.Module):
    def __init__(self, kernel_size1:int = 5, kernel_size2:int = 5, cnn_internal:int = 16, cnn_out:int = 32, hidden_size:int = 64, num_layers:int = 1):
        super(CNN_LSTM, self).__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_internal, kernel_size=kernel_size1, stride=1, padding=kernel_size1//2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(cnn_internal, cnn_out, kernel_size=kernel_size2, padding=kernel_size2//2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Fully connected
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x: (batch, 1, seq_len)
        x = self.cnn(x)

        # reshape for LSTM: (batch, seq, features)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        # take last timestep
        out = lstm_out[:, -1, :]

        out = self.fc(out)
        return out

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

# -----------------------------
# TRAIN FUNCTION
# -----------------------------
def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# -----------------------------
# VALIDATION FUNCTION
# -----------------------------
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# -----------------------------
# EVALUATION
# -----------------------------
def evaluate(model:CNN_LSTM, test_loader, model_str:str=""):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)

            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).to(DEVICE).numpy()

            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    print(f"{"For model with parameters:":<30} {"Classification Report:":<60} {"confusion_matrix"}")
    rep = str(classification_report(all_labels, all_preds)).splitlines()
    conf = str(confusion_matrix(all_labels, all_preds)).splitlines()
    mod = model_str.splitlines()
    longest = max(len(mod), len(rep))
    for i in range(longest):
        print(f"{mod[i] if i < len(mod) else "" :<30} {rep[i] if i < len(rep) else "" :<60} {conf[i] if i < len(conf) else ""}")

def evaluate_with_return(model:CNN_LSTM, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)

            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

def train_epoch(model, train_loader, val_loader, criterion, optimizer):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_loss = validate(model, val_loader, criterion)
    return train_loss, val_loss
    

def grid_search(signals, labels, parameters:dict, epochs:int=20):
    par_kernel1 = parameters.get("kernel_size1", [5])
    par_kernel2 = parameters.get("kernel_size2", [5])
    par_internal = parameters.get("cnn_internal", [16])
    par_cnn_out = parameters.get("cnn_out", [32])
    par_hidden = parameters.get("hidden_size", [64])
    par_layers = parameters.get("num_layers", [1])

    # X_train, X_test_val, y_train, y_test_val = train_test_split(
    #     signals, labels, test_size=0.3, random_state=42, stratify=labels
    # )

    # X_test, X_val, y_test, y_val = train_test_split(
    #     X_test_val, y_test_val, test_size=0.5, random_state=42, stratify=y_test_val
    # )

    signals, X_test, labels, y_test = train_test_split(signals, labels, test_size=0.1, random_state=42, stratify=labels)

    #X_train, y_train = create_dataset(X_train, y_train)
    X_test, y_test = create_dataset(X_test, y_test)
    #X_val, y_val = create_dataset(X_val, y_val)

    #train_loader = DataLoader(EEGDataset(X_train, y_train, is_train=True), batch_size=BATCH_SIZE, shuffle=True)
    #val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=BATCH_SIZE, num_workers=4)

    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    results = []
    print("Starting grid search...")
    for k1 in par_kernel1:
        for k2 in par_kernel2:
            for internal in par_internal:
                for out in par_cnn_out:
                    for hidden in par_hidden:
                        for layers in par_layers:
                            metrics = []
                            for train_idx, val_idx in kfold.split(signals, labels):
                                X_train = signals[train_idx]
                                y_train = labels[train_idx]
                                X_test = signals[val_idx]
                                y_test = labels[val_idx]
                                X_train, y_train = create_dataset(X_train, y_train)
                                X_test, y_test = create_dataset(X_test, y_test)
                                train_loader = DataLoader(EEGDataset(X_train, y_train, is_train=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
                                val_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=BATCH_SIZE, num_workers=4)
                                model = CNN_LSTM(k1, k2, internal, out, hidden, layers).to(DEVICE)
                                weights = torch.tensor([1.0, 2.0]).to(DEVICE)
                                criterion = nn.CrossEntropyLoss(weight=weights).to(DEVICE)
                                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
                                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
                                early_stopping = EarlyStopping(patience=5, path='best_EEG_grid.pt')
                                for epoch in range(epochs):
                                    train_loss, val_loss = train_epoch(model, train_loader, val_loader, criterion, optimizer)
                                    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

                                    # Update Scheduler and check Early Stopping
                                    scheduler.step(val_loss)
                                    early_stopping(val_loss, model)

                                    if early_stopping.early_stop:
                                        print("Early stopping triggered. Training halted.")
                                        break
                                # print("\n--- Final Evaluation (Best Model) ---")
                                model.load_state_dict(torch.load('best_EEG_grid.pt'))
                                m = evaluate_with_return(model, test_loader)
                                print(m)
                                metrics.append(list(m.values()))
                            model_str = f"k1: {k1}\nk2: {k2}\ncnn_internal: {internal}\ncnn_out: {out}\nhidden: {hidden}\nlayers: {layers}"
                            avg_cv = np.mean(metrics, axis=0)
                            results.append({"parameters": model_str, 'accuracy': avg_cv[0], 'precision': avg_cv[1], 'recall': avg_cv[2], 'f1': avg_cv[3], 'method': 'grid-search + 5-Fold CV'})   
                            export(results)
                            return                     
    #pd.DataFrame(results).to_csv("GridSearch_results.csv") 

def export(results):
    pd.DataFrame(results).to_csv("GridSearch_results.csv", index=False) 

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    print("Loading data...")
    signals, labels = load_data(DATA_DIR)

    print("Preprocessing...")
    signals = preprocess_signals(signals)

    signals = np.array(signals)
    labels = np.array(labels)

    # Create split first, and then segment, to avoid data leakage
    print("Train/Test split...")
    X_train, X_test_val, y_train, y_test_val = train_test_split(
        signals, labels, test_size=0.3, random_state=42, stratify=labels
    )

    X_test, X_val, y_test, y_val = train_test_split(
        X_test_val, y_test_val, test_size=0.5, random_state=42, stratify=y_test_val
    )

    print("Segmenting...")
    X_train, y_train = create_dataset(X_train, y_train)
    X_test, y_test = create_dataset(X_test_val, y_test_val)
    X_val, y_val = create_dataset(X_val, y_val)

    train_loader = DataLoader(EEGDataset(X_train, y_train, is_train=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=BATCH_SIZE)

    model = CNN_LSTM().to(DEVICE)

    weights = torch.tensor([1.0, 2.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    early_stopping = EarlyStopping(patience=5, path='best_EEG.pt')

    # 4. Training Loop
    # print("--- Starting Training ---")
    # for epoch in range(EPOCHS):
    #     train_loss = train(model, train_loader, criterion, optimizer)
    #     val_loss = validate(model, val_loader, criterion)
        
    #     print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    #     # Update Scheduler and check Early Stopping
    #     scheduler.step(val_loss)
    #     early_stopping(val_loss, model)

    #     if early_stopping.early_stop:
    #         print("Early stopping triggered. Training halted.")
    #         break

    # # 5. Final Evaluation on Test Set
    # print("\n--- Final Evaluation (Best Model) ---")
    # model.load_state_dict(torch.load('best_cnn_lstm.pt'))
    # evaluate(model, test_loader)

    print("Running grid search...")
    parameters = {
        "kernel_size1": [5, 9],
        "kernel_size2": [5, 9],
        "cnn_internal": [16, 32],
        "cnn_out": [32, 64],
        "hidden_size": [64, 128],
        "num_layers": [1, 2]
    }
    grid_search(signals, labels, parameters, 20)

if __name__ == "__main__":
    main()