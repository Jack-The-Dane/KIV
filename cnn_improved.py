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
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

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
def load_data(data_dir, folders:list[str], label:int):
    X = []
    y = []

    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                signal = np.loadtxt(os.path.join(folder_path, file))
                X.append(signal)
                y.append(label)


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
def segment_signal(signal, segment_length, overlap):
    step_size = int(segment_length * (1 - overlap))
    segments = []
    for i in range(0, len(signal) - segment_length + 1, step_size):
        segments.append(signal[i:i+segment_length])
    return segments

def create_dataset(signals, labels, overlap):
    X_segments = []
    y_segments = []

    for sig, label in zip(signals, labels):
        segments = segment_signal(sig, SEGMENT_LENGTH, overlap)
        for seg in segments:
            X_segments.append(seg)
            y_segments.append(label)

    return np.array(X_segments), np.array(y_segments)

# -----------------------------
# DATASET CLASS
# -----------------------------
class EEGDataset(Dataset):
    def __init__(self, X, y, is_train = False):
        X_data, y_data = create_dataset(X, y, overlap=0.75 if is_train else 0.0)
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.long)
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
DEFAULT_PARAMS = {
        "kernel_size1": [5],
        "kernel_size2": [5],
        "cnn_internal": [16],
        "cnn_out": [32],
        "hidden_size": [64],
        "num_layers": [1],
        "dropout": [0.0],
        "learning_rate": [1e-3],
        "weight_decay": [1e-4]
    }
class CNN_LSTM(nn.Module):
    def __init__(self, parameters:dict = {}):
        super(CNN_LSTM, self).__init__()
        kernel_size1 = parameters.get("kernel_size1", 5)
        kernel_size2 = parameters.get("kernel_size2", 5)
        cnn_internal = parameters.get("cnn_internal", 16)
        cnn_out = parameters.get("cnn_out", 32)
        hidden_size = parameters.get("hidden_size", 64)
        num_layers = parameters.get("num_layers", 1)
        dropout = parameters.get("dropout", 0.0)
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_internal, kernel_size=kernel_size1, stride=1, padding=kernel_size1//2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(cnn_internal, cnn_out, kernel_size=kernel_size2, padding=kernel_size2//2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
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
            #print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
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
    conf = confusion_matrix(all_labels, all_preds, normalize="all")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "c11":conf[0][0], "c12":conf[0][1], "c21":conf[1][0], "c22":conf[1][1]}

def train_epoch(model, train_loader, val_loader, criterion, optimizer):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_loss = validate(model, val_loader, criterion)
    return train_loss, val_loss

def train_model(model, criterion, optimizer, scheduler, early_stopping, train_loader, val_loader, epochs = 50):
    losses = {}
    losses["val_loss"] = []
    losses["train_loss"] = []
    
    for epoch in range(epochs):
        train_loss, val_loss = train_epoch(model, train_loader, val_loader, criterion, optimizer)
        #print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        # Update Scheduler and check Early Stopping
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        losses["val_loss"].append(val_loss)
        losses["train_loss"].append(train_loss)
        if early_stopping.early_stop:
            #print("Early stopping triggered. Training halted.")
            break

    return losses

def init_train_eval_model(model_params:dict, train_set:EEGDataset, val_set:EEGDataset, test_set:EEGDataset, model_name:str, epochs):
    results = model_params

    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=0)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=0, shuffle=True) # Shuffle training data each epoch
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=0)
    
    model = CNN_LSTM(model_params).to(DEVICE)
    
    weights = torch.tensor([1.0, 2.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params.get("learning_rate", 1e-3), weight_decay=model_params.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    early_stopping = EarlyStopping(patience=5, path=f'models/{model_name}.pt')
    print(f"Model {model_name} training started", flush=True)
    losses = train_model(model, criterion, optimizer, scheduler, early_stopping, train_loader, val_loader)
    # print("\n--- Final Evaluation (Best Model) ---")
    model.load_state_dict(torch.load(f'models/{model_name}.pt'))
    m = evaluate_with_return(model, test_loader)
    results.update(m)
    results.update(losses)
    return results

def run_config(p, signals, labels, X_test, y_test, epochs, num_conf):
    print(f"Configuration {num_conf} started", flush=True)


    metrics = []
    
    print(f"Configuration {num_conf} started", flush=True)

    X_train, X_val, y_train, y_val = train_test_split(signals, labels, test_size=0.2, stratify=labels, random_state=42)

    X_train, y_train = create_dataset(X_train, y_train, 0.75)
    # No overlap for validation and test data
    X_val, y_val = create_dataset(X_val, y_val, 0.0)
    X_test, y_test = create_dataset(X_test, y_test, 0.0)
    
    train_set = EEGDataset(X_train, y_train, is_train=True)
    val_set = EEGDataset(X_val, y_val)
    test_set = EEGDataset(X_test, y_test)

    # test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)
    # train_loader = DataLoader(EEGDataset(X_train, y_train, is_train=True), batch_size=BATCH_SIZE, num_workers=0, shuffle=True) # Shuffle training data each epoch
    # val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=BATCH_SIZE, num_workers=0)
    
    # model = CNN_LSTM(p).to(DEVICE)
    
    # weights = torch.tensor([1.0, 2.0]).to(DEVICE)
    # criterion = nn.CrossEntropyLoss(weight=weights).to(DEVICE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=p.get("learning_rate", 1e-3), weight_decay=p.get("weight_decay", 1e-4))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    # early_stopping = EarlyStopping(patience=5, path=f'models/best_conf{num_conf}.pt')
    # print(f"Configuration {num_conf} training started", flush=True)
    # losses = train_model(model, criterion, optimizer, scheduler, early_stopping, train_loader, val_loader)
    # # print("\n--- Final Evaluation (Best Model) ---")
    # model.load_state_dict(torch.load(f'models/best_conf{num_conf}.pt'))
    # m = evaluate_with_return(model, test_loader)

    # p.update(m)
    # p.update({
    #     'method': 'grid-search' if method is None else method,
    #     "config_num": num_conf
    # })
    # p.update(losses)
    results = init_train_eval_model(p, train_set, val_set, test_set, f"conf{num_conf}", 50)
    results.update({
        'method': 'grid-search',
        "config_num": num_conf
    })
    #print(p)
    print(f"Configuration {num_conf} finished", flush=True)
    return results

def write_results_to_csv(results, filename):
    if not os.path.exists(filename + ".csv"):
        df = pd.DataFrame(results)
        df.to_csv(filename + ".csv", index=False)
    else:
        file_df = pd.read_csv(filename + ".csv")
        df = pd.DataFrame(results)
        combined = pd.concat([file_df, df], axis=0, ignore_index=True)
        combined.to_csv(filename + ".csv", index=False)

def grid_search(signals, labels, parameters:dict, epochs:int=20):
    keys = parameters.keys()
    values = parameters.values()

    parameter_grid = [
        dict(zip(keys, combo))
        for combo in product(*values)
    ]

    # Take 10% for validation, that none of the models ever see
    signals, X_test, labels, y_test = train_test_split(signals, labels, test_size=0.1, random_state=42, stratify=labels)

    results = []
    print("Starting grid search...")

    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=11, mp_context=ctx) as executor:
        futures = [
            executor.submit(run_config, p, signals, labels, X_test, y_test, 50, i)
            for i, p in enumerate(parameter_grid)
        ]
        print(f"Submitted all {len(parameter_grid)} configs", flush=True)
        for f in as_completed(futures):
            results.append(f.result())
            write_results_to_csv([f.result()], "grid_search_partial")

    export(results, "grid_search")
    return results

def export(results, name:str):
    df = pd.DataFrame(results)
    df.to_csv(name + ".csv", index=False)
    df.to_parquet(name + ".parquet")

def kfold_cv(params: list[dict], signals, labels, DataClass = EEGDataset):
    X_train_val, X_test, y_train_val, y_test = train_test_split(signals, labels, test_size=0.1, random_state=32, stratify=labels)
    test_set = DataClass(X_test, y_test)
    kfold = StratifiedKFold(5, shuffle=True)
    final_metrics = []
    i = 0
    j = 0
    for p in params:
        temp_metrics = p
        temp_metrics["method"] = "five-fold CV"
        temp_sum = {}
        for train_idx, val_idx in kfold.split(X_train_val, y_train_val):

            X_train = X_train_val[train_idx]
            y_train = y_train_val[train_idx]
            X_val = X_train_val[val_idx]
            y_val = y_train_val[val_idx]
            
            train_set = DataClass(X_train, y_train, is_train=True)
            val_set = DataClass(X_val, y_val)
            
            res = init_train_eval_model(p, train_set, val_set, test_set, f"kfold_{j}_{i}", 50)
            temp_metrics[f"fold{i}_losses"] = res["losses"]
            temp_metrics[f"fold{i}_accuracy"] = res["accuracy"]
            temp_metrics[f"fold{i}_precision"] = res["precision"]
            temp_metrics[f"fold{i}_recall"] = res["recall"]
            temp_metrics[f"fold{i}_f1"] = res["f1"]
            temp_metrics[f"fold{i}_c11"] = res["c11"]
            temp_metrics[f"fold{i}_c12"] = res["c12"]
            temp_metrics[f"fold{i}_c21"] = res["c21"]
            temp_metrics[f"fold{i}_c22"] = res["c22"]
            for key in res.keys():
                if key == "losses":
                    continue
                if key in temp_sum.keys():
                    temp_sum[key] += res[key]
                else:
                    temp_sum[key] = res[key]
            
            i += 1
        avg_metrics = {f"avg_{key}": temp_sum[key]/float(i) for key in temp_sum.keys()}
        temp_metrics.update(avg_metrics)
        final_metrics.append(temp_metrics)
        i = 0
        j += 1
    return final_metrics

def hold_out_validation(signals, labels, params = DEFAULT_PARAMS, DataClass = EEGDataset):
    print("Train/Test split...")
    X_train, X_test_val, y_train, y_test_val = train_test_split(
        signals, labels, test_size=0.2, random_state=42, stratify=labels
    )
   
    X_test, X_val, y_test, y_val = train_test_split(
        X_test_val, y_test_val, test_size=0.5, random_state=42, stratify=y_test_val
    )

    train_set = DataClass(X_train, y_train, is_train=True)
    val_set = DataClass(X_val, y_val)
    test_set = DataClass(X_test, y_test)

    metrics = init_train_eval_model(params, train_set, val_set, test_set, "Holdout", 50)
    metrics["method"] = "holdout"
# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    print("Loading data...")
    h_signals, h_labels = load_data(DATA_DIR, ["A", "B"], 0)
    s_signals, s_labels = load_data(DATA_DIR, ["E"], 1)
    signals = h_signals + s_signals
    labels = h_labels + s_labels

    print("Preprocessing...")
    signals = preprocess_signals(signals)

    signals = np.array(signals)
    labels = np.array(labels)

    # Create split first, and then segment, to avoid data leakage

    holdout = hold_out_validation(signals, labels)
    export(holdout, "holdout")
    # print("Segmenting...")
    # X_train, y_train = create_dataset(X_train, y_train)
    # X_test, y_test = create_dataset(X_test_val, y_test_val)
    # X_val, y_val = create_dataset(X_val, y_val)

    # train_loader = DataLoader(EEGDataset(X_train, y_train, is_train=True), batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=BATCH_SIZE)
    # test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=BATCH_SIZE)

    # model = CNN_LSTM().to(DEVICE)

    # weights = torch.tensor([1.0, 2.0]).to(DEVICE)
    # criterion = nn.CrossEntropyLoss(weight=weights)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    # early_stopping = EarlyStopping(patience=5, path='best_holdout.pt')

    # # 4. Training Loop
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
        "kernel_size1": [7, 11],
        "kernel_size2": [7, 11],
        "cnn_internal": [64, 128],
        "cnn_out": [128, 256],
        "hidden_size": [128, 256],
        "num_layers": [1, 2, 3],
        "dropout": [0.5, 0.6],
        "learning_rate": [1e-3],
        "weight_decay": [1e-3, 1e-4, 1e-5]
    }
    # grid_search(signals, labels, parameters, 20)

    ## Get the 10 best models (f1 score) and then do 5-fold cross validation on those 10 models


if __name__ == "__main__":
    main()
