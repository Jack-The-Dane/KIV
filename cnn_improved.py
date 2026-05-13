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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self):
        super(CNN_LSTM, self).__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        # Fully connected
        self.fc = nn.Linear(64, 2)

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
def evaluate(model, test_loader):
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

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    print("Loading data...")
    signals, labels = load_data(DATA_DIR)

    print("Preprocessing...")
    signals = preprocess_signals(signals)

    print("Segmenting...")
    X, y = create_dataset(signals, labels)

    print("Train/Test split...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
    )

    train_dataset = EEGDataset(X_train, y_train, is_train=True)
    test_dataset = EEGDataset(X_test, y_test)

    train_loader = DataLoader(EEGDataset(X_train, y_train, is_train=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=BATCH_SIZE)

    model = CNN_LSTM().to(DEVICE)

    weights = torch.tensor([1.0, 2.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    early_stopping = EarlyStopping(patience=5, path='best_cnn_lstm.pt')

    # 4. Training Loop
    print("--- Starting Training ---")
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Update Scheduler and check Early Stopping
        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Training halted.")
            break

    # 5. Final Evaluation on Test Set
    print("\n--- Final Evaluation (Best Model) ---")
    model.load_state_dict(torch.load('best_cnn_lstm.pt'))
    evaluate(model, test_loader)

if __name__ == "__main__":
    main()