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
def segment_signal(signal, segment_length):
    segments = []
    for i in range(0, len(signal) - segment_length, segment_length):
        segment = signal[i:i+segment_length]
        segments.append(segment)
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
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Add channel dimension for CNN: (1, seq_len)
        return self.X[idx].unsqueeze(0), self.y[idx]

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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = EEGDataset(X_train, y_train)
    test_dataset = EEGDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = CNN_LSTM().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Training...")
    for epoch in range(EPOCHS):
        loss = train(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")

    print("Evaluating...")
    evaluate(model, test_loader)

if __name__ == "__main__":
    main()