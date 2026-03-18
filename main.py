import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter

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
setA: list[list[float]] = load_txt_from_zip("dataset/Z.zip")
setB: list[list[float]] = load_txt_from_zip("dataset/O.zip")
setC: list[list[float]] = load_txt_from_zip("dataset/N.zip")
setD: list[list[float]] = load_txt_from_zip("dataset/F.zip")
setE: list[list[float]] = load_txt_from_zip("dataset/S.zip")
"""
Preprocessing
"""

def bandpass_filter(signal, low=0.5, high=40, fs=173.61, order=5):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)


def normalize(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


def segment_signal(signal, window_size=256, stride=128):
    segments = []
    for i in range(0, len(signal) - window_size, stride):
        segments.append(signal[i:i+window_size])
    return np.array(segments)

"""
Augmentations
"""

def add_noise(x, noise_level=0.05):
    return x + noise_level * np.random.randn(*x.shape)

def time_shift(x, shift_max=20):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(x, shift)

def amplitude_scale(x, scale_range=(0.8, 1.2)):
    scale = np.random.uniform(*scale_range)
    return x * scale

def time_mask(x, mask_ratio=0.1):
    length = len(x)
    mask_size = int(length * mask_ratio)
    start = np.random.randint(0, length - mask_size)
    x[start:start+mask_size] = 0
    return x

def augment(x):
    x = x.copy()
    if np.random.rand() < 0.5:
        x = add_noise(x)
    if np.random.rand() < 0.5:
        x = time_shift(x)
    if np.random.rand() < 0.5:
        x = amplitude_scale(x)
    if np.random.rand() < 0.5:
        x = time_mask(x)
    return x

"""
Contrastive pairs
"""
class EEGDataset(Dataset):
    def __init__(self, signals):
        self.signals = signals

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = self.signals[idx]

        x1 = augment(x)
        x2 = augment(x)

        x1 = torch.tensor(x1, dtype=torch.float32).unsqueeze(0)
        x2 = torch.tensor(x2, dtype=torch.float32).unsqueeze(0)

        return x1, x2

"""
Encoder
"""
class Encoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze(-1)
        return self.fc(x)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=64, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.projector = ProjectionHead()

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)

"""
Contrastive loss
"""
def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim = sim / temperature
    sim = sim.masked_fill(mask, -1e9)

    positives = torch.cat([
        torch.diag(sim, batch_size),
        torch.diag(sim, -batch_size)
    ])

    labels = torch.zeros(2 * batch_size).long().to(z.device)

    loss = F.cross_entropy(sim, labels)
    return loss

"""
Training loop
"""
def train(model, dataloader, epochs=50, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for x1, x2 in dataloader:
            x1, x2 = x1.cuda(), x2.cuda()

            z1 = model(x1)
            z2 = model(x2)

            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

"""
Extract embeddings
"""
def extract_embeddings(model, signals):
    model.eval()
    embeddings = []

    with torch.no_grad():
        for x in signals:
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
            h = model.encoder(x)
            embeddings.append(h.cpu().numpy())

    return np.vstack(embeddings)

"""
Gaussian Mixture Model
"""
def cluster_embeddings(embeddings, n_components=2):
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    gmm = GaussianMixture(n_components=n_components)
    labels = gmm.fit_predict(embeddings)

    return labels

# Assume raw_signals is a list of EEG signals
setA.extend(setB)
setA.extend(setE)
raw_signals = setA
print(len(raw_signals))
print(len(raw_signals[-1]))

processed = []
for sig in raw_signals:
    sig = bandpass_filter(sig)
    sig = normalize(sig)
    segs = segment_signal(sig)
    processed.extend(segs)

processed = np.array(processed)

dataset = EEGDataset(processed)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = SimCLR().cuda()

train(model, loader, epochs=50)

embeddings = extract_embeddings(model, processed)

labels = cluster_embeddings(embeddings)

print("Cluster labels:", labels)