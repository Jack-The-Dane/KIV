print("SCRIPT STARTED")
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
print("import numpy")
import numpy as np
from glob import glob
print("import scipy")
from scipy import interpolate, signal
print("import sklearn")
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
print("import torch")
import torch
print("import torch.nn")
import torch.nn as nn
print("import torch.optim")
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
print("SCRIPT STARTED")

def bandpass_filter(signal, low=0.5, high=40, fs=173.61):
    nyq = 0.5 * fs
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

def load_eeg_folder(folder_path, label):
    data = []
    labels = []
    
    files = sorted(glob(os.path.join(folder_path, "*.txt")))
    
    for f in files:
        signal_data = np.loadtxt(f)
        data.append(signal_data)
        labels.append(label)
    
    return data, labels

def interpolate_signal(signal):
    if np.any(np.isnan(signal)):
        x = np.arange(len(signal))
        mask = ~np.isnan(signal)
        f = interpolate.interp1d(x[mask], signal[mask], kind='linear', fill_value="extrapolate")
        signal = f(x)
    return signal

SEGMENT_SIZE = int(2 * 173.61)  # ≈ 347

def segment_signal(signal):
    segments = []
    for i in range(0, len(signal) - SEGMENT_SIZE, SEGMENT_SIZE):
        seg = signal[i:i + SEGMENT_SIZE]
        segments.append(seg)
    return segments

def normalize_segment(segment):
    return (segment - np.mean(segment)) / (np.std(segment) + 1e-8)

def preprocess(data, labels):
    segments = []
    seg_labels = []
    
    for signal, label in zip(data, labels):
        signal = interpolate_signal(signal)
        signal = bandpass_filter(signal)
        segs = segment_signal(signal)
        
        for seg in segs:
            seg = normalize_segment(seg)
            segments.append(seg)
            seg_labels.append(label)
    
    return np.array(segments), np.array(seg_labels)

class Encoder(nn.Module):
    def __init__(self, input_dim=347, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)
    
def augment(signal):
    # Add noise
    signal = signal + np.random.normal(0, 0.05, signal.shape)

    # Random scaling
    signal = signal * np.random.uniform(0.5, 1.5)

    # Time shift (stronger)
    shift = np.random.randint(0, len(signal)//4)
    signal = np.roll(signal, shift)

    # Random masking (VERY important)
    mask_len = np.random.randint(20, 80)
    start = np.random.randint(0, len(signal) - mask_len)
    signal[start:start+mask_len] = 0

    return signal
# def augment(signal):
#     # jitter
#     noise = np.random.normal(0, 0.05, signal.shape)
#     signal = signal + noise
    
#     # random scaling
#     signal = signal * np.random.uniform(0.8, 1.2)
    
#     return signal

# def contrastive_loss(z1, z2, temperature=0.5):
#     z1 = nn.functional.normalize(z1, dim=1)
#     z2 = nn.functional.normalize(z2, dim=1)
    
#     representations = torch.cat([z1, z2], dim=0)
#     similarity_matrix = torch.matmul(representations, representations.T)
    
#     batch_size = z1.shape[0]
#     labels = torch.arange(batch_size).to(z1.device)
#     labels = torch.cat([labels, labels])
    
#     mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
#     similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
    
#     positives = torch.sum(z1 * z2, dim=1)
#     positives = torch.cat([positives, positives])
    
#     logits = similarity_matrix / temperature
    
#     loss = nn.CrossEntropyLoss()(logits, labels)
#     return loss

def contrastive_loss(z1, z2, temperature=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)

    batch_size = z1.size(0)

    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
    similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

    positives = torch.sum(z1 * z2, dim=1)
    positives = torch.cat([positives, positives], dim=0)

    logits = similarity_matrix / temperature
    labels = torch.zeros(2 * batch_size, dtype=torch.long).to(z1.device)

    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

def train_contrastive(X, epochs=20, batch_size=128):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    model = Encoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    for epoch in range(epochs):
        perm = torch.randperm(len(X_tensor))
        
        for i in range(0, len(X_tensor), batch_size):
            idx = perm[i:i+batch_size]
            batch = X_tensor[idx].to(device)
            
            aug1 = []
            aug2 = []

            for x in batch:
                x_np = x.cpu().numpy().copy()  # ensure safe memory
                aug1.append(augment(x_np))
                aug2.append(augment(x_np))

            aug1 = torch.tensor(np.array(aug1), dtype=torch.float32).to(device)
            aug2 = torch.tensor(np.array(aug2), dtype=torch.float32).to(device)

            # aug1 = torch.tensor([augment(x.numpy()) for x in batch]).float().to(device)
            # aug2 = torch.tensor([augment(x.numpy()) for x in batch]).float().to(device)
            
            z1 = model(aug1)
            z2 = model(aug2)
            
            loss = contrastive_loss(z1, z2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return model

def get_embeddings(model, X):
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        embeddings = model(X_tensor).numpy()
    return embeddings

def train_gmm(embeddings):
    gmm = GaussianMixture(n_components=2, covariance_type='full')
    gmm.fit(embeddings)
    return gmm

def evaluate_gmm(gmm: GaussianMixture, embeddings, true_labels):
    probs = gmm.predict_proba(embeddings)
    preds = np.argmax(probs, axis=1)
    
    # Align clusters with labels
    if accuracy_score(true_labels, preds) < 0.5:
        preds = 1 - preds
    
    acc = accuracy_score(true_labels, preds)
    auc = roc_auc_score(true_labels, probs[:,1])
    
    print("Accuracy:", acc)
    print("AUC:", auc)
    print(classification_report(true_labels, preds))

def pipeline():
    # Load data
    print("Loading")
    A_data, A_labels = load_eeg_folder("dataset/A", 0)
    B_data, B_labels = load_eeg_folder("dataset/B", 0)
    E_data, E_labels = load_eeg_folder("dataset/E", 1)
    print("Loaded all datasets")
    data = A_data + B_data + E_data
    labels = A_labels + B_labels + E_labels

    # Preprocess
    X, y = preprocess(data, labels)
    print("Preprocessed data")
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("Split data")
    # Train contrastive model
    model = train_contrastive(X_train, 50, 128)
    print("Trained contrastive network")
    # Embeddings
    train_emb = get_embeddings(model, X_train)
    test_emb = get_embeddings(model, X_test)

    scaler = StandardScaler()
    train_emb = scaler.fit_transform(train_emb)
    test_emb = scaler.transform(test_emb)
    print("Got embeddings")
    # GMM
    gmm = train_gmm(train_emb)
    print("Trained GMM")
    # Evaluate
    evaluate_gmm(gmm, test_emb, y_test)
    print("Evaluated GMM")

try:
    pipeline()
except Exception as e:
    print(e)
# if __name__ == "__main__":
#     pipeline()