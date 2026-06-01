import zipfile
import numpy as np
import csv
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler


def lowpass_filter(signal, cutoff=40, fs=173.61, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, signal)


def apply_filter(X):
    return np.array([lowpass_filter(signal) for signal in X])


def load_txt_from_zip(zip_path):
    arrays = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        txt_files = sorted([f for f in zf.namelist() if f.lower().endswith(".txt")])
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
    X0 = [x.flatten() for s in class0_sets for x in s]
    X1 = [x.flatten() for s in class1_sets for x in s]

    X = np.array(X0 + X1)
    y = np.array([0] * len(X0) + [1] * len(X1))

    idx = np.arange(len(X))
    np.random.shuffle(idx)

    return X[idx], y[idx]


def extract_features(X):
    features = []
    for signal in X:
        mean = np.mean(signal)
        std = np.std(signal)
        maximum = np.max(signal)
        minimum = np.min(signal)
        energy = np.sum(signal**2)
        features.append([mean, std, maximum, minimum, energy])
    return np.array(features)


# =========================
# Load datasets
# =========================
setA = load_txt_from_zip("dataset/A.zip")
setB = load_txt_from_zip("dataset/B.zip")
setC = load_txt_from_zip("dataset/C.zip")
setD = load_txt_from_zip("dataset/D.zip")
setE = load_txt_from_zip("dataset/E.zip")

# =========================
# Prepare dataset (AB vs E)
# =========================
X, y = prepare_binary_dataset([setA, setB], [setE])

# =========================
# Preprocessing
# =========================
X_filtered = apply_filter(X)
X_features = extract_features(X_filtered)

print("Feature shape:", X_features.shape)

# =========================
# Train model ONCE
# =========================
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_all = scaler.transform(X_features)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =========================
# ⏱️ Measure inference time
# =========================

# ---- Single run (test set)
start = time.perf_counter()
y_pred_test = model.predict(X_test)
end = time.perf_counter()

print(f"Inference time (test set): {end - start:.6f} seconds")

# ---- Single run (full dataset)
start = time.perf_counter()
y_pred_all = model.predict(X_all)
end = time.perf_counter()

print(f"Inference time (full dataset): {end - start:.6f} seconds")

# ---- Multiple runs for stable average
runs = 100
times = []

for _ in range(runs):
    start = time.perf_counter()
    model.predict(X_all)
    end = time.perf_counter()
    times.append(end - start)

print("\nAverage inference time over 100 runs:")
print(f"Mean: {np.mean(times):.6f} seconds")
print(f"Std:  {np.std(times):.6f} seconds")

# =========================
# Save timing results
# =========================
# with open("inference_times.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["run_time_seconds"])
#     for t in times:
#         writer.writerow([t])
