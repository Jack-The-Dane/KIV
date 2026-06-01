from cnn_improved import init_train_eval_model, EEGDataset, PARAMETER_MAMES, load_data, DATA_DIR, preprocess_signals, CNN_LSTM, DEVICE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
import torch
import time

params = pd.read_csv("results/kfold-top5_eeg.csv")
print(params.columns)
params = params.sort_values("avg_f1", axis=0, ascending=False).head(1)[PARAMETER_MAMES]
print(params)
params = params.to_dict("records")[0]
hx, hy = load_data(DATA_DIR, ["A", "B"], 0)
sx, sy = load_data(DATA_DIR, ["E"], 1)
X = hx + sx
y = hy + sy

X = preprocess_signals(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, stratify=y_train, test_size=0.125)

train_set = EEGDataset(X_train, y_train, is_train=True)
val_set = EEGDataset(X_val, y_val)
test_set = EEGDataset(X_test, y_test)

#init_train_eval_model(params, train_set, val_set, test_set, "inference", 50)

model = CNN_LSTM(params).to(DEVICE)
model.load_state_dict(torch.load("inference_models/inference.pt"))

model.eval()
with torch.no_grad():
    X = np.array(X)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    start_one = time.perf_counter()
    model(X_tensor)
    end_one = time.perf_counter()
    print(f"Elapsed time for one dataset: {end_one - start_one}")
    times = []
    for i in range(100):
        start = time.perf_counter()
        model(X_tensor)
        end = time.perf_counter()
        times.append(end - start)

    print("\nAverage inference time over 100 runs:")
    print(f"Mean: {np.mean(times):.6f} seconds")
    print(f"Std:  {np.std(times):.6f} seconds")
    # # 4. Prepare your new input data
    # # Let's say your sequence length is 100.
    # # Model expects: (batch_size, channels, sequence_length) -> (1, 1, 100)
    # raw_data = np.random.randn(100) # Replace with your real 1D data array
    # tensors = [torch.tensor(raw_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE) for raw_data in signals]
    # start_time = time.time()
    # for raw_data in signals:
    #     logits = model(input_tensor)  # Shape: (1, 2)
            
    #         # # If doing classification (since your output size is 2):
    #         # probabilities = torch.softmax(logits, dim=1)
    #         # predicted_class = torch.argmax(probabilities, dim=1).item()
    # end_time = time.time()
    # print(f"Start time: {start_time}")
    # print(f"End time: {end_time}")
    # print(f"Elapsed time: {end_time-start_time}")
