import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# PARAMETERS = ["kernel_size1",
#             "kernel_size2",
#             "cnn_internal",
#             "cnn_out",
#             "hidden_size",
#             "num_layers",
#             "dropout",
#             "learning_rate",
#             "weight_decay"]

# METRICS = ["f1", "precision", "recall", "accuracy"]
# PM = []
# PM.extend(PARAMETERS)
# PM.extend(METRICS)
# print(PM)

# df1 = pd.read_parquet("results/grid_search1.parquet")
# df2 = pd.read_parquet("results/grid_search2.parquet")
# df1["run"] = 1
# df2["run"] = 2
# df = pd.concat([df1, df2], axis=0, ignore_index=True)
# print(df.columns)
# sorted_df = df.sort_values("f1", ascending=False, axis=0)
# print(sorted_df[PM].head(10))

# kfold = pd.read_parquet("kfold-cv.parquet")
# cols = [c for c in kfold.columns if "avg" in c]
# print(kfold[cols].sort_values("avg_f1", ascending=False))
# kfold[PARAMETERS].to_csv("top10_models.csv", index=False)
# kfold.sort_values("avg_f1", ascending=False)[PARAMETERS].head(1).to_csv("top_model.csv", index=False)

# rps_k = pd.read_parquet("kfold_rps.parquet")
# cols = [c for c in rps_k.columns if "avg" in c]
# print("KFold RPS: ")
# print(rps_k[cols].sort_values("avg_f1", ascending=False))

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

A = load_data("dataset", ["A"], 0)
B = load_data("dataset", ["B"], 0)
E = load_data("dataset", ["E"], 1)

A1 = A[0][0]
B1 = B[0][0]
E1 = E[0][0]

fig, axs = plt.subplots(3, 1, figsize=(15,5))
axs[0].plot(A1, label="A")
axs[1].plot(B1, label="B")
axs[2].plot(E1, label="E")
axs[2].set_xlabel("Sample")
axs[0].set_ylabel("Microvolt")
axs[1].set_ylabel("Microvolt")
axs[2].set_ylabel("Microvolt")
axs[0].legend(loc="upper right")
axs[1].legend(loc="upper right")
axs[2].legend(loc="upper right")
fig.suptitle("One EEG measurement from subsets A, B and E")
plt.savefig(fname="report/figures/EEG_data_vis.png")
plt.show()