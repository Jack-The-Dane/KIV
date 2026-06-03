import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
from cnn_improved import PARAMETER_MAMES
from sklearn.metrics import ConfusionMatrixDisplay

import seaborn as sn

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
# # df1["run"] = 1
# # df2["run"] = 2
# df = pd.concat([df1, df2], axis=0, ignore_index=True)
# print(df.columns)
# for p in PARAMETER_MAMES:
#     print(f"{p}: {df[p].unique()}")
# quit()
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

paper = pd.read_csv("GestureClassification/PaperData.csv", header=None)
rock = pd.read_csv("GestureClassification/RockData.csv", header=None)
print(rock.shape)
print(paper.shape)
#quit()
fig, ax = plt.subplots(2, 1)
ax[0].plot(paper.loc[0,:], label="Paper")
ax[1].plot(rock.loc[0,:], label="Rock")
ax[0].legend(loc="upper right")
ax[1].legend(loc="upper right")
ax[1].set_xlabel("Sample")
ax[0].set_ylabel("Unit unknown")
ax[1].set_ylabel("Unit unknown")
plt.suptitle("Measurements from rock and paper datasets")
#plt.savefig("report/figures/gesture_data.png")
#plt.show()
plt.clf()


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

A1 = A[0][0][:1000]
B1 = B[0][0][:1000]
E1 = E[0][0][:1000]

fig, axs = plt.subplots(3, 1)
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
fig.suptitle("Segment of EEG measurement from subsets A, B and E")
#plt.savefig(fname="report/figures/EEG_data_vis.png")
#plt.show()
plt.cla()
plt.clf()
#quit()
def plot_loss(val_loss, train_loss, filename = None):
    fig, ax = plt.subplots(figsize=(8,6))
    epochs = np.arange(1, len(val_loss)+1, 1)
    min_loss_epoch = float(np.argmin(val_loss)+1)
    min_val_loss = min(val_loss)
    ax.plot(epochs, val_loss, label="Validation loss")
    ax.plot(epochs, train_loss, label="Training loss")
    ax.axhline(min_val_loss, color="r", linestyle="dashed", linewidth=0.5)
    ax.axvline(min_loss_epoch, color="r", linestyle="dashed", linewidth=0.5)
    # --- Add Custom Ticks Here ---
    # Get current ticks, append the new value, and set them back
    #ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # For the X-axis (Epochs)
    current_xticks = list(ax.get_xticks())
    # Optional: remove auto-ticks that are too close to your custom tick to avoid overlapping text
    current_xticks = [x for x in current_xticks if abs(x - min_loss_epoch) > 0.5] 
    ax.set_xticks(current_xticks + [min_loss_epoch])

    # For the Y-axis (Loss)
    current_yticks = list(ax.get_yticks())
    # Optional: remove auto-ticks that are too close to your custom tick to avoid overlapping text
    current_yticks = [y for y in current_yticks if abs(y - min_val_loss) > 0.01]
    ax.set_yticks(current_yticks + [min_val_loss])
    # ------------------------------
    plt.xlim(left=1, right=max(epochs))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and validation loss of the first fold in 5-fold cross validation")
    if filename != None:
        plt.savefig(f"report/figures/{filename}.png")
    plt.show()
    plt.clf()

eeg_top5 = pd.read_parquet("results/kfold-top5_eeg.parquet")
rps_top5 = pd.read_parquet("results/kfold_top5_models_rps.parquet")

avg_cols = ["avg_f1","avg_precision","avg_recall","avg_accuracy","avg_c11","avg_c12","avg_c21","avg_c22"]

print(rps_top5[PARAMETER_MAMES+avg_cols])
eeg_best = eeg_top5.sort_values("avg_f1", axis=0, ascending=False).head(1)
print("Best EEG model:")
print(eeg_best[PARAMETER_MAMES+["avg_f1"]])
print("Best EEG model confusion matrix:")
conf_EEG = np.array([[eeg_best["avg_c11"].to_list()[0], eeg_best["avg_c12"].to_list()[0]],
                     [eeg_best["avg_c21"].to_list()[0], eeg_best["avg_c22"].to_list()[0]]])
print(conf_EEG)

eeg_val_loss1 = eeg_best["fold0_val_loss"].to_list()[0]
eeg_train_loss1 = eeg_best["fold0_train_loss"].to_list()[0]
#print("Validation loss:", eeg_val_loss1)
#plot_loss(eeg_val_loss1, eeg_train_loss1, "eeg_loss")

rps_best = rps_top5.sort_values("avg_f1", axis=0, ascending=False).head(1)
print("Best RPS model:")
print(rps_best[PARAMETER_MAMES+["avg_f1"]])
print("Best rps model confusion matrix:")
conf_rps = np.array([[rps_best["avg_c11"].to_list()[0], rps_best["avg_c12"].to_list()[0]],
                     [rps_best["avg_c21"].to_list()[0], rps_best["avg_c22"].to_list()[0]]])
print(conf_rps)
rps_val_loss = rps_best["fold0_val_loss"].to_list()[0]
rps_train_loss = rps_best["fold0_train_loss"].to_list()[0]
#plot_loss(rps_val_loss, rps_train_loss, "rps_loss")
#fig, ax = plt.subplots(figsize=(5,5))
# ConfusionMatrixDisplay(conf_EEG, display_labels=["Healthy", "Seizure"]).plot()
# plt.title("Confusion matrix of healthy vs seizure classification")
# plt.savefig("report/figures/EEG_confusion.png")
# plt.show()
# plt.clf()


# ConfusionMatrixDisplay(conf_rps, display_labels=["Paper", "Rock"]).plot()
# plt.title("Confusion matrix of paper vs rock gesture detection")
# plt.savefig("report/figures/RPS_confusion.png")
# plt.show()
# plt.clf()

# v = [eeg_best[f"fold{i}_val_loss"] for i in range(5)]
# for t in v:
#     print(t[1].shape)