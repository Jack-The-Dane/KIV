import pandas as pd

PARAMETERS = ["kernel_size1",
            "kernel_size2",
            "cnn_internal",
            "cnn_out",
            "hidden_size",
            "num_layers",
            "dropout",
            "learning_rate",
            "weight_decay"]

METRICS = ["f1", "precision", "recall", "accuracy"]
PM = []
PM.extend(PARAMETERS)
PM.extend(METRICS)
print(PM)

df1 = pd.read_parquet("results/grid_search1.parquet")
df2 = pd.read_parquet("results/grid_search2.parquet")
df1["run"] = 1
df2["run"] = 2
df = pd.concat([df1, df2], axis=0, ignore_index=True)
print(df.columns)
sorted_df = df.sort_values("f1", ascending=False, axis=0)
print(sorted_df[PM].head(10))

kfold = pd.read_parquet("kfold-cv.parquet")
cols = [c for c in kfold.columns if "avg" in c]
print(kfold[cols].sort_values("avg_f1", ascending=False))
kfold[PARAMETERS].to_csv("top10_models.csv", index=False)
kfold.sort_values("avg_f1", ascending=False)[PARAMETERS].head(1).to_csv("top_model.csv", index=False)

rps_k = pd.read_parquet("kfold_rps.parquet")
cols = [c for c in rps_k.columns if "avg" in c]
print("KFold RPS: ")
print(rps_k[cols].sort_values("avg_f1", ascending=False))
