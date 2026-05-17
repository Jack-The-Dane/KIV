import pandas as pd

PARAMETERS = ["kernel_size1",
            "kernel_size2",
            "cnn_internal",
            "cnn_out",
            "hidden_size",
            "num_layers",
            "dropout",
            "learning_rate",
            "weight_decay",
            "run"]

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

kfold = pd.read_csv("kfold-cv.parquet")
print(kfold)