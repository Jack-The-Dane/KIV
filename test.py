import pandas as pd

data = pd.read_parquet("grid_search.parquet")
print(data["fold2_val_loss"][0][0])