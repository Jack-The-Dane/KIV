import pandas as pd
import matplotlib.pyplot as plt

# Load without headers
df = pd.read_csv("GestureClassification/PaperData.csv", header=None)

# Grab the first row (excluding the label at the end if it exists)
first_gesture = df.iloc[0, :-1].values 

plt.figure(figsize=(15, 5))
plt.plot(first_gesture)
plt.title("Visualizing a Single Gesture Row")
plt.xlabel("Data Point Index")
plt.ylabel("Amplitude")
plt.show()