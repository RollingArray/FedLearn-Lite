import numpy as np
import pandas as pd
import os

# Parameters
TOTAL_SAMPLES = 1000000
NUM_CLIENTS = 3
NOISE_STD = 0.5
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# Equation: y = 2x + 1 + noise
x_all = np.sort(np.random.uniform(0, 10, TOTAL_SAMPLES))
noise_all = np.random.normal(0, NOISE_STD, TOTAL_SAMPLES)
y_all = 2 * x_all + 1 + noise_all

# Combine and shuffle the dataset
df_full = pd.DataFrame({"x": x_all, "y": y_all})
df_shuffled = df_full.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into N clients
split_dfs = np.array_split(df_shuffled, NUM_CLIENTS)

for i, df in enumerate(split_dfs, start=1):
    file_path = f"{DATA_DIR}/client-data-{i}.csv"
    df.to_csv(file_path, index=False)
    print(f"✅ Saved {file_path} with {len(df)} samples")

print("🎉 All client datasets split from a common dataset based on y = 2x + 1 + noise")
