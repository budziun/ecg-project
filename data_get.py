import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Source of Data
# https://www.kaggle.com/datasets/shayanfazeli/heartbeat

# Load Data
train_df = pd.read_csv('mitbih_train.csv', header=None)
test_df = pd.read_csv('mitbih_test.csv', header=None)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

X_train_full = train_df.iloc[:, :-1].values
y_train_full = train_df.iloc[:, -1].values.astype(int)

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values.astype(int)

# Split train  train/val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.15,
    stratify=y_train_full,
    random_state=42
)

print(f"\n{'='*60}")
print("Dataset Statistics")
print('='*60)
print(f"\nTrain: {len(X_train)} samples")
print(f"Val: {len(X_val)} samples")
print(f"Test: {len(X_test)} samples")

print(f"\nClass distribution:")
print(f"{'Class':<10} {'Train':<10} {'Val':<10} {'Test':<10}")
print('-'*60)
for i in range(5):
    train_count = (y_train == i).sum()
    val_count = (y_val == i).sum()
    test_count = (y_test == i).sum()
    print(f"{i:<10} {train_count:<10} {val_count:<10} {test_count:<10}")

# Reshape for CNN (187 timesteps, 1 channel)
X_train = X_train.reshape(-1, 187, 1)
X_val = X_val.reshape(-1, 187, 1)
X_test = X_test.reshape(-1, 187, 1)

print(f"\nReshaped data:")
print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")

import os
os.makedirs('data', exist_ok=True)

np.save('data/X_train.npy', X_train)
np.save('data/y_train.npy', y_train)
np.save('data/X_val.npy', X_val)
np.save('data/y_val.npy', y_val)
np.save('data/X_test.npy', X_test)
np.save('data/y_test.npy', y_test)

print("\n✓ Data saved to 'data/' directory!")
