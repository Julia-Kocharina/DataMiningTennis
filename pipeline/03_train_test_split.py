# ============================================================
# TRAIN TEST SPLIT

# Train on: 2000–2024 historical tennis
# Predict:  the unseen 2025 season
# 2026 excluded due to incomplete data (matches, series) and to avoid future leakage

# ============================================================

import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

INPUT_FILE = DATA_DIR / "cleaned_2000_2026.csv"

TRAIN_OUTPUT_FILE = DATA_DIR / "train.csv"
TEST_OUTPUT_FILE = DATA_DIR / "test.csv"

# ============================================================
# LOAD CLEANED DATASET
# ============================================================

print("Loading cleaned dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"Dataset shape: {df.shape}")

# ============================================================
# CONVERT DATE COLUMN
# ============================================================

print("Converting dates...")

df["tourney_date"] = pd.to_datetime(
    df["tourney_date"]
)

# ============================================================
# TIME BASED SPLIT
# ============================================================
# Train:
#   2000 -> 2024
#
# Test:
#   2025 season only
#
# Why:
# - avoids future leakage
# - avoids incomplete 2026 season
# - realistic sports prediction setup
# ============================================================

print("Creating time-based split...")

train_df = df[
    df["tourney_date"] < "2025-01-01"
]

test_df = df[
    (df["tourney_date"] >= "2025-01-01") &
    (df["tourney_date"] < "2026-01-01")
]

# ============================================================
# REMOVE DATE COLUMN
# ============================================================
# Date was only needed for chronological splitting.
# Remove it before training.
# ============================================================

train_df = train_df.drop(
    columns=["tourney_date"]
)

test_df = test_df.drop(
    columns=["tourney_date"]
)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ============================================================
# SAVE DATASETS
# ============================================================

print("Saving datasets...")

train_df.to_csv(
    TRAIN_OUTPUT_FILE,
    index=False
)

test_df.to_csv(
    TEST_OUTPUT_FILE,
    index=False
)

print("\nTrain dataset saved to:")
print(TRAIN_OUTPUT_FILE)

print("\nTest dataset saved to:")
print(TEST_OUTPUT_FILE)

print(f"\nFinal train shape: {train_df.shape}")
print(f"Final test shape: {test_df.shape}")

print("\nTrain/test split completed successfully.")