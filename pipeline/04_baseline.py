# ============================================================
# BASELINE MODELS EVALUATION
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "data" 
VISUALISATION_DIR = ROOT_DIR / "data" / "visualisation"

#VISUALISATION_DIR.mkdir(
#    parents=True,
#    exist_ok=True
#)

TRAIN_FILE = DATASET_DIR / "train.csv"
TEST_FILE = DATASET_DIR / "test.csv"

# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ============================================================
# USE TEST SET
# ============================================================

df = test_df.copy()

# ============================================================
# BASELINE 1 - RANK
# ============================================================
# rank_diff = p1_rank - p2_rank
#
# Lower rank is better.
# If rank_diff < 0:
#     Player 1 has better rank
#     predict Player 1 wins
# ============================================================

print("\n================================================")
print("BASELINE 1 - RANK BASELINE")
print("================================================")

df["prediction_rank"] = (
    df["rank_diff"] < 0
).astype(int)

accuracy = accuracy_score(
    df["target"],
    df["prediction_rank"]
)

print(f"Accuracy: {accuracy:.4f}")

report = classification_report(
    df["target"],
    df["prediction_rank"]
)

print(report)

cm = confusion_matrix(
    df["target"],
    df["prediction_rank"]
)

cm_display = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

cm_display.plot()

plt.title("Baseline 1 - Rank Prediction")

rank_plot_path = (
    VISUALISATION_DIR /
    "baseline_rank_confusion_matrix.png"
)

plt.savefig(
    rank_plot_path,
    bbox_inches="tight"
)

plt.close()

print(f"Rank confusion matrix saved to:\n{rank_plot_path}")

# ============================================================
# BASELINE 2 - ELO
# ============================================================
# elo_diff = p1_elo - p2_elo
#
# If elo_diff > 0:
#     Player 1 has higher Elo
# ============================================================

print("\n================================================")
print("BASELINE 2 - ELO BASELINE")
print("================================================")

df["prediction_elo"] = (
    df["elo_diff"] > 0
).astype(int)

accuracy = accuracy_score(
    df["target"],
    df["prediction_elo"]
)

print(f"Accuracy: {accuracy:.4f}")

report = classification_report(
    df["target"],
    df["prediction_elo"]
)

print(report)

cm = confusion_matrix(
    df["target"],
    df["prediction_elo"]
)

cm_display = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

cm_display.plot()

plt.title("Baseline 2 - Elo Prediction")

elo_plot_path = (
    VISUALISATION_DIR /
    "baseline_elo_confusion_matrix.png"
)

plt.savefig(
    elo_plot_path,
    bbox_inches="tight"
)

plt.close()

print(f"Elo confusion matrix saved to:\n{elo_plot_path}")

print("\nBaseline evaluation completed successfully.")