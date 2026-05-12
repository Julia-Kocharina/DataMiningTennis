# ============================================================
# MODEL TRAINING
# HYPERPARAMETER TUNING + CROSS VALIDATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import (
    TimeSeriesSplit,
    GridSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.ensemble import RandomForestClassifier

# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "data" 
VISUALISATION_DIR = ROOT_DIR / "data" / "visualisation"

VISUALISATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

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
# FEATURES / TARGET
# ============================================================

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]

X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

# ============================================================
# TIME SERIES CROSS VALIDATION
# ============================================================
# Important:
# We use TimeSeriesSplit instead of random CV
# because tennis data is chronological.
# ============================================================

tscv = TimeSeriesSplit(n_splits=5)

# ============================================================
# MODEL
# ============================================================

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

# ============================================================
# HYPERPARAMETER GRID
# ============================================================

param_grid = {

    "n_estimators": [100, 200],

    "max_depth": [5, 10, 20],

    "min_samples_split": [2, 5],

    "min_samples_leaf": [1, 2],

    "max_features": ["sqrt"]
}

# ============================================================
# GRID SEARCH
# ============================================================

print("\nStarting hyperparameter tuning...")

grid_search = GridSearchCV(

    estimator=rf,

    param_grid=param_grid,

    cv=tscv,

    scoring="accuracy",

    verbose=2,

    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)

# ============================================================
# BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_

print("\n================================================")
print("BEST HYPERPARAMETERS")
print("================================================")

print(grid_search.best_params_)

print("\nBest Cross Validation Accuracy:")
print(f"{grid_search.best_score_:.4f}")

# ============================================================
# TEST EVALUATION
# ============================================================

print("\n================================================")
print("TEST SET EVALUATION")
print("================================================")

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Test Accuracy: {accuracy:.4f}")

report = classification_report(
    y_test,
    y_pred
)

print("\nClassification Report:\n")
print(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

cm_display = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

cm_display.plot()

plt.title(
    "Random Forest - Confusion Matrix"
)

plot_path = (
    VISUALISATION_DIR /
    "random_forest_confusion_matrix.png"
)

plt.savefig(
    plot_path,
    bbox_inches="tight"
)

plt.close()

print(f"\nConfusion matrix saved to:\n{plot_path}")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({

    "feature": X_train.columns,

    "importance": best_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\n================================================")
print("TOP 15 MOST IMPORTANT FEATURES")
print("================================================")

print(feature_importance.head(15))

# ============================================================
# PLOT FEATURE IMPORTANCE
# ============================================================

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.gca().invert_yaxis()

plt.title(
    "Top 15 Feature Importances"
)

plt.xlabel("Importance")

importance_plot_path = (
    VISUALISATION_DIR /
    "random_forest_feature_importance.png"
)

plt.savefig(
    importance_plot_path,
    bbox_inches="tight"
)

plt.close()

print(f"\nFeature importance plot saved to:\n{importance_plot_path}")

print("\nModel training completed successfully.")