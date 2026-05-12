# ============================================================
# SUPPORT VECTOR MACHINE (SVM)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    TimeSeriesSplit,
    GridSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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

tscv = TimeSeriesSplit(
    n_splits=5
)

# ============================================================
# PIPELINE
# ============================================================
# Important:
# SVM requires feature scaling.
# ============================================================

pipeline = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "svm",
        SVC(
            probability=True,
            random_state=42
        )
    )
])

# ============================================================
# HYPERPARAMETER GRID
# ============================================================

param_grid = {

    "svm__C": [0.1, 1, 10],

    "svm__kernel": ["linear", "rbf"],

    "svm__gamma": ["scale", "auto"]
}

# ============================================================
# GRID SEARCH
# ============================================================

print("\nStarting hyperparameter tuning...")

grid_search = GridSearchCV(

    estimator=pipeline,

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
print("BEST PARAMETERS")
print("================================================")

print(grid_search.best_params_)

print("\nBest Cross Validation Accuracy:")
print(f"{grid_search.best_score_:.4f}")

# ============================================================
# TEST PREDICTIONS
# ============================================================

y_pred = best_model.predict(X_test)

y_prob = best_model.predict_proba(X_test)[:, 1]

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n================================================")
print("TEST METRICS")
print("================================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

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
    "SVM Confusion Matrix"
)

cm_path = (
    VISUALISATION_DIR /
    "svm_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    bbox_inches="tight"
)

plt.close()

print(f"\nConfusion matrix saved to:\n{cm_path}")

print("\nSVM training completed successfully.")