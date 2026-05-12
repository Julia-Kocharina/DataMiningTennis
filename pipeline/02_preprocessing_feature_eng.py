# ============================================================
# TENNIS MATCH PREDICTION - PREPROCESSING & FEATURE ENGINEERING
# ============================================================

import pandas as pd
import numpy as np
import random
from pathlib import Path
from collections import defaultdict

pd.set_option("display.max_columns", None)

# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

INPUT_FILE = DATA_DIR / "raw_2000_2026.csv"
OUTPUT_FILE = DATA_DIR / "cleaned_2000_2026.csv"

# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"Initial dataset shape: {df.shape}")

# ============================================================
# REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()

# ============================================================
# CONVERT DATES
# ============================================================

print("Converting dates...")

df["tourney_date"] = pd.to_datetime(
    df["tourney_date"],
    format="%Y%m%d",
    errors="coerce"
)

df = df.dropna(subset=["tourney_date"])

df = df.sort_values(
    "tourney_date"
).reset_index(drop=True)

# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

numeric_cols = [
    "draw_size",
    "match_num",
    "winner_ht",
    "winner_age",
    "winner_rank",
    "winner_rank_points",
    "loser_ht",
    "loser_age",
    "loser_rank",
    "loser_rank_points",
    "best_of"
]

for col in numeric_cols:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

# ============================================================
# REMOVE DATA LEAKAGE FEATURES
# ============================================================

print("Removing leakage features...")

leakage_cols = [

    "score",
    "minutes",

    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",

    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced"
]

existing_leakage_cols = [
    col for col in leakage_cols
    if col in df.columns
]

df = df.drop(columns=existing_leakage_cols)

# ============================================================
# REMOVE LOW QUALITY COLUMNS
# ============================================================

drop_cols = [
    "winner_seed",
    "winner_entry",
    "loser_seed",
    "loser_entry"
]

existing_drop_cols = [
    col for col in drop_cols
    if col in df.columns
]

df = df.drop(columns=existing_drop_cols)

# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print("Handling missing values...")

num_cols = df.select_dtypes(
    include=["float64", "int64"]
).columns

for col in num_cols:

    median_value = df[col].median()

    df[col] = df[col].fillna(
        median_value
    )

valid_surfaces = [
    "Hard",
    "Clay",
    "Grass",
    "Carpet"
]

df = df[
    df["surface"].isin(valid_surfaces)
]

cat_cols = df.select_dtypes(
    include=["object"]
).columns

for col in cat_cols:

    df[col] = df[col].fillna(
        "Unknown"
    )

# ============================================================
# ELO FEATURES
# ============================================================

print("Calculating Elo ratings...")

INITIAL_ELO = 1500
K_FACTOR = 32

player_elo = defaultdict(
    lambda: INITIAL_ELO
)

surface_elo = {
    "Hard": defaultdict(lambda: INITIAL_ELO),
    "Clay": defaultdict(lambda: INITIAL_ELO),
    "Grass": defaultdict(lambda: INITIAL_ELO),
    "Carpet": defaultdict(lambda: INITIAL_ELO)
}

winner_elo_before = []
loser_elo_before = []

winner_surface_elo_before = []
loser_surface_elo_before = []

# ============================================================
# ELO FUNCTIONS
# ============================================================

def expected_score(rating_a, rating_b):

    return 1 / (
        1 + 10 ** ((rating_b - rating_a) / 400)
    )


def update_elo(rating, expected, actual):

    return rating + K_FACTOR * (
        actual - expected
    )

# ============================================================
# CALCULATE ELO
# ============================================================

for _, row in df.iterrows():

    winner = row["winner_id"]
    loser = row["loser_id"]

    surface = row["surface"]

    w_elo = player_elo[winner]
    l_elo = player_elo[loser]

    winner_elo_before.append(w_elo)
    loser_elo_before.append(l_elo)

    w_expected = expected_score(
        w_elo,
        l_elo
    )

    l_expected = expected_score(
        l_elo,
        w_elo
    )

    player_elo[winner] = update_elo(
        w_elo,
        w_expected,
        1
    )

    player_elo[loser] = update_elo(
        l_elo,
        l_expected,
        0
    )

    w_surface_elo = surface_elo[surface][winner]
    l_surface_elo = surface_elo[surface][loser]

    winner_surface_elo_before.append(
        w_surface_elo
    )

    loser_surface_elo_before.append(
        l_surface_elo
    )

    w_surface_expected = expected_score(
        w_surface_elo,
        l_surface_elo
    )

    l_surface_expected = expected_score(
        l_surface_elo,
        w_surface_elo
    )

    surface_elo[surface][winner] = update_elo(
        w_surface_elo,
        w_surface_expected,
        1
    )

    surface_elo[surface][loser] = update_elo(
        l_surface_elo,
        l_surface_expected,
        0
    )

df["winner_elo"] = winner_elo_before
df["loser_elo"] = loser_elo_before

df["winner_surface_elo"] = (
    winner_surface_elo_before
)

df["loser_surface_elo"] = (
    loser_surface_elo_before
)

# ============================================================
# RECENT FORM FEATURES
# ============================================================

print("Calculating recent form...")

player_recent_matches = defaultdict(list)

winner_recent_form = []
loser_recent_form = []

for _, row in df.iterrows():

    winner = row["winner_id"]
    loser = row["loser_id"]

    w_history = player_recent_matches[winner]

    if len(w_history) == 0:
        w_form = 0.5
    else:
        w_form = np.mean(
            w_history[-5:]
        )

    winner_recent_form.append(w_form)

    l_history = player_recent_matches[loser]

    if len(l_history) == 0:
        l_form = 0.5
    else:
        l_form = np.mean(
            l_history[-5:]
        )

    loser_recent_form.append(l_form)

    player_recent_matches[winner].append(1)
    player_recent_matches[loser].append(0)

df["winner_recent_form"] = (
    winner_recent_form
)

df["loser_recent_form"] = (
    loser_recent_form
)

# ============================================================
# HEAD TO HEAD FEATURES
# ============================================================

print("Calculating head-to-head features...")

h2h = defaultdict(int)

winner_h2h_before = []
loser_h2h_before = []

for _, row in df.iterrows():

    winner = row["winner_id"]
    loser = row["loser_id"]

    winner_key = (winner, loser)
    loser_key = (loser, winner)

    winner_h2h_before.append(
        h2h[winner_key]
    )

    loser_h2h_before.append(
        h2h[loser_key]
    )

    h2h[winner_key] += 1

df["winner_h2h"] = winner_h2h_before
df["loser_h2h"] = loser_h2h_before

# ============================================================
# CREATE PLAYER1 VS PLAYER2 DATASET
# ============================================================

print("Creating symmetric dataset...")

processed_rows = []

for _, row in df.iterrows():

    if random.random() < 0.5:

        target = 1

        p1_rank = row["winner_rank"]
        p2_rank = row["loser_rank"]

        p1_age = row["winner_age"]
        p2_age = row["loser_age"]

        p1_ht = row["winner_ht"]
        p2_ht = row["loser_ht"]

        p1_elo = row["winner_elo"]
        p2_elo = row["loser_elo"]

        p1_surface_elo = row["winner_surface_elo"]
        p2_surface_elo = row["loser_surface_elo"]

        p1_form = row["winner_recent_form"]
        p2_form = row["loser_recent_form"]

        p1_h2h = row["winner_h2h"]
        p2_h2h = row["loser_h2h"]

    else:

        target = 0

        p1_rank = row["loser_rank"]
        p2_rank = row["winner_rank"]

        p1_age = row["loser_age"]
        p2_age = row["winner_age"]

        p1_ht = row["loser_ht"]
        p2_ht = row["winner_ht"]

        p1_elo = row["loser_elo"]
        p2_elo = row["winner_elo"]

        p1_surface_elo = row["loser_surface_elo"]
        p2_surface_elo = row["winner_surface_elo"]

        p1_form = row["loser_recent_form"]
        p2_form = row["winner_recent_form"]

        p1_h2h = row["loser_h2h"]
        p2_h2h = row["winner_h2h"]

    processed_rows.append({

        "surface": row["surface"],
        "tourney_level": row["tourney_level"],
        "indoor": row["indoor"],
        "round": row["round"],
        "best_of": row["best_of"],
        "tourney_date": row["tourney_date"],

        "rank_diff": p1_rank - p2_rank,
        "age_diff": p1_age - p2_age,
        "height_diff": p1_ht - p2_ht,
        "elo_diff": p1_elo - p2_elo,
        "surface_elo_diff": p1_surface_elo - p2_surface_elo,
        "recent_form_diff": p1_form - p2_form,
        "h2h_diff": p1_h2h - p2_h2h,

        "target": target
    })

# ============================================================
# FINAL DATAFRAME
# ============================================================

model_df = pd.DataFrame(
    processed_rows
)

# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

categorical_cols = [
    "surface",
    "tourney_level",
    "indoor",
    "round"
]

model_df = pd.get_dummies(
    model_df,
    columns=categorical_cols,
    drop_first=True
)

# ============================================================
# SAVE CLEANED DATASET
# ============================================================

print("Saving cleaned dataset...")

model_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Cleaned dataset saved to:\n{OUTPUT_FILE}")
print(f"Final dataset shape: {model_df.shape}")