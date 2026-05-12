# ============================================================
#
# EXPLORATORY DATA ANALYSIS FOR TENNIS DATASET
#
# ALL OUTPUTS ARE SAVED INTO:
# Root/Data/Visualisations/
#
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

pd.set_option("display.max_columns", None)

# ============================================================
# 1. DEFINE PROJECT PATHS
# ============================================================

PIPELINE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PIPELINE_DIR.parent

DATA_DIR = ROOT_DIR / "Data"

VIS_DIR = DATA_DIR / "Visualisations"

# Create folder if it does not exist
VIS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "raw_2000_2026.csv"

# ============================================================
# 2. LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"Dataset shape: {df.shape}")

# ============================================================
# 3. BASIC CLEANING
# ============================================================

print("Cleaning dataset...")

# ------------------------------------------------
# Convert date column
# ------------------------------------------------

df["tourney_date"] = pd.to_datetime(
    df["tourney_date"],
    format="%Y%m%d",
    errors="coerce"
)

# ------------------------------------------------
# Numeric columns
# ------------------------------------------------

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
    "best_of",
    "minutes",
    "w_ace",
    "l_ace",
    "w_df",
    "l_df",
    "w_svpt",
    "l_svpt"
]

for col in numeric_cols:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

# ------------------------------------------------
# Keep only valid surfaces
# ------------------------------------------------

valid_surfaces = [
    "Hard",
    "Clay",
    "Grass",
    "Carpet"
]

df = df[
    df["surface"].isin(valid_surfaces)
]

print(f"Dataset shape after cleaning: {df.shape}")

# ============================================================
# 4. SAFE HISTOGRAM FUNCTION
# ============================================================

def safe_histogram(
    data,
    bins,
    title,
    xlabel,
    ylabel,
    output_path
):

    data = pd.to_numeric(
        data,
        errors="coerce"
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    data = np.asarray(
        data,
        dtype=np.float64
    )

    if len(data) == 0:
        print(f"Skipping empty histogram: {title}")
        return

    min_val = np.min(data)
    max_val = np.max(data)

    if min_val == max_val:
        max_val += 1

    bin_edges = np.linspace(
        min_val,
        max_val,
        bins + 1
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        data,
        bins=bin_edges
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.savefig(
        output_path,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")

# ============================================================
# 5. SAVE DATASET INFO
# ============================================================

print("Saving dataset information...")

info_file = VIS_DIR / "dataset_info.txt"

with open(info_file, "w") as f:

    f.write("TENNIS DATASET INFORMATION\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Dataset Shape: {df.shape}\n\n")

    f.write("COLUMN TYPES\n")
    f.write("-" * 40 + "\n")
    f.write(str(df.dtypes))
    f.write("\n\n")

    f.write("DESCRIPTIVE STATISTICS\n")
    f.write("-" * 40 + "\n")
    f.write(str(df.describe(include="all")))
    f.write("\n\n")

print(f"Saved: {info_file}")

# ============================================================
# 6. MISSING VALUES ANALYSIS
# ============================================================

print("Analyzing missing values...")

missing_values = (
    df.isnull()
    .sum()
    .sort_values(ascending=False)
)

missing_percentage = (
    df.isnull()
    .mean() * 100
).sort_values(ascending=False)

missing_df = pd.DataFrame({
    "Missing Count": missing_values,
    "Missing Percentage": missing_percentage
})

missing_file = VIS_DIR / "missing_values.csv"

missing_df.to_csv(missing_file)

print(f"Saved: {missing_file}")

# ============================================================
# 7. SURFACE DISTRIBUTION
# ============================================================

print("Creating surface distribution plot...")

surface_counts = df["surface"].value_counts()

plt.figure(figsize=(8, 8))

plt.pie(
    surface_counts,
    labels=surface_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Surface Type Distribution")

surface_plot = VIS_DIR / "surface_distribution.png"

plt.savefig(
    surface_plot,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {surface_plot}")

# ============================================================
# 8. TOURNAMENT LEVEL DISTRIBUTION
# ============================================================

print("Creating tournament level plot...")

level_mapping = {
    "G": "Grand Slam",
    "M": "Masters 1000",
    "500": "ATP 500",
    "250": "ATP 250",
    "F": "ATP Finals",
    "D": "Davis Cup",
    "A": "ATP Tour",
    "O": "Other"
}

df["tourney_level_full"] = (
    df["tourney_level"]
    .map(level_mapping)
)

level_counts = (
    df["tourney_level_full"]
    .value_counts()
)

plt.figure(figsize=(10, 6))

level_counts.plot(kind="bar")

plt.xlabel("Tournament Level")
plt.ylabel("Count")
plt.title("Tournament Level Distribution")

plt.xticks(rotation=45)

level_plot = VIS_DIR / "tournament_levels.png"

plt.savefig(
    level_plot,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {level_plot}")

# ============================================================
# 9. INDOOR VS OUTDOOR
# ============================================================

print("Creating indoor/outdoor plot...")

indoor_mapping = {
    "I": "Indoor",
    "O": "Outdoor"
}

df["court_type"] = (
    df["indoor"]
    .map(indoor_mapping)
)

court_counts = (
    df["court_type"]
    .value_counts()
)

plt.figure(figsize=(7, 7))

plt.pie(
    court_counts,
    labels=court_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Indoor vs Outdoor Matches")

court_plot = VIS_DIR / "indoor_outdoor.png"

plt.savefig(
    court_plot,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {court_plot}")

# ============================================================
# 10. PLAYER AGE DISTRIBUTION
# ============================================================

print("Creating player age distribution...")

all_ages = pd.concat([
    df["winner_age"],
    df["loser_age"]
])

safe_histogram(
    data=all_ages,
    bins=30,
    title="Player Age Distribution",
    xlabel="Age",
    ylabel="Frequency",
    output_path=VIS_DIR / "player_age_distribution.png"
)

# ============================================================
# 11. PLAYER HEIGHT DISTRIBUTION
# ============================================================

print("Creating player height distribution...")

all_heights = pd.concat([
    df["winner_ht"],
    df["loser_ht"]
])

safe_histogram(
    data=all_heights,
    bins=30,
    title="Player Height Distribution",
    xlabel="Height (cm)",
    ylabel="Frequency",
    output_path=VIS_DIR / "player_height_distribution.png"
)

# ============================================================
# 12. MATCH DURATION DISTRIBUTION
# ============================================================

print("Creating match duration distribution...")

safe_histogram(
    data=df["minutes"],
    bins=40,
    title="Match Duration Distribution",
    xlabel="Minutes",
    ylabel="Frequency",
    output_path=VIS_DIR / "match_duration_distribution.png"
)

# ============================================================
# 13. ATP RANK DISTRIBUTION
# ============================================================

print("Creating ATP rank distribution...")

all_ranks = pd.concat([
    df["winner_rank"],
    df["loser_rank"]
])

safe_histogram(
    data=all_ranks,
    bins=50,
    title="ATP Ranking Distribution",
    xlabel="ATP Rank",
    ylabel="Frequency",
    output_path=VIS_DIR / "atp_rank_distribution.png"
)

# ============================================================
# 14. HANDEDNESS DISTRIBUTION
# ============================================================

print("Creating handedness distribution...")

all_hands = pd.concat([
    df["winner_hand"],
    df["loser_hand"]
])

hand_counts = all_hands.value_counts()

plt.figure(figsize=(7, 5))

hand_counts.plot(kind="bar")

plt.xlabel("Handedness")
plt.ylabel("Count")
plt.title("Player Handedness Distribution")

plt.xticks(rotation=0)

hand_plot = VIS_DIR / "handedness_distribution.png"

plt.savefig(
    hand_plot,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {hand_plot}")

# ============================================================
# 15. RANK DIFFERENCE DISTRIBUTION
# ============================================================

print("Creating rank difference distribution...")

rank_difference = (
    pd.to_numeric(df["loser_rank"], errors="coerce")
    -
    pd.to_numeric(df["winner_rank"], errors="coerce")
)

safe_histogram(
    data=rank_difference,
    bins=50,
    title="Ranking Difference Distribution",
    xlabel="Loser Rank - Winner Rank",
    ylabel="Frequency",
    output_path=VIS_DIR / "rank_difference_distribution.png"
)

# ============================================================
# 16. ACES DISTRIBUTION
# ============================================================

print("Creating aces distribution...")

winner_aces = pd.to_numeric(
    df["w_ace"],
    errors="coerce"
)

winner_aces = winner_aces.replace(
    [np.inf, -np.inf],
    np.nan
).dropna()

winner_aces = np.asarray(
    winner_aces,
    dtype=np.float64
)

loser_aces = pd.to_numeric(
    df["l_ace"],
    errors="coerce"
)

loser_aces = loser_aces.replace(
    [np.inf, -np.inf],
    np.nan
).dropna()

loser_aces = np.asarray(
    loser_aces,
    dtype=np.float64
)

combined = np.concatenate([
    winner_aces,
    loser_aces
])

min_val = np.min(combined)
max_val = np.max(combined)

if min_val == max_val:
    max_val += 1

bins = np.linspace(
    min_val,
    max_val,
    41
)

plt.figure(figsize=(10, 6))

plt.hist(
    winner_aces,
    bins=bins,
    alpha=0.7,
    label="Winners"
)

plt.hist(
    loser_aces,
    bins=bins,
    alpha=0.7,
    label="Losers"
)

plt.xlabel("Number of Aces")
plt.ylabel("Frequency")
plt.title("Aces Distribution")

plt.legend()

aces_plot = VIS_DIR / "aces_distribution.png"

plt.savefig(
    aces_plot,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {aces_plot}")

# ============================================================
# 17. MATCHES PER YEAR
# ============================================================

print("Creating yearly matches plot...")

df["year"] = (
    df["tourney_date"]
    .dt.year
)

matches_per_year = (
    df["year"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(14, 6))

plt.plot(
    matches_per_year.index,
    matches_per_year.values
)

plt.xlabel("Year")
plt.ylabel("Matches")
plt.title("Matches Per Year")

plt.grid(True)

year_plot = VIS_DIR / "matches_per_year.png"

plt.savefig(
    year_plot,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {year_plot}")

# ============================================================
# 18. CORRELATION MATRIX
# ============================================================

print("Creating correlation matrix...")

correlation_features = [
    "winner_age",
    "loser_age",
    "winner_ht",
    "loser_ht",
    "winner_rank",
    "loser_rank",
    "winner_rank_points",
    "loser_rank_points",
    "minutes",
    "w_ace",
    "l_ace"
]

corr_df = df[
    correlation_features
].apply(
    pd.to_numeric,
    errors="coerce"
)

corr_matrix = corr_df.corr()

plt.figure(figsize=(12, 10))

plt.imshow(
    corr_matrix,
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(correlation_features)),
    correlation_features,
    rotation=90
)

plt.yticks(
    range(len(correlation_features)),
    correlation_features
)

plt.title("Feature Correlation Matrix")

corr_plot = VIS_DIR / "correlation_matrix.png"

plt.savefig(
    corr_plot,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {corr_plot}")

# ============================================================
# 19. TOP PLAYERS
# ============================================================

print("Creating top players plot...")

top_players = (
    df["winner_name"]
    .value_counts()
    .head(15)
)

plt.figure(figsize=(12, 7))

top_players.plot(kind="bar")

plt.xlabel("Player")
plt.ylabel("Match Wins")
plt.title("Top 15 Players by Match Wins")

plt.xticks(rotation=45)

top_players_plot = VIS_DIR / "top_players.png"

plt.savefig(
    top_players_plot,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {top_players_plot}")

# ============================================================
# 20. SAVE SUMMARY
# ============================================================

print("Saving statistical summary...")

summary_file = VIS_DIR / "dataset_summary.txt"

with open(summary_file, "w") as f:

    f.write("TENNIS DATASET SUMMARY\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Dataset Shape: {df.shape}\n\n")

    f.write("SURFACE DISTRIBUTION\n")
    f.write("-" * 40 + "\n")
    f.write(str(surface_counts))
    f.write("\n\n")

    f.write("TOURNAMENT LEVEL DISTRIBUTION\n")
    f.write("-" * 40 + "\n")
    f.write(str(level_counts))
    f.write("\n\n")

    f.write("MISSING VALUES\n")
    f.write("-" * 40 + "\n")
    f.write(str(missing_df))
    f.write("\n\n")

    f.write("NUMERICAL STATISTICS\n")
    f.write("-" * 40 + "\n")
    f.write(str(df.describe()))

print(f"Saved: {summary_file}")

# ============================================================
# FINISHED
# ============================================================

print("\nEDA completed successfully.")
print(f"\nAll outputs saved into:\n{VIS_DIR}")