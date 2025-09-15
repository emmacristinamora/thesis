import pandas as pd
from pathlib import Path

# === PATHS === #
REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_DIR / "data"
INPUT_FILE = DATA_DIR / "debates_df_themes.csv"

ANNOTATION_DIR = DATA_DIR / "rhetoric"
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD DEBATES DATASET === #
df = pd.read_csv(INPUT_FILE)
print(f"Loaded debates dataset with {len(df)} rows.")

# === FILTER: DROP MODERATORS === #
df_candidates = df[df["speaker_normalized"].str.contains("Candidate", na=False)].copy()
print(f"Candidate-only utterances: {len(df_candidates)}")

# === KEEP ONLY USEFUL COLUMNS === #
df_candidates = df_candidates[[
    "debate_id", "utterance_id", "year", "decade", "party", "theme_name", "text"
]]

# === BALANCED SAMPLING === #
N_TOTAL = 300
groups = df_candidates.groupby(["party", "decade"])
N_PER_GROUP = max(1, N_TOTAL // len(groups))

samples = []
for (party, decade), group in groups:
    n = min(N_PER_GROUP, len(group))
    if n > 0:
        samples.append(group.sample(n=n, random_state=42))

df_sample = pd.concat(samples).sample(frac=1, random_state=42).reset_index(drop=True)

# add empty label column
df_sample["label"] = ""  # to be filled with attack / acclaim / defense

# === SAVE ANNOTATION DATASET === #
OUT_FILE = ANNOTATION_DIR / "debates_rhetoric_annotation.csv"
df_sample.to_csv(OUT_FILE, index=False)

print(f"Saved balanced annotation dataset: {OUT_FILE} ({len(df_sample)} rows)")
print(df_sample.head(10))