import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

# === PATHS === #
REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_DIR / "data"
CLEANED_DIR = DATA_DIR / "processed_clean"
METADATA_FILE = DATA_DIR / "debates_metadata.csv"
OUTPUT_FILE = DATA_DIR / "initial_debates_dataset.csv"

# === LOAD METADATA === #
metadata = pd.read_csv(METADATA_FILE)
metadata.columns = metadata.columns.str.strip()
metadata["filename"] = metadata["filename"].str.replace(".txt", "", regex=False)

# === SPEAKER & PARTY INFERENCE === #
def get_speaker_and_party(norm_speaker, meta_row):
    role_map = {
        "Candidate_R": ("candidate_R", "Republican"),
        "Candidate_D": ("candidate_D", "Democrat"),
        "Candidate_I": ("candidate_I", "Independent"),
        "Moderator":   (None, np.nan)
    }

    role = str(norm_speaker).strip()
    if role in role_map:
        meta_key, party = role_map[role]
        if meta_key:
            name = str(meta_row.get(meta_key, "")).strip()
            return name, party
        else:
            return "Moderator", np.nan
    else:
        return "", ""

# === WINNER PARTY INFERENCE === #
def get_winner_party(meta_row):
    winner_name = str(meta_row.get("winner", "")).strip().lower()

    for role_key, party in {
        "candidate_R": "Republican",
        "candidate_D": "Democrat",
        "candidate_I": "Independent"
    }.items():
        candidate_name = str(meta_row.get(role_key, "")).strip().lower()
        if candidate_name and winner_name in candidate_name:
            return party

    return ""

# === COMBINE ALL CLEANED FILES === #
print("🚀 Creating final combined dataset...")

all_rows = []
cleaned_files = sorted(CLEANED_DIR.glob("*_utterances.csv"))

for file in tqdm(cleaned_files, desc="Merging files", ncols=100):
    debate_id = file.stem.replace("_utterances", "")
    meta_row = metadata[metadata["filename"] == debate_id]

    if meta_row.empty:
        print(f"⚠️  Skipping {debate_id} (no metadata match)")
        continue

    row = meta_row.iloc[0]
    year = row["year"]
    debate_type = row["debate_type"]
    winner = row["winner"]
    winner_party = get_winner_party(row)

    df = pd.read_csv(file)
    df["speaker"] = ""
    df["party"] = ""

    for idx, norm_speaker in enumerate(df["speaker_normalized"]):
        speaker_name, party = get_speaker_and_party(norm_speaker, row)
        df.at[idx, "speaker"] = speaker_name
        df.at[idx, "party"] = party

    df["winner"] = winner
    df["winner_party"] = winner_party
    df["year"] = year
    df["debate_type"] = debate_type
    df["debate_id"] = debate_id

    all_rows.append(df)

# === CONCAT & SAVE === #
final_df = pd.concat(all_rows, ignore_index=True)
final_df = final_df[[
    "text", "speaker_normalized", "speaker", "party",
    "winner", "winner_party", "year", "debate_type", "debate_id"
]]

final_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Final dataset saved to: {OUTPUT_FILE}")
print(f"🧾 Total utterances: {len(final_df)}")