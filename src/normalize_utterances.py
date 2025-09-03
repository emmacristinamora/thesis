import os
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# === PATHS === #
REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CLEANED_DIR = DATA_DIR / "processed_clean"
METADATA_FILE = DATA_DIR / "debates_metadata.csv"

# === CREATE OUTPUT DIR === #
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD & CLEAN METADATA === #
metadata = pd.read_csv(METADATA_FILE)
metadata.columns = metadata.columns.str.strip()
metadata["filename"] = metadata["filename"].str.replace(".txt", "", regex=False)
print(f"✅ Loaded metadata with {len(metadata)} rows.")

# === GET ALL UTTERANCE FILES === #
utterance_files = sorted(PROCESSED_DIR.glob("*_utterances.csv"))
print(f"✅ Found {len(utterance_files)} utterance files.")

# === NORMALIZE SPEAKER NAMES === #
def normalize_speaker_name(name: str) -> str:
    """Cleans and standardizes a speaker name string."""
    if pd.isna(name):
        return ""
    name = name.strip()
    name = re.sub(r"\.+$", "", name)     # remove trailing periods
    name = re.sub(r"\s+", " ", name)     # normalize spacing
    name = name.title()                  # uppercase fix
    return name

def clean_for_matching(name: str) -> str:
    """Standardize a name for fuzzy matching (lowercase, remove parentheses etc)."""
    name = str(name).lower().strip()
    name = re.sub(r"\(.*?\)", "", name)       # remove (Jr), (Sr), (Hillary), etc.
    name = re.sub(r"[^a-z\s]", "", name)      # remove punctuation
    name = re.sub(r"\s+", " ", name)          # normalize spacing
    return name

# === AMBIGUITY CACHE FOR "PRESIDENT" ETC. === #
ambiguous_resolution_cache = {}

# === MATCH TO CANONICAL ROLES === #
def match_role(name: str, canonical_map: dict, debate_id: str) -> str:
    """Match speaker name to canonical roles using token overlap."""
    name_clean = clean_for_matching(name)
    name_tokens = set(name_clean.split())

    for full_name, role in canonical_map.items():
        if not full_name:
            continue
        candidate_clean = clean_for_matching(full_name)
        candidate_tokens = set(candidate_clean.split())

        # match if there's any word overlap 
        if name_tokens & candidate_tokens:
            return role

    # generic moderator labels
    if any(k in name_clean for k in ["moderator", "question", "q", "audience"]):
        return "Moderator"

    # ambiguous "President"
    if "president" in name_clean:
        if debate_id in ambiguous_resolution_cache:
            return ambiguous_resolution_cache[debate_id]

        print(f"\n🟡 Ambiguous speaker name: '{name}' in debate {debate_id}")
        while True:
            choice = input("Who is 'President' referring to? (R / D / I / M): ").strip().upper()
            if choice in ["R", "D", "I", "M"]:
                resolved = {
                    "R": "Candidate_R",
                    "D": "Candidate_D",
                    "I": "Candidate_I",
                    "M": "Moderator"
                }[choice]
                ambiguous_resolution_cache[debate_id] = resolved
                return resolved
    # fallback
    return "Moderator"

# === PROCESS ALL FILES === #
def normalize_all_speakers():
    print("🚀 Starting speaker normalization...")

    for file in tqdm(utterance_files, desc="Normalizing speakers", ncols=100):
        try:
            df = pd.read_csv(file)
            debate_id = file.stem.replace("_utterances", "")
            meta_row = metadata[metadata["filename"] == debate_id]

            if meta_row.empty:
                print(f"❌ No metadata for debate_id: {debate_id}, skipping.")
                continue

            row = meta_row.iloc[0]

            # canonical role map 
            canonical_map = {
                str(row.get("candidate_R", "")): "Candidate_R",
                str(row.get("candidate_D", "")): "Candidate_D",
                str(row.get("candidate_I", "")): "Candidate_I",
                str(row.get("moderator", "")): "Moderator"
            }

            # normalize speaker roles
            normalized_roles = []
            for speaker in df["speaker"]:
                norm_name = normalize_speaker_name(speaker)
                role = match_role(norm_name, canonical_map, debate_id)
                normalized_roles.append(role)

            # insert into dataframe
            df.insert(2, "speaker_normalized", normalized_roles)

            # save result
            output_path = CLEANED_DIR / file.name
            df.to_csv(output_path, index=False)
            print(f"✅ Saved: {output_path.name}")

        except Exception as e:
            print(f"❌ Error processing {file.name}: {e}")

    print("🎉 All speakers normalized.")

# === MAIN ENTRY === #
if __name__ == "__main__":
    normalize_all_speakers()