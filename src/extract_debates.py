import os
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import numpy as np

# === BASE REPO PATH === #
REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_PATH = DATA_DIR / "debates_metadata.csv"
LOG_DIR = REPO_DIR / "logs"
LOG_FILE = LOG_DIR / "extract_transcripts.log"

# === SETUP LOGGING === #
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def sanitize(text):
    return text.replace(" ", "").replace(".", "").replace(",", "").replace("/", "_")

def extract_transcripts():
    logging.info("Starting debate transcript extraction...")

    # ensure raw data directory exists
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # load and clean metadata
    df = pd.read_csv(METADATA_PATH)
    df["filename"] = df["filename"].astype("string")
    df.replace("-", np.nan, inplace=True)

    for idx, row in df.iterrows():
        if pd.isna(row["transcript_url"]) or row.get("transcript_downloaded", False) is True:
            continue

        # build filename
        year = str(row["year"])
        number = str(row["debate_number"])
        debate_type = sanitize(str(row["debate_type"]).capitalize())
        cand_r = sanitize(str(row["candidate_R"])) if pd.notna(row["candidate_R"]) else None
        cand_d = sanitize(str(row["candidate_D"])) if pd.notna(row["candidate_D"]) else None
        cand_i = sanitize(str(row["candidate_I"])) if pd.notna(row["candidate_I"]) else None 

        # handle missing candidates
        if cand_d and cand_i:
            candidates = f"{cand_r}_{cand_d}_{cand_i}"
        elif cand_d:
            candidates = f"{cand_r}_{cand_d}"
        elif cand_i:
            candidates = f"{cand_r}_{cand_i}"
        else:
            candidates = f"{cand_r}_NA"

        filename = f"{year}_{number}_{debate_type}_{candidates}.txt"
        filepath = RAW_DIR / filename

        # skip if already downloaded
        if filepath.exists():
            logging.info(f"Skipping {filename}, already exists.")
            df.at[idx, "filename"] = filename
            df.at[idx, "transcript_downloaded"] = True
            continue

        url = row["transcript_url"]

        for attempt in range(3):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                transcript_div = soup.find("div", class_="field-docs-content")

                if transcript_div:
                    text = transcript_div.get_text(separator="\n", strip=True)
                else:
                    text = soup.get_text(separator="\n", strip=True)  # fallback

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(text)

                df.at[idx, "filename"] = filename
                df.at[idx, "transcript_downloaded"] = True
                logging.info(f"Downloaded: {filename}")
                break
            except Exception as e:
                logging.warning(f"Attempt {attempt+1} failed for {url}: {e}")
                time.sleep(2)
        else:
            logging.error(f"Failed to download {url} after 3 attempts.")

    # save updated metadata
    df.to_csv(METADATA_PATH, index=False)
    logging.info("Updated metadata saved.")

if __name__ == "__main__":
    extract_transcripts()