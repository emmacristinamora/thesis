import os
import re
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# === PATHS === #
REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = REPO_DIR / "logs"
LOG_FILE = LOG_DIR / "extract_utterances.log"

# === FILES TO PROCESS BY FORMAT TYPE === #
DEBATE_FILES = {
    "all_caps_inline": [
        "1960_1_Presidential_Nixon_Kennedy.txt",
        "1960_2_Presidential_Nixon_Kennedy.txt",
        "1960_3_Presidential_Nixon_Kennedy.txt",
        "1960_4_Presidential_Nixon_Kennedy.txt",
        "1976_1_Presidential_Ford_Carter.txt",
        "1976_2_Presidential_Ford_Carter.txt",
        "1976_3_Vice_presidential_Dole_Mondale.txt",
        "1976_4_Presidential_Ford_Carter.txt",
        "1980_1_Presidential_Reagan_Anderson.txt",
        "1980_2_Presidential_Reagan_Carter.txt",
        "1984_2_Vice_presidential_Bush(Sr)_Ferraro.txt",
        "1988_1_Presidential_Bush(Sr)_Dukakis.txt",
        "1988_2_Vice_presidential_Quayle_Bentsen.txt",
        "1988_3_Presidential_Bush(Sr)_Dukakis.txt",
        "1992_2_Vice_presidential_Quayle_AlGore_Stockdale.txt",
        "1996_2_Vice_presidential_Kemp_AlGore.txt",
        "2000_1_Presidential_Bush(Jr)_AlGore.txt",
        "2000_2_Vice_presidential_Cheney_Lieberman.txt",
        "2000_3_Presidential_Bush(Jr)_AlGore.txt",
        "2000_4_Presidential_Bush(Jr)_AlGore.txt",
        "2004_2_Vice_presidential_Cheney_Edwards.txt",
        "2008_1_Presidential_McCain_Obama.txt"
    ],
    "all_caps_newline": [
        "2008_2_Vice_presidential_Palin_Biden.txt",
        "2008_3_Presidential_McCain_Obama.txt",
        "2008_4_Presidential_McCain_Obama.txt",
        "2012_2_Vice_presidential_Ryan_Biden.txt",
        "2016_1_Presidential_Trump_Clinton(Hillary).txt",
        "2016_2_Vice_presidential_Pence_Kaine.txt",
        "2016_3_Presidential_Trump_Clinton(Hillary).txt",
        "2016_4_Presidential_Trump_Clinton(Hillary).txt",
        "2020_1_Presidential_Trump_Biden.txt",
        "2020_2_Vice_presidential_Pence_Harris.txt",
        "2020_3_Presidential_Trump_Biden.txt",
        "2024_1_Presidential_Trump_Biden.txt",
        "2024_2_Presidential_Trump_Harris.txt",
        "2024_3_Vice_presidential_Vance_Walz.txt"
    ],
    "title_newline": [
        "1984_1_Presidential_Reagan_Mondale.txt",
        "1984_3_Presidential_Reagan_Mondale.txt",
        "1992_1_Presidential_Bush(Sr)_Clinton(Bill)_Perot.txt",
        "1992_3_Presidential_Bush(Sr)_Clinton(Bill)_Perot.txt",
        "1992_4_Presidential_Bush(Sr)_Clinton(Bill)_Perot.txt",
        "1996_1_Presidential_Dole_Clinton(Bill).txt",
        "1996_3_Presidential_Dole_Clinton(Bill).txt",
        "2004_1_Presidential_Bush(Jr)_Kerry.txt",
        "2004_3_Presidential_Bush(Jr)_Kerry.txt",
        "2004_4_Presidential_Bush(Jr)_Kerry.txt",
        "2012_1_Presidential_Romney_Obama.txt",
        "2012_3_Presidential_Romney_Obama.txt",
        "2012_4_Presidential_Romney_Obama.txt"
    ]
}

# === LOGGING SETUP === #
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === REGEX PATTERNS FOR CAPS FORMATS === #
PATTERNS = {
    "all_caps_inline": re.compile(r"\n?([A-Z ]{2,}):\s?(.*?)(?=\n[A-Z ]{2,}:|\Z)", re.DOTALL),
    "all_caps_newline": re.compile(r"\n?([A-Z ]{2,}):\n(.*?)(?=\n[A-Z ]{2,}:|\Z)", re.DOTALL)
}

# === TITLE SPEAKER DETECTION === #
def is_speaker_line(line, max_len=30):
    line = line.strip()
    prefixes = [
        'Mr\\.', 'Ms\\.', 'Mrs\\.', 'President', 'The President',
        'Governor', 'Gov\\.', 'Senator', 'Sen\\.', 'Question', 'Q', 'Moderator'
    ]
    # match standalone titles or title + name, followed by a period
    pattern = r'^(' + '|'.join(prefixes) + r')(?:\s+[A-Z][a-z]+)?\.$'
    return len(line) <= max_len and line.endswith('.') and re.match(pattern, line)

# === PARSERS === #
def parse_regex_format(file_path, pattern, debate_id):
    # read entire file content
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    utterances = []
    # extract all matches using the regex pattern
    for i, match in enumerate(re.finditer(pattern, text)):
        speaker = match.group(1).strip()
        content = match.group(2).strip().replace("\n", " ")
        utterances.append({
            "utterance_id": f"{debate_id}_{i+1}",
            "debate_id": debate_id,
            "speaker": speaker,
            "text": content
        })

    return pd.DataFrame(utterances)

def parse_title_newline_format(file_path, debate_id):
    # read file line by line
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    utterances = []
    current_speaker = None
    current_utterance = []

    for line in lines:
        if is_speaker_line(line):
            # save previous speaker's utterance
            if current_speaker and current_utterance:
                utterances.append({
                    "debate_id": debate_id,
                    "speaker": current_speaker.rstrip('.'),
                    "text": ' '.join(current_utterance).strip()
                })
                current_utterance = []
            current_speaker = line.strip()
        else:
            # collect utterance text
            current_utterance.append(line.strip())

    # append final utterance if exists
    if current_speaker and current_utterance:
        utterances.append({
            "debate_id": debate_id,
            "speaker": current_speaker.rstrip('.'),
            "text": ' '.join(current_utterance).strip()
        })

    df = pd.DataFrame(utterances)
    df.insert(0, "utterance_id", range(1, len(df) + 1))
    return df

# === MAIN RUNNER === #
def process_utterances():
    logging.info("Starting utterance extraction...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # count total files for progress bar
    total_files = sum(len(files) for files in DEBATE_FILES.values())
    pbar = tqdm(total=total_files, desc="Processing debates", ncols=100)

    for format_type, file_list in DEBATE_FILES.items():
        for filename in file_list:
            file_path = RAW_DIR / filename
            debate_id = filename.replace(".txt", "")

            if not file_path.exists():
                msg = f"❌ File not found: {filename}"
                print(msg)
                logging.warning(msg)
                pbar.update(1)
                continue

            try:
                # select parser based on format type
                if format_type in ["all_caps_inline", "all_caps_newline"]:
                    df = parse_regex_format(file_path, PATTERNS[format_type], debate_id)
                elif format_type == "title_newline":
                    df = parse_title_newline_format(file_path, debate_id)
                else:
                    msg = f"⚠️ Unknown format type: {format_type}"
                    print(msg)
                    logging.error(msg)
                    pbar.update(1)
                    continue

                # save output to processed directory
                output_path = PROCESSED_DIR / f"{debate_id}_utterances.csv"
                df.to_csv(output_path, index=False)
                msg = f"✅ {len(df)} utterances extracted from {filename}"
                print(msg)
                logging.info(msg)
            except Exception as e:
                msg = f"❌ Failed to process {filename}: {e}"
                print(msg)
                logging.error(msg)

            pbar.update(1)

    pbar.close()
    print("🎉 All utterances extracted.")
    logging.info("Utterance extraction complete.")

# === ENTRYPOINT === #
if __name__ == "__main__":
    process_utterances()