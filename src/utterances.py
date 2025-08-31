import re
import pandas as pd
from pathlib import Path

# === CONFIGURATION === #
TRANSCRIPT_DIR = Path("../data/raw")
OUTPUT_DIR = Path("../data/processed")
DEBATE_FILES = {
    "1984_1": "1984_1_Presidential_Reagan_Mondale.txt"
}

# === UTILITY === #
def normalize(text):
    return text.replace('\r\n', '\n')

# === NEW HONORIFIC-AWARE PATTERN === #
SPEAKER_PATTERN = re.compile(
    r"(?:^|\n)(Mr\.|Ms\.|Mrs\.|The President|Senator|Governor|Moderator)? ?([A-Z][a-z]+(?: [A-Z][a-z]+)*?)\.\n", re.MULTILINE
)

def parse_honorific_format(debate_id, filename):
    file_path = TRANSCRIPT_DIR / filename
    with open(file_path, "r", encoding="utf-8") as f:
        text = normalize(f.read())

    matches = list(SPEAKER_PATTERN.finditer(text))
    utterances = []

    for i, match in enumerate(matches):
        speaker = f"{match.group(1) or ''} {match.group(2)}".strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        utterance = text[start:end].strip().replace('\n', ' ')
        if utterance:
            utterances.append({
                "utterance_id": f"{debate_id}_{i+1}",
                "debate_id": debate_id,
                "speaker": speaker,
                "text": utterance
            })

    df = pd.DataFrame(utterances)
    output_path = OUTPUT_DIR / f"{debate_id}_utterances.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Parsed {len(df)} utterances to {output_path.name}")

# === RUN === #
if __name__ == "__main__":
    for debate_id, filename in DEBATE_FILES.items():
        parse_honorific_format(debate_id, filename)