# === RHETORICAL LABELING PIPELINE (FEW-SHOT, API) ===
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import re

# === PATHS ===
REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_DIR / "data"
ANNOTATED_FILE = DATA_DIR / "rhetoric" / "annotated_debates_rhetoric.csv"
DEBATES_FILE   = DATA_DIR / "debates_df_themes.csv"
OUTPUT_FILE    = DATA_DIR / "rhetoric" / "debates_rhetoric.csv"

# === LOAD DATA ===
df_fewshot = pd.read_csv(ANNOTATED_FILE).dropna(subset=["label"]).reset_index(drop=True)
df_debates = pd.read_csv(DEBATES_FILE).reset_index(drop=True)

print(f"[INFO] Few-shot dataset: {len(df_fewshot)} rows")
print(f"[INFO] Debates dataset: {len(df_debates)} rows")

# === BUILD FEW-SHOT EXAMPLES ===
examples = []
for label in ["attack", "acclaim", "defense"]:
    subset = df_fewshot[df_fewshot["label"] == label].sample(3, random_state=42)
    for _, row in subset.iterrows():
        examples.append(f'Text: "{row["text"]}"\nLabel: {row["label"].capitalize()}')

fewshot_block = "\n\n".join(examples)

# === PROMPT TEMPLATE ===
system_prompt = """You are an expert political communication analyst.
You classify U.S. presidential debate utterances into one of three rhetorical strategies
based on Benoit's functional theory of political discourse:

- attack: Criticizing an opponent.
- acclaim: Promoting oneself, one’s record, or one’s proposals.
- defense: Responding to criticism or clarifying one’s record.

Always return exactly ONE label: attack, acclaim, or defense.
If an utterance seems to fit multiple labels, choose the predominant one.
Do NOT return multiple labels. Just return: attack, acclaim, or defense."""

def make_prompt(utterance: str) -> str:
    return f"""{system_prompt}

Here are some labeled examples:

{fewshot_block}

Now classify the following utterance:

Text: "{utterance}"
Label:"""

# === OPENAI API CONFIG ===
from openai import OpenAI
client = OpenAI(api_key="")

valid_labels = {"attack", "acclaim", "defense"}

def clean_label(raw_label: str) -> str:
    """Normalize and extract the label."""
    raw_label = raw_label.strip().lower()  # normalize to lowercase
    for lbl in valid_labels:
        if lbl in raw_label:
            return lbl
    return "unspecified"  # fallback

def classify_batch(texts):
    results = []
    for text in tqdm(texts, desc="Classifying"):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": make_prompt(text)},
                ],
                temperature=0.0,
                max_tokens=5
            )
            raw = response.choices[0].message.content
            label = clean_label(raw)
            results.append(label)
        except Exception as e:
            print("[ERROR]", e)
            results.append("unspecified")
    return results

# === RUN CLASSIFICATION ===
texts = df_debates["text"].fillna("").tolist()
labels = classify_batch(texts)

# === SAVE RESULTS ===
df_debates["rhetoric_label"] = labels

# keep only relevant columns
cols_keep = ["utterance_id", "debate_id", "speaker", "party", "text", "rhetoric_label"]

# make sure columns exist (some datasets may not have all)
cols_keep = [c for c in cols_keep if c in df_debates.columns]
df_slim = df_debates[cols_keep]
df_slim.to_csv(OUTPUT_FILE, index=False)

print(f"[DONE] Saved slim labeled dataset: {OUTPUT_FILE}")
print(df_slim.head())

# === SANITY CHECKS ===
print("\n[INFO] Label distribution:")
print(df_slim["rhetoric_label"].value_counts())

print("\n[INFO] Random sample predictions:")
print(df_slim[["text", "rhetoric_label"]].sample(5, random_state=42))