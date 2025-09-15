# === IDEOLOGICAL LABELING PIPELINE (FEW-SHOT, API) ===
# map debate utterances into Political Compass-style coordinates:
# econ: -1 (Left) .. +1 (Right)
# soc:  -1 (Libertarian) .. +1 (Authoritarian)

from pathlib import Path
import os
import re
import time
import json
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

# OpenAI >= 1.x client
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# === PATHS & CONFIG ===

REPO_DIR      = Path(__file__).resolve().parents[1]
DATA_DIR      = REPO_DIR / "data"
INPUT_FILE    = DATA_DIR / "rhetoric" / "debates_rhetoric.csv"
OUTPUT_FILE   = DATA_DIR / "ideological_drift" / "debates_ideology.csv"

SEED              = 42
MIN_WORDS         = 10           # skip very short/empty utterances
SKIP_MODERATORS   = True         # skip moderators
ENSEMBLE_N        = 2            # how many prompt variants to average
MODEL_NAME        = "gpt-4o-mini"
TEMPERATURE       = 0.0
MAX_TOKENS        = 16           # only need small JSON
CHECKPOINT_EVERY  = 100          # save partial results periodically
BACKOFF_BASE_S    = 1.0
BACKOFF_MAX_S     = 30.0

random.seed(SEED)
np.random.seed(SEED)

# === FEW-SHOT EXAMPLES ===

EXAMPLES = [
    {"text": "We should end mass surveillance programs and protect civil liberties.", "econ": -0.2, "soc": -0.8},
    {"text": "Raise taxes on the wealthy to fund public healthcare and education.", "econ": -0.8, "soc": -0.2},
    {"text": "Cut corporate taxes and increase law-and-order policies to reduce crime.", "econ": 0.8, "soc": 0.6},
    {"text": "Increase military spending and strengthen border security with strict enforcement.", "econ": 0.4, "soc": 0.8},
    {"text": "Deregulate small businesses and reduce government oversight over private life.", "econ": 0.7, "soc": -0.4},
    {"text": "Ban corporate political donations and impose strict platform rules to stop misinformation.", "econ": -0.6, "soc": 0.5},
]

def examples_block():
    lines = []
    for ex in EXAMPLES:
        lines.append(
            f'Text: "{ex["text"]}"\n'
            f'JSON: {{"econ": {ex["econ"]}, "soc": {ex["soc"]}}}'
        )
    return "\n\n".join(lines)

FEWSHOTS = examples_block()

# === PROMPT VARIANTS ===

SYSTEM_MSG = (
    "You are an expert political science coder. "
    "Your task is to map a single debate utterance to a compact political position."
)

def build_prompt_a(utterance: str) -> str:
    return f"""
Classify the ideological content of the utterance on two independent axes:

Economic axis (econ):
-1 = Left (redistribution, public welfare, regulation)
 0 = Neutral/mixed
+1 = Right (markets, low taxes, deregulation)

Social axis (soc):
-1 = Libertarian (civil liberties, personal freedom, light policing)
 0 = Neutral/mixed
+1 = Authoritarian (law-and-order, state control, surveillance)

If a signal is weak/unclear on a dimension, use 0.0.

Return ONLY valid JSON with two floats in [-1, 1] and no extra keys:
{{"econ": <float>, "soc": <float>}}

Few-shot examples:
{FEWSHOTS}

Utterance: "{utterance}"
JSON:
""".strip()

def build_prompt_b(utterance: str) -> str:
    return f"""
You will output a strict JSON object with two keys: econ and soc. Values are floats in [-1,1].

Definitions:
econ: economic left (-1) to economic right (+1).
soc: libertarian (-1) to authoritarian (+1).

Guidelines:
- Focus on policy direction and values expressed.
- If ambiguous/neutral for a dimension, output 0.0
- Do not include explanations, just JSON.

Examples:
{FEWSHOTS}

Utterance: "{utterance}"
JSON:
""".strip()

PROMPT_BUILDERS = [build_prompt_a, build_prompt_b]

# === HELPERS ===

BRACKETED = re.compile(r"\s*[\[\(].*?[\]\)]\s*")
SPACES    = re.compile(r"\s+")

def clean_text(t: str) -> str:
    t = re.sub(BRACKETED, " ", str(t))
    t = re.sub(SPACES, " ", t).strip()
    return t

def is_moderator(row) -> bool:
    val = str(row.get("speaker", "")).strip().lower()
    return val == "moderator" if SKIP_MODERATORS else False

def too_short(text: str) -> bool:
    return len(text.split()) < MIN_WORDS

def clip_unit(x):
    try:
        xf = float(x)
    except Exception:
        return 0.0
    return max(-1.0, min(1.0, xf))

def call_openai(prompt: str):
    backoff = BACKOFF_BASE_S
    while True:
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user",   "content": prompt},
                ],
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg or "please try again" in msg or "timeout" in msg:
                sleep_s = min(BACKOFF_MAX_S, backoff * (1.5 + random.random()))
                print(f"[WAIT] Hit API rate limit, sleeping {sleep_s:.1f}s before retry...")
                time.sleep(sleep_s)
                backoff = sleep_s
                continue
            else:
                print(f"[ERROR] Non-retryable error: {e}")
                raise

def classify_utterance(text: str, ensemble_n: int = ENSEMBLE_N):
    text = clean_text(text)
    scores = []
    for i in range(ensemble_n):
        builder = PROMPT_BUILDERS[i % len(PROMPT_BUILDERS)]
        prompt  = builder(text)
        try:
            js = call_openai(prompt)
            econ = clip_unit(js.get("econ", 0.0))
            soc  = clip_unit(js.get("soc",  0.0))
            scores.append((econ, soc))
        except Exception:
            scores.append((0.0, 0.0))
    econ_vals = np.array([s[0] for s in scores], dtype=float)
    soc_vals  = np.array([s[1] for s in scores], dtype=float)
    return {
        "econ": float(np.mean(econ_vals)),
        "soc":  float(np.mean(soc_vals)),
        "econ_std": float(np.std(econ_vals, ddof=0)),
        "soc_std":  float(np.std(soc_vals,  ddof=0)),
    }

# === MAIN ===

def main():
    print(f"[INFO] Reading {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    # Initialize columns with correct dtypes
    if "notes" not in df.columns:
        df["notes"] = pd.Series([np.nan] * len(df), dtype="object")
    for col in ["econ", "soc", "econ_std", "soc_std"]:
        if col not in df.columns:
            df[col] = np.nan

    to_process_idx = []
    for i, row in df.iterrows():
        text = clean_text(row.get("text", ""))
        if SKIP_MODERATORS and is_moderator(row):
            df.at[i, "notes"] = "moderator"
            continue
        if too_short(text):
            df.at[i, "notes"] = "too_short"
            continue
        if pd.isna(row.get("rhetoric_label")):
            df.at[i, "notes"] = "missing_rhetoric"
            continue
        to_process_idx.append(i)

    print(f"[INFO] Total rows: {len(df)} | To classify: {len(to_process_idx)} | Skipped: {len(df)-len(to_process_idx)}")

    for n, i in enumerate(tqdm(to_process_idx, desc="Classifying ideology")):
        text = df.at[i, "text"]
        results = classify_utterance(text, ENSEMBLE_N)
        df.at[i, "econ"]     = results["econ"]
        df.at[i, "soc"]      = results["soc"]
        df.at[i, "econ_std"] = results["econ_std"]
        df.at[i, "soc_std"]  = results["soc_std"]
        if (n + 1) % CHECKPOINT_EVERY == 0:
            tmp_path = OUTPUT_FILE.with_suffix(".checkpoint.csv")
            df.to_csv(tmp_path, index=False)
            print(f"[CHECKPOINT] Saved -> {tmp_path}")

    cols_keep = ["utterance_id", "debate_id", "year", "speaker", "party", "text",
                 "rhetoric_label", "econ", "soc", "econ_std", "soc_std", "notes"]
    cols_keep = [c for c in cols_keep if c in df.columns]
    df_slim = df[cols_keep].copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_slim.to_csv(OUTPUT_FILE, index=False)
    print(f"[DONE] Saved ideology annotations -> {OUTPUT_FILE}")

    avail = df_slim.dropna(subset=["econ", "soc"])
    if "party" in avail.columns:
        print("\n[INFO] Party-wise mean positions (econ, soc):")
        print(avail.groupby("party")[["econ", "soc"]].mean().round(3))
    if "year" in avail.columns:
        print("\n[INFO] Decade-wise mean positions (econ, soc):")
        decade = (avail["year"] // 10 * 10).astype("Int64")
        print(avail.groupby(decade)[["econ", "soc"]].mean().round(3))

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your environment before running.")
    main()