# === WORD DRIFT ANALYSIS (UPDATED WITH ALIASES) ===
# track how selected political keywords drift in meaning across decades
# using canonicalized anchors (aliases collapsed)

import pandas as pd
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from tqdm import tqdm
import json
from sklearn.metrics.pairwise import cosine_similarity

# === PATHS ===
REPO_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR   = REPO_DIR / "data"
INPUT_FILE = DATA_DIR / "debates_df_themes.csv"
OUTPUT_DIR = DATA_DIR / "ideological_drift"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "word_drift.csv"

# === CONFIG ===
MIN_WORDS = 8     # skip utterances shorter than this
EMB_SIZE = 100
WINDOW = 5
MIN_COUNT = 3
WORKERS = 4

# === ANCHORS & ALIASES (canonicalization) ===
ANCHORS = {
    "freedom": ["freedom", "freedoms", "liberty", "liberties", "free"],
    "america": ["america", "american", "americans"],
    "taxes": ["tax", "taxes", "taxation", "taxpayer", "taxpayers"]
}

# === REVERSE LOOKUP FOR CANONICALIZATION ===
ALIAS2ANCHOR = {}
for anchor, aliases in ANCHORS.items():
    for alias in aliases:
        ALIAS2ANCHOR[alias] = anchor

# === LOAD DATA ===
print(f"[INFO] Reading {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

# === BASIC TOKENIZATION & CANONICALIZATION ===
def preprocess_and_canonicalize(text):
    if pd.isna(text):
        return []
    tokens = simple_preprocess(str(text), deacc=True, min_len=2)
    canonicalized = [ALIAS2ANCHOR.get(tok, tok) for tok in tokens]
    return canonicalized

df["tokens"] = df["text"].apply(preprocess_and_canonicalize)

# use existing decade column
decades = sorted(df["decade"].dropna().unique())
print(f"[INFO] Found decades: {decades}")

# === TRAIN W2V MODELS PER DECADE ===
models = {}
for dec in decades:
    corpus = df.loc[df["decade"] == dec, "tokens"].tolist()
    # skip if no tokens
    if len(corpus) == 0:
        continue
    model = Word2Vec(
        sentences=corpus,
        vector_size=EMB_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=1,  # skip-gram (better for semantics)
        epochs=20
    )
    models[dec] = model.wv
    print(f"[INFO] Trained Word2Vec for {dec}s on {len(corpus)} utterances")

# === ANALYZE DRIFT ===
rows = []
for word in ANCHORS.keys():   # iterate over canonical anchors only
    prev_vec = None
    for dec in decades:
        wv = models.get(dec, None)
        if wv is None or word not in wv.key_to_index:
            rows.append({
                "word": word,
                "decade": dec,
                "utterance_count": 0,
                "neighbors": json.dumps([]),
                "similarity_to_prev": np.nan
            })
            continue

        # frequency (utterances containing this anchor)
        utterance_count = sum(word in toks for toks in df.loc[df["decade"] == dec, "tokens"])

        # neighbors (semantic field)
        neighbors = []
        try:
            neighbors = [w for w, _ in wv.most_similar(word, topn=10)]
        except KeyError:
            neighbors = []

        # similarity to previous decade
        vec = wv[word]
        sim = np.nan
        if prev_vec is not None:
            sim = float(cosine_similarity([vec], [prev_vec])[0][0])
        prev_vec = vec

        rows.append({
            "word": word,
            "decade": dec,
            "utterance_count": utterance_count,
            "neighbors": json.dumps(neighbors),
            "similarity_to_prev": sim
        })

# === SAVE OUTPUT ===
out_df = pd.DataFrame(rows)
out_df.to_csv(OUTPUT_FILE, index=False)
print(f"[DONE] Saved word drift results -> {OUTPUT_FILE}")
print(out_df.head())