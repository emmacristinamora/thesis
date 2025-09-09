from pathlib import Path
import re
import hashlib
import pandas as pd

# === CONFIG ===
REPO_DIR  = Path(".").resolve().parents[0]
DATA_DIR  = REPO_DIR / "data"
MEDIA_DIR = REPO_DIR / "data" / "media"

INPUT_CSV  = MEDIA_DIR / "media_articles_clean_final.csv"
OUTPUT_CSV = DATA_DIR / "media_articles_dataset.csv"

MIN_WORDS = 100                  # drop bodies with fewer words than this
NORMALIZE_NYP_NS = True          # remove 'ns' bullets at line starts for NYP

# lines that are clearly boilerplate/metadataE
# each pattern is matched
# against a *single line* with ^...$ anchoring where appropriate
BOILERPLATE_LINE_PATTERNS = [
    r"^Appeared in the .*$",
    r"^Write to .+@.+\..+$",
    r"^Copyright \d{4}.+All Rights Reserved\.?$",
    r"^All Rights Reserved\.?$",
    r"^Dow Jones & Company, Inc\.$",
    r"^N\.Y\.P\. Holdings, Inc\.$",
    r"^The New York Times Company$",
    r"^Document [A-Z0-9]+$",
    r"^SC$|^ED$|^PG$|^LA$|^CY$|^NS$|^RE$|^PUB$|^AN$|^IPD$|^SE$|^IN$",
    r"^English$",                    # lone language line
    r".*@nypost\.com$",              # an entire line ending with a nypost email
    r"@nypost\.com",                 # or any embedded nypost email
    r"^\s*---+\s*$",                 # horizontal-rule style separator
    r"^Subscribe to WSJ.*$",         # WSJ subscription footer
]

# if a body ends with any of these tails, trim them off (conservative)
TAIL_TRIM_PATTERNS = [
    r"(?s)(.*?)\n+Copyright \d{4}.+All Rights Reserved\.\s*$",
    r"(?s)(.*?)\n+Write to .+@.+\..+\s*$",
    r"(?s)(.*?)\n+Subscribe to WSJ.*\s*$",
]

# === HELPERS ===
def hash_body(text: str) -> str:
    """Stable fingerprint for deduping."""
    norm = re.sub(r"\s+", " ", (text or "")).strip().lower()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()

def strip_boilerplate_lines(text: str) -> str:
    """Remove lines that are obviously boilerplate metadata."""
    if not isinstance(text, str) or not text.strip():
        return text
    lines = text.splitlines()
    cleaned = []
    patterns = [re.compile(p) for p in BOILERPLATE_LINE_PATTERNS]
    for ln in lines:
        line = ln.strip()
        if not line:
            cleaned.append(ln)
            continue
        if any(p.match(line) for p in patterns):
            continue  # drop this line
        cleaned.append(ln)
    t = "\n".join(cleaned)

    # optional tail trimming
    for pat in TAIL_TRIM_PATTERNS:
        m = re.match(pat, t)
        if m:
            t = m.group(1).rstrip()

    return t

def normalize_nyp_bullets(text: str) -> str:
    """
    NYP sometimes has list markers like 'nsSomething...' at start of lines.
    This removes leading 'ns' tokens at line start (with or without space).
    """
    if not NORMALIZE_NYP_NS or not isinstance(text, str):
        return text
    return re.sub(r"(?m)^\s*ns\s*(?=\S)", "", text)

def word_count(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\b\w+\b", text))

# === MAIN ===
def main():
    print("[CONFIG]")
    print(f"  REPO_DIR: {REPO_DIR}")
    print(f"  MEDIA_DIR: {MEDIA_DIR}")
    print(f"  INPUT : {INPUT_CSV}")
    print(f"  OUTPUT: {OUTPUT_CSV}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_CSV}")

    print(f"[INFO] Reading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    n0 = len(df)
    print(f"[INFO] Rows in: {n0:,}")

    required = ["body", "year", "theme", "outlet"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # initial stats
    df["body_words"] = df["body"].apply(word_count)
    print("\n[QC] Initial body word stats:")
    print(df["body_words"].describe())

    # clean first (boilerplate + NYP bullets)
    print("\n[STEP] Trimming boilerplate lines + optional NYP bullet normalization …")
    def _clean_body(row):
        txt = strip_boilerplate_lines(row["body"])
        if str(row.get("outlet", "")).lower() == "nyp":
            txt = normalize_nyp_bullets(txt)
        return txt

    df["body"] = df.apply(_clean_body, axis=1)

    # recompute after cleaning
    df["body_words"] = df["body"].apply(word_count)
    df["body_len"]   = df["body"].apply(lambda t: len(t) if isinstance(t, str) else 0)
    if "headline" in df.columns:
        df["headline_len"] = df["headline"].apply(lambda t: len(str(t)) if pd.notnull(t) else 0)

    # deduplicate on cleaned body
    print("\n[STEP] Dropping exact duplicate bodies …")
    df["body_fp"] = df["body"].apply(hash_body)
    before = len(df)
    df = df.drop_duplicates(subset=["body_fp"], keep="first").reset_index(drop=True)
    print(f"  [OK] Dropped {before - len(df)} duplicate rows")

    # drop short articles after cleaning
    print(f"\n[STEP] Dropping very short bodies (< {MIN_WORDS} words) …")
    before = len(df)
    df = df[df["body_words"] >= MIN_WORDS].reset_index(drop=True)
    print(f"  [OK] Dropped {before - len(df)} short rows")

    print("\n[QC] Post-clean body word stats:")
    print(df["body_words"].describe())

    # final save
    keep_cols = [c for c in [
        "year", "theme", "outlet", "outlet_leaning",
        "article_number", "headline", "body", "source_file"
    ] if c in df.columns] + ["body_words", "body_len"]

    df_final = df[keep_cols].copy()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SAVED] {len(df_final):,} rows → {OUTPUT_CSV}")

    print("\n[SUMMARY] Counts by year/theme/outlet after cleaning:")
    try:
        summary = (
            df_final.groupby(["year", "theme", "outlet"])
                    .size().reset_index(name="n")
                    .sort_values(["year", "theme", "outlet"])
        )
        print(summary.to_string(index=False))
    except Exception as e:
        print(f"  [WARN] Could not print summary: {e}")

if __name__ == "__main__":
    main()