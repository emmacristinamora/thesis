from pathlib import Path
import re
import argparse
import pandas as pd
import numpy as np


# === PATH CONFIG ===
REPO_DIR = Path(".").resolve().parents[0]
DATA_DIR = REPO_DIR / "data"
MEDIA_DIR = DATA_DIR / "media"

INPUT_CSV_DEFAULT  = MEDIA_DIR / "media_articles_split.csv"
OUTPUT_CSV_DEFAULT = MEDIA_DIR / "media_articles_clean.csv"

TAG_TOKENS = {
    "HD","BY","WC","PD","SN","SC","PG","LA","CY","LP","TD",
    "NS","RE","IPC","IPD","PUB","AN","ART","SE","ED","CO","NOTE"
}
TAIL_TAGS = {"NS","RE","IPC","IPD","PUB","AN","ART","SE"}

# tail phrases that often mark the end (we hard-cut if they appear)
TAIL_PHRASES = [
    "Dow Jones & Company, Inc.",
    "The New York Times Company",
    "Document ",                  # e.g., "Document J0000..."
]

# outlet → leaning (to align with debates dataset coding idea)
OUTLET_LEANING = {
    "nyp": "R",   # New York Post
    "nyt": "D",   # New York Times
    "wsj": np.nan # Wall Street Journal (center-right/market—leave NaN)
}


# === HELPERS ===
# simple cache for tag line presence checks
TAG_LINE_CACHE = {}

def has_tag_line(chunk: str, tag: str) -> bool:
    """Return True if a line that is exactly the tag exists anywhere in chunk."""
    key = (id(chunk), tag)
    # id(chunk) only helps per-run caching; safe to compute each time too
    pattern = rf"(?m)^\s*{re.escape(tag)}\s*$"
    return re.search(pattern, chunk) is not None

def _is_tag_line(s: str) -> bool:
    return s.strip() in TAG_TOKENS


def _extract_block(chunk: str, start_tag: str, end_tags: set) -> str:
    """
    Return text between start_tag and the first line that is a tail tag (end_tags).
    Also hard-cut on TAIL_PHRASES if they appear in the block.
    """
    lines = chunk.splitlines()
    out, started = [], False
    for ln in lines:
        s = ln.strip()
        if not started:
            if s == start_tag:
                started = True
            continue
        # inside block:
        if s in end_tags:  # hit a tail tag like NS/RE/...
            break
        out.append(ln)

    text = "\n".join(out).strip()

    # hard cutoff at known phrases if present
    cut_positions = [text.find(p) for p in TAIL_PHRASES if p in text]
    if cut_positions:
        tcut = min([c for c in cut_positions if c >= 0], default=-1)
        if tcut > 0:
            text = text[:tcut].rstrip()

    # drop any tag-only lines that slipped in
    text = "\n".join([ln for ln in text.splitlines() if not _is_tag_line(ln)]).strip()

    # collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def extract_headline(chunk: str, fallback: str = "") -> str:
    """
    Headline = lines after 'HD' and before 'BY' (preferred),
               else stop at first tag line.
    """
    lines = chunk.splitlines()
    col, seen_hd = [], False
    for ln in lines:
        s = ln.strip()
        if not seen_hd:
            if s == "HD":
                seen_hd = True
            continue
        if s == "BY" or _is_tag_line(s):
            break
        col.append(ln)

    headline = " ".join([c.strip() for c in col if c.strip()]) or fallback
    # normalize whitespace in headline
    headline = re.sub(r"\s+", " ", headline).strip()
    return headline


def extract_body(chunk: str, outlet: str = "") -> str:
    """
    Prefer TD…tail. If no TD exists, fall back to LP…tail.
    Then run outlet-specific cleanup for footer lines.
    """
    # robust tag presence (anywhere, full line)
    has_td = has_tag_line(chunk, "TD")
    has_lp = has_tag_line(chunk, "LP")

    if has_td:
        body = _extract_block(chunk, "TD", TAIL_TAGS)
    elif has_lp:
        body = _extract_block(chunk, "LP", TAIL_TAGS)
    else:
        return ""  # no TD/LP → no body

    # generic footer noise to strip from body lines
    drop_line_patterns = [
        r"^Appeared in the .* print edition.*",  # WSJ print footer
        r"^Write to .+@wsj\.com$",
        r"^Copyright \d{4} .* All Rights Reserved\.$",
        r"^\(c\)\s*\d{4}.*All rights reserved\.$",
        r"^All Rights Reserved\.$",
        r"^.*This material may not be published.*$",
        r"^.*Distributed by.*$",
    ]
    compiled = [re.compile(pat) for pat in drop_line_patterns]

    cleaned_lines = []
    for ln in body.splitlines():
        s = ln.strip()
        if s in TAG_TOKENS:
            continue
        if any(rgx.match(s) for rgx in compiled):
            continue
        # drop straggler factiva index codes like "gdip : ..." "usa : ..." etc.
        if ":" in s and len(s) < 120 and re.match(r"^[a-z]{2,5}\s*:\s*", s, flags=re.I):
            continue
        cleaned_lines.append(ln)

    body = "\n".join(cleaned_lines).strip()

    # normalize whitespace
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    return body


# === PIPELINE ===
def main(input_csv: Path, output_csv: Path, preview_n: int = 5):
    print(f"[INFO] Reading split articles from: {input_csv}")
    df = pd.read_csv(input_csv)

    required_cols = {"text", "year", "theme", "outlet", "article_number", "source_file"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    print(f"[INFO] Rows in: {len(df):,}")
    print("[STEP] Extracting headline & body …")

    headlines = []
    bodies = []
    td_flags = []
    lp_flags = []

    for i, row in df.iterrows():
        chunk   = str(row["text"] or "")
        outlet  = str(row["outlet"]).lower().strip()

        # flags (for diagnostics only)
        has_td = has_tag_line(chunk, "TD")
        has_lp = has_tag_line(chunk, "LP")
        td_flags.append(has_td)
        lp_flags.append(has_lp)

        headline = extract_headline(chunk, fallback="")
        body     = extract_body(chunk, outlet=outlet)

        headlines.append(headline)
        bodies.append(body)

        if (i+1) % 200 == 0 or i == len(df)-1:
            print(f"  [..] processed {i+1:,}/{len(df):,}")

    df["headline"] = headlines
    df["body"]     = bodies
    df["has_td"]   = td_flags
    df["has_lp"]   = lp_flags

    # add outlet_leaning
    df["outlet"] = df["outlet"].str.lower().str.strip()
    df["outlet_leaning"] = df["outlet"].map(OUTLET_LEANING)

    # small quality checks
    df["headline_len"] = df["headline"].fillna("").str.len()
    df["body_len"]     = df["body"].fillna("").str.len()
    df["body_words"]   = df["body"].fillna("").str.split().map(len)

    print("\n[QC] Body length stats (chars):")
    print(df["body_len"].describe())
    print("\n[QC] Body word count stats:")
    print(df["body_words"].describe())

    # drop rows with empty body (rare but safer)
    before = len(df)
    df = df[df["body"].str.strip().str.len() > 0].copy()
    after = len(df)
    if after < before:
        print(f"[CLEAN] Dropped {before-after} rows with empty body.")

    # final column order (minimal + useful diagnostics)
    cols = [
        "year", "theme", "outlet", "outlet_leaning", "article_number",
        "headline", "body",
        "has_td", "has_lp",
        "source_file",
        "body_words", "body_len", "headline_len"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df_out = df[cols].copy()

    # save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"\n[SAVED] {len(df_out):,} cleaned rows → {output_csv}")

    # preview
    print("\n[PREVIEW] Sample rows:")
    with pd.option_context("display.max_colwidth", 120):
        print(df_out.head(preview_n))


# === ENTRY POINT ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Factiva-split media articles into headline/body with TD-first logic.")
    parser.add_argument("--input",  type=str, default=str(INPUT_CSV_DEFAULT),  help="Path to media_articles_split.csv")
    parser.add_argument("--output", type=str, default=str(OUTPUT_CSV_DEFAULT), help="Path to save cleaned CSV")
    parser.add_argument("--preview", type=int, default=5, help="Number of preview rows to print")
    args = parser.parse_args()

    in_csv  = Path(args.input)
    out_csv = Path(args.output)

    print("[CONFIG]")
    print("  REPO_DIR:", REPO_DIR)
    print("  MEDIA_DIR:", MEDIA_DIR)
    print("  INPUT :", in_csv)
    print("  OUTPUT:", out_csv)

    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    main(in_csv, out_csv, preview_n=args.preview)