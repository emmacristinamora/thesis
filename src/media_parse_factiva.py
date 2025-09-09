from pathlib import Path
import re
import pandas as pd

# === PATH CONFIG ===
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "media" / "factiva_raw"
OUT_CSV = ROOT / "data" / "media" / "media_articles_split.csv"

# === SPLIT PATTERN ===
# split *before* each 'HD' line (Factiva article header)
HD_SPLIT = re.compile(r'(?m)(?=^\s*HD\s*$)')

# === HELPERS ===
def parse_fileinfo(fname: str):
    """Extract year, theme, outlet from filename: factiva_{theme}_{year}_{outlet}.txt"""
    stem = Path(fname).stem
    if not stem.startswith("factiva_"):
        return None
    parts = stem.split("_")
    outlet = parts[-1].lower()
    year = int(parts[-2])
    theme = "_".join(parts[1:-2])
    return year, theme, outlet

def clean_block(text: str) -> str:
    """Normalize whitespace & control characters in a block of text."""
    t = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t

def is_real_article(block: str) -> bool:
    """Check if a block looks like a real Factiva article."""
    has_pd = re.search(r'(?m)^\s*PD\s*$', block) is not None
    has_sn = re.search(r'(?m)^\s*SN\s*$', block) is not None
    has_lp = re.search(r'(?m)^\s*LP\s*$', block) is not None
    return has_pd and has_sn and has_lp

def read_text(path: Path) -> str:
    """Read a text file with fallback encoding."""
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")

def split_factiva_file(path: Path):
    """Split a Factiva export into individual article blocks."""
    print(f"   [READ] {path.name}")
    raw = read_text(path)
    print(f"   [INFO] Raw length: {len(raw):,} characters")

    raw = clean_block(raw)
    print(f"   [CLEAN] Normalized text length: {len(raw):,}")

    chunks = HD_SPLIT.split(raw)
    print(f"   [SPLIT] Found {len(chunks)} raw chunks (including preamble)")

    if chunks and not re.match(r'(?m)^\s*HD\s*$', chunks[0].splitlines()[0] if chunks[0].splitlines() else ""):
        print("   [DROP] Dropping preamble before first HD")
        chunks = chunks[1:]

    blocks = [clean_block(c) for c in chunks if c.strip()]
    print(f"   [BLOCKS] {len(blocks)} chunks after cleaning/empty-drop")

    real = [b for b in blocks if is_real_article(b)]
    print(f"   [FILTER] {len(real)} look like real articles (kept)")

    return real

# === MAIN ===
def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("factiva_*.txt"))
    print(f"[INFO] Scanning {len(files)} files from {RAW_DIR}")

    rows = []
    counts = []

    for fp in files:
        info = parse_fileinfo(fp.name)
        if info is None:
            print(f"[SKIP] Unexpected filename: {fp.name}")
            continue
        year, theme, outlet = info

        print(f"\n[PROCESS] {fp.name} → theme={theme}, year={year}, outlet={outlet}")
        blocks = split_factiva_file(fp)
        n_blocks = len(blocks)

        for i, b in enumerate(blocks, start=1):
            rows.append({
                "text": b,
                "year": year,
                "theme": theme,
                "outlet": outlet,
                "article_number": i,
                "source_file": fp.name
            })

        counts.append({"year": year, "theme": theme, "outlet": outlet, "n": n_blocks})
        print(f"   [DONE] {fp.name} → {n_blocks} articles extracted")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[SAVED] {len(df):,} rows → {OUT_CSV}")

    if counts:
        cdf = pd.DataFrame(counts).sort_values(["year","theme","outlet"]).reset_index(drop=True)
        print("\n[SUMMARY] Counts by file (theme/year/outlet):")
        print(cdf)
        print("\n[PREVIEW] First 3 rows:")
        print(df.head(3))

if __name__ == "__main__":
    main()