"""
download_zenodo.py — 下載 Zenodo 多模態榴槤數據集
189 個榴槤樣本：聲學敲擊音 + RGB 圖像 + 多光譜 + 熱成像 + CSV
DOI: 10.5281/zenodo.18603796
授權: CC BY

文件清單:
  README.md                        ~44 KB
  durian_characteristics_cleaned.csv  ~44 KB  ← 標籤/特徵元數據
  dataset_clean_sound.zip          ~1.5 GB   ← 敲擊音 (Priority 1)
  dataset_clean_rgb.zip            ~18.9 GB  ← RGB 圖像 (Priority 1)
  dataset_clean_multispectral.zip  ~928 MB   ← 多光譜 (Optional)
  dataset_clean_thermal.zip        ~512 MB   ← 熱成像 (Optional)

用法:
  python download_zenodo.py              # 下載 CSV + README
  python download_zenodo.py sound        # 下載聲音數據
  python download_zenodo.py rgb          # 下載 RGB 圖像
  python download_zenodo.py sound rgb    # 下載聲音 + RGB (推薦)
  python download_zenodo.py all          # 下載全部 (>21 GB!)
"""
import os
import sys
import urllib.request
import json
from pathlib import Path

ZENODO_API = "https://zenodo.org/api/records/18603796"
DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "zenodo"

# Mapping: user-friendly modality names → filename keywords
MODALITY_MAP = {
    "sound":         "dataset_clean_sound",
    "audio":         "dataset_clean_sound",   # alias
    "rgb":           "dataset_clean_rgb",
    "multispectral": "dataset_clean_multispectral",
    "thermal":       "dataset_clean_thermal",
    "csv":           ".csv",
    "readme":        "README",
}

# Always-download files (tiny, essential)
ESSENTIAL_FILES = {"README.md", "durian_characteristics_cleaned.csv"}


def get_record() -> dict:
    """Fetch Zenodo record metadata via API."""
    print("Fetching Zenodo record metadata...")
    req = urllib.request.Request(ZENODO_API)
    req.add_header("User-Agent", "DurianAI-DataDownloader/1.0")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def download_file(url: str, filepath: Path, size_bytes: int):
    """Download a file with progress reporting."""
    size_mb = size_bytes / (1024 * 1024)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.exists() and filepath.stat().st_size == size_bytes:
        print(f"  [SKIP] {filepath.name} (already complete, {size_mb:.1f} MB)")
        return True

    print(f"  [DOWN] {filepath.name} ({size_mb:.1f} MB)...")
    try:
        def report(block_num, block_size, total_size):
            if total_size > 0 and block_num % 100 == 0:
                pct = min(100, block_num * block_size / total_size * 100)
                print(f"         {pct:.0f}%", end="\r")

        urllib.request.urlretrieve(url, str(filepath), reporthook=report)
        print(f"  [DONE] {filepath.name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {filepath.name}: {e}")
        if filepath.exists():
            filepath.unlink()  # Remove incomplete file
        return False


def main():
    """Main entry: parse args and download."""
    filter_args = [a.lower() for a in sys.argv[1:]]

    # Resolve modality filters → filename keywords
    if "all" in filter_args:
        keyword_filters = None  # No filter = download all
    elif filter_args:
        keyword_filters = set()
        for arg in filter_args:
            if arg in MODALITY_MAP:
                keyword_filters.add(MODALITY_MAP[arg])
            else:
                print(f"  [WARN] Unknown modality '{arg}'. Valid: {', '.join(MODALITY_MAP.keys())}")
    else:
        # Default: only essentials (CSV + README)
        keyword_filters = {".csv", "README"}

    try:
        record = get_record()
    except Exception as e:
        print(f"Error fetching Zenodo API: {e}")
        print("Manual download: https://zenodo.org/records/18603796")
        sys.exit(1)

    title = record.get("metadata", {}).get("title", "N/A")
    print(f"Record: {title}")

    files = record.get("files", [])
    print(f"Total files: {len(files)}\n")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    failed = 0

    for f in files:
        filename = f["key"]
        url = f["links"]["self"]
        size = f["size"]
        filepath = DATA_DIR / filename

        # Decide whether to download
        should_download = False
        if filename in ESSENTIAL_FILES:
            should_download = True
        elif keyword_filters is None:
            should_download = True
        else:
            for kw in keyword_filters:
                if kw.lower() in filename.lower():
                    should_download = True
                    break

        if not should_download:
            print(f"  [SKIP] {filename} ({size/1024/1024:.1f} MB) — not in filter")
            skipped += 1
            continue

        ok = download_file(url, filepath, size)
        if ok:
            downloaded += 1
        else:
            failed += 1

    print(f"\nDone. Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")
    print(f"Data directory: {DATA_DIR.resolve()}")

    if not filter_args or filter_args == []:
        print("\nTip: Run with 'sound' or 'rgb' to download actual data:")
        print("  python download_zenodo.py sound rgb")


if __name__ == "__main__":
    main()
