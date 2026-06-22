"""
download_dalvii.py — 下載 Dalvii GitHub 榴槤敲擊音數據集
92 個 WAV 文件，Dona 品種，2 類:
  75-85% (未熟/unripe) : 45 個
  95-Ripe (熟/ripe)    : 47 個
授權: 開源 (MIT-like)
Source: https://github.com/Dalvii/durian-maturity-classification

用法:
  python download_dalvii.py              # 下載到默認目錄
  python download_dalvii.py /path/to/dir # 下載到指定目錄
"""
import os
import sys
import json
import urllib.request
from pathlib import Path

GITHUB_REPO = "Dalvii/durian-maturity-classification"
AUDIO_PATH = "AUDIO_DATA"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{AUDIO_PATH}"
GITHUB_RAW = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{AUDIO_PATH}"

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "dalvii_audio"

# Label mapping (filename keyword → unified label)
LABEL_MAP = {
    "75-85%": "unripe",
    "95-Ripe": "ripe",
}


def get_file_list() -> list[dict]:
    """Get list of WAV files via GitHub API."""
    req = urllib.request.Request(GITHUB_API)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "DurianAI-DataDownloader/1.0")

    with urllib.request.urlopen(req, timeout=30) as resp:
        contents = json.loads(resp.read().decode())

    return [f for f in contents if f["name"].endswith(".wav")]


def infer_label(filename: str) -> str | None:
    """Infer unified label from filename."""
    for key, label in LABEL_MAP.items():
        if key in filename:
            return label
    return None


def download(output_dir: Path | None = None):
    """Download all WAV files from Dalvii dataset."""
    output_dir = output_dir or DEFAULT_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching file list from GitHub ({GITHUB_REPO})...")
    try:
        wav_files = get_file_list()
    except Exception as e:
        print(f"Error fetching GitHub API: {e}")
        print("Try manual clone:")
        print(f"  git clone https://github.com/{GITHUB_REPO}.git")
        print(f"  cp -r durian-maturity-classification/AUDIO_DATA/* {output_dir}")
        return False

    print(f"Found {len(wav_files)} WAV files\n")

    # Count by class
    class_counts: dict[str, int] = {}
    downloaded = 0
    skipped = 0
    failed = 0

    for i, f in enumerate(wav_files, 1):
        filename = f["name"]
        url = f["download_url"]
        size_kb = f["size"] / 1024
        filepath = output_dir / filename
        label = infer_label(filename)

        if label:
            class_counts[label] = class_counts.get(label, 0) + 1

        if filepath.exists() and filepath.stat().st_size == f["size"]:
            print(f"  [{i:2d}/{len(wav_files)}] [SKIP] {filename} ({size_kb:.0f} KB) [{label}]")
            skipped += 1
            continue

        print(f"  [{i:2d}/{len(wav_files)}] [DOWN] {filename} ({size_kb:.0f} KB) [{label}]...")
        try:
            urllib.request.urlretrieve(url, str(filepath))
            downloaded += 1
        except Exception as e:
            print(f"  [{i:2d}/{len(wav_files)}] [FAIL] {filename}: {e}")
            failed += 1

    print(f"\nResults: Downloaded={downloaded}, Skipped={skipped}, Failed={failed}")
    print(f"Output: {output_dir.resolve()}")
    print("\nClass distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls:10s}: {count} files")

    # Write metadata JSON
    meta = {
        "dataset": "dalvii",
        "source": f"https://github.com/{GITHUB_REPO}",
        "total_files": len(wav_files),
        "classes": class_counts,
        "label_map": LABEL_MAP,
        "format": "WAV, 16kHz mono",
        "species": "Dona durian",
        "note": "2-class dataset (no overripe). Use only unripe+ripe samples.",
    }
    with open(output_dir / "dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {output_dir / 'dataset_meta.json'}")

    return failed == 0


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    ok = download(out_dir)
    sys.exit(0 if ok else 1)
