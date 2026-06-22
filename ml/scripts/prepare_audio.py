"""
prepare_audio.py — 音頻預處理管線
統一採樣率 → 提取 Mel Spectrogram/MFCC → 標籤映射 → 分割訓練/驗證/測試集

支持數據集:
  dalvii  : 92 WAV, 2-class, 標籤從文件名解析
  zenodo  : 189 WAV (解壓後), 3-class, 標籤從 CSV 解析

統一 3-class 標籤: unripe / ripe / overripe

用法:
  python prepare_audio.py                          # 處理 dalvii (默認)
  python prepare_audio.py --dataset zenodo         # 處理 zenodo
  python prepare_audio.py --dataset dalvii zenodo  # 合並處理
  python prepare_audio.py --feature mfcc           # 提取 MFCC 替代 Mel
"""
import os
import sys
import csv
import glob
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

# Try importing audio/ML libs (not available in CI, installed in Colab/venv)
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("[WARN] librosa not installed. Run: pip install librosa soundfile")

try:
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] sklearn not installed. Run: pip install scikit-learn")

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "audio"

TARGET_SR = 16000
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 40
SEGMENT_DURATION = 2.0   # seconds — 2s covers a typical knock+resonance
OVERLAP_RATIO = 0.5      # 50% overlap for more segments

# Unified 3-class label system
VALID_LABELS = {"unripe", "ripe", "overripe"}

# ============================================================
# Label Mapping per Dataset
# ============================================================

def get_dalvii_label(filename: str) -> Optional[str]:
    """Parse label from Dalvii filename convention.
    Format: YYYYMMDD_HHMMSS_Dona_{ripeness}output.wav
    Examples:
      20240929_023338_Dona_75-85%output.wav  → unripe
      20241002_083225_Dona_95-Ripeoutput.wav → ripe
    """
    name = Path(filename).stem
    if "75-85%" in name:
        return "unripe"
    elif "95-Ripe" in name or "95%Ripe" in name:
        return "ripe"
    return None


def get_zenodo_label_from_code(code: str) -> Optional[str]:
    """Parse label from Zenodo Code field.
    Format: {Maturity}_{Class}_{Ripeness}_{Replicate}
    Maturity: IM (Immature), M (Mature), OM (Overmature)
    Class: CA, CB, CC
    Ripeness: UN (Unripe), RI (Ripe), OR (Overripe)
    Examples:
      IM_CA_UN_1  → unripe
      M_CB_RI_92  → ripe
      OM_CC_OR_189 → overripe
    """
    parts = code.split("_")
    if len(parts) < 3:
        return None
    ripeness_code = parts[2]
    mapping = {"UN": "unripe", "RI": "ripe", "OR": "overripe"}
    return mapping.get(ripeness_code)


def load_zenodo_csv_labels(csv_path: Path) -> dict[str, str]:
    """Load Zenodo CSV and build Code → label mapping.
    Only includes samples with valid Actual_Ripening_Status.
    Excludes: Disease samples, rows with empty status.
    Returns: {Code: label}
    """
    label_map = {}
    excluded = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get("Code", "").strip()
            actual = row.get("Actual_Ripening_Status", "").strip()
            ripeness = row.get("Ripeness", "").strip()
            disease = row.get("Disease", "0").strip()

            # Skip disease samples
            if actual == "Disease" or disease not in ("0", "1.0", ""):
                excluded.append((code, f"Disease={disease}"))
                continue

            # Skip empty actual status
            if not actual:
                excluded.append((code, "empty Actual_Ripening_Status"))
                continue

            # Map Actual_Ripening_Status to unified label
            status_map = {
                "Unripe": "unripe",
                "Ripe": "ripe",
                "Overripe": "overripe",
            }
            label = status_map.get(actual)
            if label is None:
                # Fallback: use Ripeness field
                ripeness_map = {"Unripe": "unripe", "Ripe": "ripe", "Overripe": "overripe"}
                label = ripeness_map.get(ripeness)

            if label:
                label_map[code] = label
            else:
                excluded.append((code, f"unknown status='{actual}'"))

    if excluded:
        print(f"  [CSV] Excluded {len(excluded)} samples:")
        for code, reason in excluded[:5]:
            print(f"    {code}: {reason}")

    print(f"  [CSV] Loaded {len(label_map)} valid labels from CSV")
    return label_map


def get_zenodo_label_from_filename(filename: str, csv_labels: dict[str, str]) -> Optional[str]:
    """Get label for a Zenodo audio file.
    Zenodo audio files are named after the Code field.
    The ZIP typically extracts to: {Code}.wav or {Code}/{Code}.wav
    """
    stem = Path(filename).stem  # e.g. "IM_CA_UN_1" or "M_CB_RI_92"

    # Try direct CSV lookup first (most accurate)
    if stem in csv_labels:
        return csv_labels[stem]

    # Fallback: parse from filename code pattern
    return get_zenodo_label_from_code(stem)


# ============================================================
# Feature Extraction
# ============================================================

def load_and_segment(audio_path: str) -> list[np.ndarray]:
    """Load audio and split into overlapping fixed-length segments."""
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)

    # Peak normalize
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y = y / max_amp

    segment_len = int(SEGMENT_DURATION * TARGET_SR)
    hop = int(segment_len * (1 - OVERLAP_RATIO))
    segments = []

    if len(y) < segment_len:
        # Pad short files
        padded = np.pad(y, (0, segment_len - len(y)), mode="constant")
        segments.append(padded)
    else:
        for start in range(0, len(y) - segment_len + 1, hop):
            segments.append(y[start : start + segment_len])

    return segments


def extract_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    """Extract log-mel spectrogram → (N_MELS, T) normalized to [0,1]."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=TARGET_SR, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize to [0, 1]
    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max - mel_min > 0:
        mel_db = (mel_db - mel_min) / (mel_max - mel_min)
    return mel_db.astype(np.float32)


def extract_mfcc(y: np.ndarray) -> np.ndarray:
    """Extract delta-MFCC features → (N_MFCC*3, T) stacked with Δ and ΔΔ."""
    mfcc = librosa.feature.mfcc(
        y=y, sr=TARGET_SR, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2])
    # Per-feature normalization
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True) + 1e-8
    return ((features - mean) / std).astype(np.float32)


def extract_features(y: np.ndarray, feature_type: str = "mel") -> np.ndarray:
    if feature_type == "mfcc":
        return extract_mfcc(y)
    elif feature_type == "mel":
        return extract_mel_spectrogram(y)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}. Choose 'mel' or 'mfcc'.")


# ============================================================
# Dataset Processing
# ============================================================

def process_dalvii(output_dir: Path, feature_type: str = "mel") -> tuple[list, list]:
    """Process Dalvii audio dataset."""
    raw_dir = RAW_DIR / "dalvii_audio"
    wav_files = list(raw_dir.glob("**/*.wav"))
    print(f"\n[Dalvii] Found {len(wav_files)} WAV files in {raw_dir}")

    if not wav_files:
        print(f"  [WARN] No files found. Run: python download_dalvii.py")
        return [], []

    features, labels = [], []
    label_counts: dict[str, int] = {}

    for wav_path in wav_files:
        label = get_dalvii_label(wav_path.name)
        if label is None:
            print(f"  [SKIP] {wav_path.name} (no label)")
            continue

        try:
            segments = load_and_segment(str(wav_path))
            for seg in segments:
                feat = extract_features(seg, feature_type)
                features.append(feat)
                labels.append(label)
            label_counts[label] = label_counts.get(label, 0) + len(segments)
        except Exception as e:
            print(f"  [FAIL] {wav_path.name}: {e}")

    print(f"  Extracted {len(features)} segments. Distribution: {label_counts}")
    return features, labels


def process_zenodo(output_dir: Path, feature_type: str = "mel") -> tuple[list, list]:
    """Process Zenodo audio dataset (after unzipping dataset_clean_sound.zip)."""
    raw_dir = RAW_DIR / "zenodo"
    csv_path = raw_dir / "durian_characteristics_cleaned.csv"

    if not csv_path.exists():
        print(f"  [WARN] CSV not found: {csv_path}")
        print("  Run: python download_zenodo.py  (downloads CSV+README)")
        return [], []

    # Load CSV labels
    csv_labels = load_zenodo_csv_labels(csv_path)

    # Find WAV files — Zenodo ZIP extracts to various structures
    # Try common locations
    search_dirs = [
        raw_dir / "dataset_clean_sound",
        raw_dir / "sound",
        raw_dir,
    ]
    wav_files = []
    for d in search_dirs:
        if d.exists():
            wav_files = list(d.glob("**/*.wav"))
            if wav_files:
                print(f"[Zenodo] Found {len(wav_files)} WAV files in {d}")
                break

    if not wav_files:
        print(f"[Zenodo] No WAV files found. Unzip dataset_clean_sound.zip to {raw_dir}")
        print(f"  Command: cd {raw_dir} && unzip dataset_clean_sound.zip")
        return [], []

    features, labels = [], []
    label_counts: dict[str, int] = {}
    skip_count = 0

    for wav_path in wav_files:
        label = get_zenodo_label_from_filename(wav_path.name, csv_labels)
        if label is None:
            skip_count += 1
            continue

        try:
            segments = load_and_segment(str(wav_path))
            for seg in segments:
                feat = extract_features(seg, feature_type)
                features.append(feat)
                labels.append(label)
            label_counts[label] = label_counts.get(label, 0) + len(segments)
        except Exception as e:
            print(f"  [FAIL] {wav_path.name}: {e}")

    print(f"  Extracted {len(features)} segments. Distribution: {label_counts}")
    if skip_count:
        print(f"  Skipped {skip_count} files (no label match)")
    return features, labels


def save_splits(
    features: list,
    labels: list,
    output_dir: Path,
    prefix: str,
    feature_type: str,
):
    """Split into train/val/test and save as .npz files."""
    if not features:
        print(f"  [WARN] No data to save for {prefix}")
        return

    X = np.array(features)[..., np.newaxis]  # Add channel dim: (N, H, W, 1)
    y = np.array(labels)

    class_names = sorted(set(y))
    print(f"\n  Classes: {class_names}")
    print(f"  Total segments: {len(X)}, Shape: {X.shape}")

    # Check class balance
    for cls in class_names:
        count = np.sum(y == cls)
        print(f"    {cls}: {count} ({count/len(y)*100:.1f}%)")

    # Stratified split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"\n  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, Xd, yd in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        np.savez_compressed(
            str(output_dir / f"{prefix}_{feature_type}_{split_name}.npz"),
            X=Xd, y=yd,
        )

    # Save class names
    with open(output_dir / "classes.txt", "w") as f:
        for cls in class_names:
            f.write(cls + "\n")

    # Save metadata
    meta = {
        "prefix": prefix,
        "feature_type": feature_type,
        "total_segments": len(X),
        "input_shape": list(X.shape[1:]),
        "classes": class_names,
        "split": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
    }
    with open(output_dir / f"{prefix}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved to: {output_dir}")


def process_and_save(dataset: str, output_dir: Path, feature_type: str = "mel"):
    """Process one dataset and save splits."""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset} | feature: {feature_type}")
    print(f"{'='*60}")

    if dataset == "dalvii":
        features, labels = process_dalvii(output_dir, feature_type)
        prefix = "dalvii"
    elif dataset == "zenodo":
        features, labels = process_zenodo(output_dir, feature_type)
        prefix = "zenodo"
    else:
        print(f"[ERROR] Unknown dataset: {dataset}")
        return

    save_splits(features, labels, output_dir, prefix, feature_type)


def combine_and_save(datasets: list[str], output_dir: Path, feature_type: str = "mel"):
    """Combine multiple datasets and save as 'combined_*' splits."""
    print(f"\n{'='*60}")
    print(f"Combining datasets: {datasets} | feature: {feature_type}")
    print(f"{'='*60}")

    all_features, all_labels = [], []

    for dataset in datasets:
        if dataset == "dalvii":
            f, l = process_dalvii(output_dir, feature_type)
        elif dataset == "zenodo":
            f, l = process_zenodo(output_dir, feature_type)
        else:
            print(f"[SKIP] Unknown dataset: {dataset}")
            continue
        all_features.extend(f)
        all_labels.extend(l)

    if all_features:
        save_splits(all_features, all_labels, output_dir, "combined", feature_type)
    else:
        print("[WARN] No data collected from any dataset")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare audio features for DurianAI")
    parser.add_argument(
        "--dataset", nargs="+", default=["dalvii"],
        choices=["dalvii", "zenodo"],
        help="Dataset(s) to process. Use multiple to combine.",
    )
    parser.add_argument(
        "--feature", default="mel", choices=["mel", "mfcc"],
        help="Feature type (mel spectrogram or MFCC)",
    )
    parser.add_argument(
        "--output", default=str(PROCESSED_DIR),
        help="Output directory for processed .npz files",
    )
    parser.add_argument(
        "--combine", action="store_true",
        help="Combine multiple datasets into a single 'combined_*' set",
    )

    args = parser.parse_args()

    if not HAS_LIBROSA or not HAS_SKLEARN:
        print("\n[ERROR] Missing dependencies. Install with:")
        print("  pip install librosa soundfile scikit-learn")
        sys.exit(1)

    output_dir = Path(args.output)

    if len(args.dataset) > 1 or args.combine:
        combine_and_save(args.dataset, output_dir, args.feature)
    else:
        process_and_save(args.dataset[0], output_dir, args.feature)


if __name__ == "__main__":
    main()
