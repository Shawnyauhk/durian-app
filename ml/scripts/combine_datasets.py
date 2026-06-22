"""
combine_datasets.py — 合並多個數據集為統一訓練集

功能:
  1. 合並音頻數據集 (Dalvii + Zenodo)
  2. 合並視覺數據集 (Roboflow + Zenodo RGB)
  3. 類別不均衡處理 (Oversampling)
  4. 生成數據集統計報告

用法:
  python combine_datasets.py audio           # 合並音頻數據集
  python combine_datasets.py vision          # 合並視覺數據集
  python combine_datasets.py all             # 合並全部

輸出:
  data/processed/audio/combined_mel_train.npz
  data/processed/audio/combined_mel_val.npz
  data/processed/audio/combined_mel_test.npz
  data/processed/vision/combined_train.npz
  data/processed/vision/combined_val.npz
  data/processed/vision/combined_test.npz
"""
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from sklearn.utils import resample
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] sklearn not found. Run: pip install scikit-learn")

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
AUDIO_DIR = PROCESSED_DIR / "audio"
VISION_DIR = PROCESSED_DIR / "vision"

VALID_LABELS = ["unripe", "ripe", "overripe"]


# ============================================================
# NPZ Loading Utilities
# ============================================================

def load_npz(path: Path) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Load .npz file → (X, y) or None if not found."""
    if not path.exists():
        print(f"  [MISS] {path}")
        return None
    data = np.load(str(path), allow_pickle=True)
    return data["X"], data["y"]


def list_available_npz(directory: Path, pattern: str = "*_train.npz") -> list[Path]:
    """List all matching .npz files in directory."""
    return sorted(directory.glob(pattern))


# ============================================================
# Class Balancing
# ============================================================

def oversample_minority(
    X: np.ndarray, y: np.ndarray, strategy: str = "max"
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes to match the majority.
    
    strategy:
      'max'  : upsample all classes to majority class size
      'mean' : upsample all classes to mean class size
    """
    if not HAS_SKLEARN:
        print("[WARN] sklearn not available, skipping oversampling")
        return X, y

    class_names, counts = np.unique(y, return_counts=True)
    if strategy == "max":
        target_n = int(np.max(counts))
    elif strategy == "mean":
        target_n = int(np.mean(counts))
    else:
        target_n = int(np.max(counts))

    print(f"\n  Class distribution before balancing:")
    for cls, cnt in zip(class_names, counts):
        print(f"    {cls}: {cnt}")
    print(f"  Target samples per class: {target_n}")

    X_balanced, y_balanced = [], []
    for cls in class_names:
        mask = y == cls
        X_cls = X[mask]
        y_cls = y[mask]
        n = len(X_cls)

        if n < target_n:
            # Oversample with replacement
            X_resampled, y_resampled = resample(
                X_cls, y_cls,
                replace=True,
                n_samples=target_n,
                random_state=42,
            )
            X_balanced.append(X_resampled)
            y_balanced.append(y_resampled)
            print(f"    {cls}: {n} → {target_n} (oversampled +{target_n-n})")
        else:
            X_balanced.append(X_cls)
            y_balanced.append(y_cls)
            print(f"    {cls}: {n} (unchanged)")

    X_out = np.concatenate(X_balanced, axis=0)
    y_out = np.concatenate(y_balanced, axis=0)

    # Shuffle
    idx = np.random.permutation(len(X_out))
    return X_out[idx], y_out[idx]


# ============================================================
# Audio Combining
# ============================================================

def combine_audio(feature_type: str = "mel", balance: bool = True):
    """Combine all available audio datasets."""
    print(f"\n{'='*60}")
    print(f"Combining audio datasets (feature={feature_type})")
    print(f"{'='*60}")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Find all available source datasets
    available = []
    for prefix in ["dalvii", "zenodo"]:
        train_path = AUDIO_DIR / f"{prefix}_{feature_type}_train.npz"
        if train_path.exists():
            available.append(prefix)
            print(f"  Found: {prefix}")
        else:
            print(f"  Missing: {prefix} (run prepare_audio.py --dataset {prefix} first)")

    if not available:
        print("\n[ERROR] No audio datasets available to combine.")
        print("Run: python prepare_audio.py --dataset dalvii (or zenodo)")
        return

    if len(available) == 1:
        print(f"\n[INFO] Only 1 dataset available: {available[0]}")
        print("Copying as 'combined'...")
        for split in ["train", "val", "test"]:
            src = AUDIO_DIR / f"{available[0]}_{feature_type}_{split}.npz"
            dst = AUDIO_DIR / f"combined_{feature_type}_{split}.npz"
            import shutil
            if src.exists():
                shutil.copy2(str(src), str(dst))
                print(f"  Copied: {dst.name}")
        return

    # Combine all splits
    combined: dict[str, tuple] = {}
    for split in ["train", "val", "test"]:
        X_all, y_all = [], []
        for prefix in available:
            result = load_npz(AUDIO_DIR / f"{prefix}_{feature_type}_{split}.npz")
            if result:
                X, y = result
                X_all.append(X)
                y_all.append(y)
                print(f"  Loaded {prefix}/{split}: {X.shape}")

        if X_all:
            X_combined = np.concatenate(X_all, axis=0)
            y_combined = np.concatenate(y_all, axis=0)
            combined[split] = (X_combined, y_combined)

    # Apply oversampling to train set
    if balance and "train" in combined:
        print(f"\nApplying oversampling to train set...")
        X_train, y_train = combined["train"]
        combined["train"] = oversample_minority(X_train, y_train, strategy="max")

    # Save
    all_classes = set()
    stats = {}
    for split, (X, y) in combined.items():
        out_path = AUDIO_DIR / f"combined_{feature_type}_{split}.npz"
        np.savez_compressed(str(out_path), X=X, y=y)
        classes, counts = np.unique(y, return_counts=True)
        all_classes.update(classes.tolist())
        stats[split] = {"n": len(X), "shape": list(X.shape[1:]),
                        "classes": dict(zip(classes.tolist(), counts.tolist()))}
        print(f"  Saved combined/{split}: {X.shape}")

    # Save classes.txt
    with open(AUDIO_DIR / "classes.txt", "w") as f:
        for cls in sorted(all_classes):
            f.write(cls + "\n")

    # Save stats
    meta = {
        "sources": available,
        "feature_type": feature_type,
        "balanced": balance,
        "splits": stats,
    }
    with open(AUDIO_DIR / "combined_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Combined audio saved to: {AUDIO_DIR}")
    print(f"  Sources: {available}")


# ============================================================
# Vision Combining
# ============================================================

def combine_vision(balance: bool = True):
    """Combine all available vision datasets."""
    print(f"\n{'='*60}")
    print("Combining vision datasets")
    print(f"{'='*60}")

    VISION_DIR.mkdir(parents=True, exist_ok=True)

    # Check available splits (vision uses generic names without prefix)
    available_splits = {}
    for split in ["train", "val", "test"]:
        path = VISION_DIR / f"vision_{split}.npz"
        if path.exists():
            result = load_npz(path)
            if result:
                X, y = result
                available_splits[split] = (X, y)
                print(f"  Found vision/{split}: {X.shape}")

    # Check for prefix-based splits (if multiple datasets processed separately)
    for prefix in ["roboflow", "roboflow_xtned", "mutruity", "zenodo", "xtned"]:
        for split in ["train", "val", "test"]:
            path = VISION_DIR / f"{prefix}_{split}.npz"
            if path.exists():
                result = load_npz(path)
                if result:
                    X, y = result
                    if split in available_splits:
                        X_prev, y_prev = available_splits[split]
                        available_splits[split] = (
                            np.concatenate([X_prev, X], axis=0),
                            np.concatenate([y_prev, y], axis=0),
                        )
                        print(f"  Merged {prefix}/{split}: +{len(X)}")
                    else:
                        available_splits[split] = (X, y)
                        print(f"  Found {prefix}/{split}: {X.shape}")

    if not available_splits:
        print("\n[ERROR] No vision data found.")
        print("Run: python prepare_vision.py --dataset roboflow")
        return

    # Apply oversampling to train
    if balance and "train" in available_splits:
        print(f"\nApplying oversampling to train set...")
        X_train, y_train = available_splits["train"]
        available_splits["train"] = oversample_minority(X_train, y_train)

    # Save combined splits
    all_classes = set()
    for split, (X, y) in available_splits.items():
        out_path = VISION_DIR / f"combined_{split}.npz"
        np.savez_compressed(str(out_path), X=X, y=y)
        all_classes.update(y.tolist())
        print(f"  Saved combined/{split}: {X.shape}")

    with open(VISION_DIR / "combined_classes.txt", "w") as f:
        for cls in sorted(all_classes):
            f.write(cls + "\n")

    print(f"\n  Combined vision saved to: {VISION_DIR}")


# ============================================================
# Statistics Report
# ============================================================

def print_dataset_report():
    """Print a summary of all available datasets."""
    print(f"\n{'='*60}")
    print("Dataset Report")
    print(f"{'='*60}")

    for modality, data_dir in [("Audio", AUDIO_DIR), ("Vision", VISION_DIR)]:
        print(f"\n{modality} ({data_dir}):")
        npz_files = sorted(data_dir.glob("*.npz")) if data_dir.exists() else []
        if not npz_files:
            print("  (none)")
            continue
        for npz in npz_files:
            try:
                data = np.load(str(npz))
                X, y = data["X"], data["y"]
                classes, counts = np.unique(y, return_counts=True)
                class_str = " | ".join(f"{c}:{n}" for c, n in zip(classes, counts))
                print(f"  {npz.name:40s} shape={X.shape}  [{class_str}]")
            except Exception as e:
                print(f"  {npz.name}: Error reading ({e})")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Combine DurianAI datasets")
    parser.add_argument(
        "modality", nargs="?", default="all",
        choices=["audio", "vision", "all", "report"],
        help="Which modality to combine (default: all)",
    )
    parser.add_argument(
        "--feature", default="mel", choices=["mel", "mfcc"],
        help="Audio feature type",
    )
    parser.add_argument(
        "--no-balance", action="store_true",
        help="Skip oversampling (use raw imbalanced data)",
    )

    args = parser.parse_args()
    balance = not args.no_balance

    if args.modality == "report":
        print_dataset_report()
    elif args.modality == "audio":
        combine_audio(args.feature, balance)
    elif args.modality == "vision":
        combine_vision(balance)
    elif args.modality == "all":
        combine_audio(args.feature, balance)
        combine_vision(balance)
        print_dataset_report()


if __name__ == "__main__":
    main()
