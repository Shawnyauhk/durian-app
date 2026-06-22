"""
prepare_vision.py — 圖像預處理管線
統一 224x224 → 歸一化 → 標籤映射 → 分割訓練/驗證/測試集

支持數據集:
  roboflow  : 1,438 張, 3類 (Ripe/Unripe/Defect)
  zenodo    : 189 樣本 RGB 圖像 (解壓後), 3類
  rom1420   : GitHub 4類圖像 (Ripe1/Ripe2/Ripe3/Ripe4)

統一 3-class 標籤: unripe / ripe / overripe
Defect 圖像: Phase 1 排除, Phase 2 加入為第4類

用法:
  python prepare_vision.py                           # 處理 roboflow (默認)
  python prepare_vision.py --dataset zenodo          # 處理 zenodo RGB
  python prepare_vision.py --dataset roboflow zenodo # 合並
  python prepare_vision.py --size 160                # 縮小圖像 (更快訓練)
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

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARN] Pillow not installed. Run: pip install Pillow")

try:
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "vision"

IMG_SIZE = 224
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ============================================================
# Label Mapping
# ============================================================

# Roboflow: folder names → unified labels
ROBOFLOW_LABEL_MAP = {
    "Ripe": "ripe",
    "ripe": "ripe",
    "Unripe": "unripe",
    "unripe": "unripe",
    # "Defect": None,  # Excluded in Phase 1; add in Phase 2
}

# Rom1420: folder names → unified labels
ROM1420_LABEL_MAP = {
    "Ripe1": "unripe",    # Least mature
    "Ripe2": "ripe",
    "Ripe3": "overripe",
    "Ripe4": "overripe",  # Most over-ripe; merge with Ripe3
}

# Zenodo RGB: folder names based on Code prefix
ZENODO_RIPENESS_MAP = {
    "UN": "unripe",
    "RI": "ripe",
    "OR": "overripe",
}


def get_zenodo_label_from_code(code: str) -> Optional[str]:
    """Parse label from Zenodo Code. Format: {M}_{C}_{R}_{N} e.g. IM_CA_UN_1"""
    parts = code.split("_")
    if len(parts) >= 3:
        return ZENODO_RIPENESS_MAP.get(parts[2])
    return None


def load_zenodo_csv_labels(csv_path: Path) -> dict[str, str]:
    """Load Zenodo CSV → {Code: label} mapping."""
    label_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get("Code", "").strip()
            actual = row.get("Actual_Ripening_Status", "").strip()
            disease = row.get("Disease", "0").strip()
            if actual == "Disease" or not actual or code == "":
                continue
            status_map = {"Unripe": "unripe", "Ripe": "ripe", "Overripe": "overripe"}
            label = status_map.get(actual)
            if label:
                label_map[code] = label
    return label_map


# ============================================================
# Image Loading
# ============================================================

def load_image(image_path: str, size: int = IMG_SIZE) -> Optional[np.ndarray]:
    """Load, resize, and normalize image to float32 [0,1]."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((size, size), Image.BILINEAR)
        return (np.array(img, dtype=np.float32) / 255.0)
    except Exception as e:
        print(f"  [FAIL] {image_path}: {e}")
        return None


def is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMG_EXTENSIONS


# ============================================================
# Dataset Processing
# ============================================================

def process_roboflow(output_dir: Path, size: int = IMG_SIZE) -> dict[str, tuple]:
    """Process Roboflow dataset (folder structure: split/class/images).
    Returns: {'train': (X, y), 'val': (X, y), 'test': (X, y)}
    """
    raw_dir = RAW_DIR / "roboflow"
    if not raw_dir.exists():
        print(f"[Roboflow] Directory not found: {raw_dir}")
        print("  Run: python download_roboflow.py")
        return {}

    splits = {}
    for split_name in ["train", "valid", "val", "test"]:
        split_dir = raw_dir / split_name
        if not split_dir.exists():
            continue

        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        if not class_dirs:
            continue

        X, y = [], []
        for cls_dir in class_dirs:
            cls_name = cls_dir.name
            unified_label = ROBOFLOW_LABEL_MAP.get(cls_name)
            if unified_label is None:
                print(f"  [SKIP] {split_name}/{cls_name} (Defect excluded in Phase 1)")
                continue

            images = [f for f in cls_dir.iterdir() if is_image_file(str(f))]
            print(f"  {split_name}/{cls_name} → {unified_label}: {len(images)} images")

            for img_path in images:
                arr = load_image(str(img_path), size)
                if arr is not None:
                    X.append(arr)
                    y.append(unified_label)

        out_key = "val" if split_name == "valid" else split_name
        if X:
            splits[out_key] = (np.array(X), np.array(y))

    return splits


def process_zenodo_rgb(output_dir: Path, size: int = IMG_SIZE) -> dict[str, tuple]:
    """Process Zenodo RGB dataset.
    After unzipping, files may be in: zenodo/dataset_clean_rgb/{Code}/RGB_{Code}.jpg
    Returns: {'train': (X, y), 'val': (X, y), 'test': (X, y)} after splitting
    """
    raw_dir = RAW_DIR / "zenodo"
    csv_path = raw_dir / "durian_characteristics_cleaned.csv"

    if not csv_path.exists():
        print(f"[Zenodo] CSV not found: {csv_path}")
        return {}

    csv_labels = load_zenodo_csv_labels(csv_path)

    # Search for RGB images in various locations
    rgb_dirs = [
        raw_dir / "dataset_clean_rgb",
        raw_dir / "rgb",
        raw_dir,
    ]
    image_files = []
    for d in rgb_dirs:
        if d.exists():
            found = [f for f in d.rglob("*") if is_image_file(str(f))]
            if found:
                image_files = found
                print(f"[Zenodo RGB] Found {len(image_files)} images in {d}")
                break

    if not image_files:
        print(f"[Zenodo RGB] No images found. Unzip dataset_clean_rgb.zip to {raw_dir}")
        print(f"  Note: File is ~19 GB. Consider downloading selectively.")
        return {}

    X, y = [], []
    label_counts: dict[str, int] = {}
    skip_count = 0

    for img_path in image_files:
        # Try to match Code from filename or parent directory
        stem = img_path.stem
        parent = img_path.parent.name

        # Try patterns: RGB_{Code}.jpg, {Code}.jpg, parent folder = Code
        code = None
        for candidate in [stem, parent,
                          stem.replace("RGB_", ""),
                          stem.split("_RGB")[0]]:
            if candidate in csv_labels:
                code = candidate
                break

        if code:
            label = csv_labels[code]
        else:
            # Fallback: parse Code from filename
            label = get_zenodo_label_from_code(stem)

        if label is None:
            skip_count += 1
            continue

        arr = load_image(str(img_path), size)
        if arr is not None:
            X.append(arr)
            y.append(label)
            label_counts[label] = label_counts.get(label, 0) + 1

    print(f"  Collected {len(X)} images. Distribution: {label_counts}")
    if skip_count:
        print(f"  Skipped {skip_count} images (no label)")

    if not X:
        return {}

    # Split for standalone zenodo dataset
    from sklearn.model_selection import train_test_split
    X_arr = np.array(X)
    y_arr = np.array(y)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_arr, y_arr, test_size=0.30, random_state=42, stratify=y_arr
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


# ============================================================
# Save Utilities
# ============================================================

def save_splits(
    splits: dict[str, tuple],
    output_dir: Path,
    prefix: str,
    size: int,
):
    """Save splits as .npz + metadata."""
    if not splits:
        print(f"  [WARN] No splits to save for {prefix}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    all_classes = set()
    meta = {"prefix": prefix, "img_size": size, "splits": {}}

    for split_name, (X, y) in splits.items():
        np.savez_compressed(str(output_dir / f"vision_{split_name}.npz"), X=X, y=y)
        all_classes.update(y.tolist())
        meta["splits"][split_name] = {"n": len(X), "shape": list(X.shape[1:])}
        print(f"  Saved {split_name}: {X.shape}  classes={sorted(set(y))}")

    class_names = sorted(all_classes)
    with open(output_dir / "classes.txt", "w") as f:
        for cls in class_names:
            f.write(cls + "\n")

    meta["classes"] = class_names
    with open(output_dir / f"{prefix}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved to: {output_dir}")


def combine_splits(splits_list: list[dict[str, tuple]]) -> dict[str, tuple]:
    """Combine multiple split dicts into one."""
    combined: dict[str, tuple] = {}
    for splits in splits_list:
        for split_name, (X, y) in splits.items():
            if split_name in combined:
                X_prev, y_prev = combined[split_name]
                combined[split_name] = (
                    np.concatenate([X_prev, X], axis=0),
                    np.concatenate([y_prev, y], axis=0),
                )
            else:
                combined[split_name] = (X, y)
    return combined


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare vision data for DurianAI")
    parser.add_argument(
        "--dataset", nargs="+", default=["roboflow"],
        choices=["roboflow", "zenodo"],
        help="Dataset(s) to process",
    )
    parser.add_argument("--size", type=int, default=IMG_SIZE, help="Image size (default 224)")
    parser.add_argument("--output", default=str(PROCESSED_DIR), help="Output directory")

    args = parser.parse_args()

    if not HAS_PIL or not HAS_SKLEARN:
        print("\n[ERROR] Missing dependencies. Install with:")
        print("  pip install Pillow scikit-learn")
        sys.exit(1)

    output_dir = Path(args.output)
    splits_list = []

    for dataset in args.dataset:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset} | size: {args.size}x{args.size}")
        print(f"{'='*60}")

        if dataset == "roboflow":
            splits = process_roboflow(output_dir, args.size)
        elif dataset == "zenodo":
            splits = process_zenodo_rgb(output_dir, args.size)
        else:
            print(f"[ERROR] Unknown dataset: {dataset}")
            continue

        if splits:
            splits_list.append(splits)

    if not splits_list:
        print("\n[ERROR] No data collected from any dataset.")
        sys.exit(1)

    if len(splits_list) == 1:
        prefix = args.dataset[0]
        final_splits = splits_list[0]
    else:
        prefix = "combined"
        final_splits = combine_splits(splits_list)

    save_splits(final_splits, output_dir, prefix, args.size)
    print(f"\nDone! Processed {sum(len(X) for X, _ in final_splits.values())} total images.")


if __name__ == "__main__":
    main()
