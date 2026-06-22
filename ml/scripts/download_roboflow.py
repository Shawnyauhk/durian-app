"""
download_roboflow.py — 下載 Roboflow 榴槤數據集

支持多個 Roboflow 開源數據集：
  1. durian-ripeness-detection-xtned  — 1,438 張，3 類 (Ripe/Unripe/Defect)
     https://universe.roboflow.com/durian-cnn/durian-ripeness-detection-xtned
  2. durian_mutruity                  — 3,000 張，3 類 (defective/immature/mature) ★ 新增！
     https://universe.roboflow.com/wjy-tis6h/durian_mutruity
     CC BY 4.0 授權

用法:
  export ROBOFLOW_API_KEY="your_key"
  python download_roboflow.py                         # 下載所有數據集
  python download_roboflow.py --dataset xtned          # 只下載原始數據集
  python download_roboflow.py --dataset mutruity       # 只下載新數據集
"""
import os
import sys
import argparse

# 支持的 Roboflow 數據集
DATASETS = {
    "xtned": {
        "workspace": "durian-cnn",
        "project": "durian-ripeness-detection-xtned",
        "description": "Original: 1,438 images, 3 classes (Ripe/Unripe/Defect)",
        "subdir": "roboflow_xtned",
        "type": "classification",
    },
    "mutruity": {
        "workspace": "wjy-tis6h",
        "project": "durian_mutruity",
        "description": "NEW: 3,000 images, 3 classes (defective/immature/mature) ★",
        "subdir": "roboflow_mutruity",
        "type": "object-detection",
    },
}

BASE_DIR = os.path.dirname(__file__)
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "raw")


def download_dataset(api_key: str, dataset_name: str):
    """Download a single Roboflow dataset."""
    info = DATASETS[dataset_name]
    data_dir = os.path.join(DEFAULT_DATA_DIR, info["subdir"])
    os.makedirs(data_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Downloading: {info['project']}")
    print(f"  {info['description']}")
    print(f"  → {os.path.abspath(data_dir)}")
    print(f"{'=' * 60}")

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow package...")
        os.system(f"{sys.executable} -m pip install -q roboflow")
        from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(info["workspace"]).project(info["project"])
    version = project_obj.versions()[0]

    print(f"Version: {version.version if hasattr(version, 'version') else 'latest'}")

    # Try to get class names — API version differences
    try:
        classes = version.classes if hasattr(version, 'classes') else project_obj.classes
        print(f"Classes: {classes}")
    except Exception:
        pass

    try:
        n_images = len(version.images) if hasattr(version, 'images') else 'N/A'
        print(f"Images:  {n_images}")
    except Exception:
        pass

    # Determine format based on project type
    # Classification → "folder" (class dirs), Object Detection → "yolov8" (txt annotations)
    is_classification = info.get("type") == "classification"
    fmt = "folder" if is_classification else "multiclass"
    
    try:
        dataset = version.download(fmt, location=data_dir)
    except Exception as e:
        if "invalid format" in str(e) or "multiclass" in str(e).lower():
            print(f"  Format 'multiclass' not supported, trying 'yolov8'...")
            dataset = version.download("yolov8", location=data_dir)
        else:
            raise e
    
    print(f"✅ Download complete: {info['project']}")
    return dataset


def print_manual_instructions():
    """Print manual download instructions for all datasets."""
    print("=" * 60)
    print("Manual Download Instructions")
    print("=" * 60)

    for name, info in DATASETS.items():
        url = f"https://universe.roboflow.com/{info['workspace']}/{info['project']}"
        data_dir = os.path.join(DEFAULT_DATA_DIR, info["subdir"])
        print(f"""
--- {info['project']} ---
  URL: {url}
  Extract to: {os.path.abspath(data_dir)}
  {info['description']}
""")

    print("""
Or set ROBOFLOW_API_KEY and let this script download automatically:
  export ROBOFLOW_API_KEY="your_key"
  python download_roboflow.py
""")


def main():
    parser = argparse.ArgumentParser(description="Download Roboflow durian datasets")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        print("⚠️ No ROBOFLOW_API_KEY environment variable found.")
        print_manual_instructions()
        sys.exit(0)

    if args.dataset == "all":
        for name in DATASETS:
            download_dataset(api_key, name)
        print(f"\n{'=' * 60}")
        print("✅ All datasets downloaded successfully!")
        print(f"{'=' * 60}")
    else:
        download_dataset(api_key, args.dataset)


if __name__ == "__main__":
    main()
