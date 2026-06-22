"""
download_roboflow.py — 下載 Roboflow 榴槤成熟度照片數據集
1,438 張圖片，3 類 (Ripe/Unripe/Defect)
授權: CC BY
"""
import os
import sys
import urllib.request
import zipfile

# Roboflow dataset URL (export as folder structure)
# Sign up at https://roboflow.com and get your API key
# Then export this dataset: https://universe.roboflow.com/durian-cnn/durian-ripeness-detection-xtned

ROBOFLOW_DATASET_URL = "https://universe.roboflow.com/durian-cnn/durian-ripeness-detection-xtned"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "roboflow")


def download_with_api(api_key: str, workspace: str = "durian-cnn", project: str = "durian-ripeness-detection-xtned"):
    """Download dataset using Roboflow API."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow...")
        os.system(f"{sys.executable} -m pip install roboflow")
        from roboflow import Roboflow

    os.makedirs(DATA_DIR, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    version = project_obj.versions()[0]
    dataset = version.download("folder", location=DATA_DIR)

    print(f"Dataset downloaded to: {DATA_DIR}")
    print(f"Classes: {version.classes}")


def download_manual():
    """Print instructions for manual download."""
    print("=" * 60)
    print("Manual Download Instructions for Roboflow Dataset")
    print("=" * 60)
    print(f"""
1. Go to: {ROBOFLOW_DATASET_URL}
2. Sign up / Log in to Roboflow (free)
3. Click "Download Dataset"
4. Select format: "Folder" (for classification)
5. Choose "show download code" or download ZIP
6. Extract to: {os.path.abspath(DATA_DIR)}

Expected structure after extraction:
  {DATA_DIR}/
  ├── train/
  │   ├── Ripe/
  │   ├── Unripe/
  │   └── Defect/
  ├── valid/
  │   ├── Ripe/
  │   ├── Unripe/
  │   └── Defect/
  └── test/
      ├── Ripe/
      ├── Unripe/
      └── Defect/

Tip: You can also use the Roboflow Python API:
  pip install roboflow
  python -c "
    from roboflow import Roboflow
    rf = Roboflow(api_key='YOUR_KEY')
    project = rf.workspace('durian-cnn').project('durian-ripeness-detection-xtned')
    project.versions()[0].download('folder')
  "
""")


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if api_key:
        download_with_api(api_key)
    else:
        print("No ROBOFLOW_API_KEY environment variable found.")
        download_manual()


if __name__ == "__main__":
    main()
