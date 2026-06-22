#!/usr/bin/env python3
"""
run_pipeline.py — 一鍵執行完整 ML Pipeline

Usage:
  python ml/run_pipeline.py validate         # 快速環境和API驗證
  python ml/run_pipeline.py download         # 下載所有數據集
  python ml/run_pipeline.py download audio   # 只下載音頻數據
  python ml/run_pipeline.py download vision  # 只下載視覺數據
  python ml/run_pipeline.py prepare          # 預處理數據
  python ml/run_pipeline.py combine          # 合並多數據集
  python ml/run_pipeline.py train            # 訓練模型
  python ml/run_pipeline.py export           # 導出 TFLite + 複製到前端
  python ml/run_pipeline.py all              # 執行全部步驟

Recommended quick start:
  1. python ml/run_pipeline.py validate
  2. python ml/run_pipeline.py download audio  (downloads Dalvii, ~5MB)
  3. python ml/run_pipeline.py prepare
  4. Upload notebooks/03_train_acoustic.ipynb to Google Colab
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
SCRIPTS_DIR = BASE_DIR / "scripts"
AUDIO_DIR = BASE_DIR / "audio"
VISION_DIR = BASE_DIR / "vision"
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
FRONTEND_MODELS = BASE_DIR / ".." / "frontend" / "public" / "models"


def run(cmd: list[str] | str, desc: str) -> bool:
    """Run a command with description. Returns True on success."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    if isinstance(cmd, str):
        print(f"  $ {cmd}")
        result = subprocess.run(cmd, shell=True)
    else:
        print(f"  $ {' '.join(str(c) for c in cmd)}")
        result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ❌ Failed (exit code {result.returncode})")
    else:
        print(f"  ✅ Done")
    return result.returncode == 0


def validate():
    """Quick pipeline validation."""
    run(
        [sys.executable, str(BASE_DIR / "validate_pipeline.py"), "--quick"],
        "Validate environment and API connectivity"
    )


def download(scope: str = "all"):
    """Download datasets."""
    print("\n📥  Downloading Datasets")

    if scope in ("all", "audio"):
        # Dalvii: ~4.5 MB, GitHub, no auth
        run(
            [sys.executable, str(SCRIPTS_DIR / "download_dalvii.py")],
            "Download Dalvii durian knock sounds (92 WAV, ~4.5 MB)"
        )

        # Zenodo CSV + README (tiny, always useful)
        run(
            [sys.executable, str(SCRIPTS_DIR / "download_zenodo.py")],
            "Download Zenodo metadata CSV & README (< 0.1 MB)"
        )

        print(f"""
  📋  OPTIONAL: Download Zenodo knock sounds (~1.5 GB)
      Command: python {SCRIPTS_DIR / 'download_zenodo.py'} sound
      Note: Large file. Use if you have bandwidth.
""")

    if scope in ("all", "vision"):
        # Roboflow vision datasets (needs API key)
        api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        if api_key:
            run(
                [sys.executable, str(SCRIPTS_DIR / "download_roboflow.py")],
                "Download Roboflow durian vision datasets: xtned (1,438) + mutruity (3,000)"
            )
        else:
            print(f"""
  ⚠️  Roboflow API key not set.

  Option 1 — Set env var and re-run:
    set ROBOFLOW_API_KEY=your_key_here
    python ml/run_pipeline.py download vision

  Option 2 — Manual download:
    1. Go to: https://universe.roboflow.com/durian-cnn/durian-ripeness-detection-xtned
    2. Go to: https://universe.roboflow.com/wjy-tis6h/durian_mutruity  ★ NEW
    3. Sign up free → Export → "Folder" format
    4. Unzip to: {DATA_RAW / 'roboflow_xtned'} and {DATA_RAW / 'roboflow_mutruity'}
""")


def prepare(scope: str = "all"):
    """Preprocess datasets into feature arrays."""
    print("\n🔄  Preprocessing Data")

    if scope in ("all", "audio"):
        run(
            [sys.executable, str(SCRIPTS_DIR / "prepare_audio.py"),
             "--dataset", "dalvii", "--feature", "mel",
             "--output", str(DATA_PROCESSED / "audio")],
            "Prepare Dalvii audio → Mel Spectrograms"
        )

        # Process Zenodo audio if the ZIP was extracted
        zenodo_sound_dir = DATA_RAW / "zenodo" / "dataset_clean_sound"
        if zenodo_sound_dir.exists():
            run(
                [sys.executable, str(SCRIPTS_DIR / "prepare_audio.py"),
                 "--dataset", "zenodo", "--feature", "mel",
                 "--output", str(DATA_PROCESSED / "audio")],
                "Prepare Zenodo audio → Mel Spectrograms"
            )

    if scope in ("all", "vision"):
        # Check for any of the vision datasets
        vision_dirs = [
            ("xtned", DATA_RAW / "roboflow_xtned"),
            ("mutruity", DATA_RAW / "roboflow_mutruity"),
        ]
        found_datasets = [name for name, d in vision_dirs if d.exists() and any(d.iterdir())]

        if found_datasets:
            run(
                [sys.executable, str(SCRIPTS_DIR / "prepare_vision.py"),
                 "--dataset"] + found_datasets + ["--output", str(DATA_PROCESSED / "vision")],
                f"Prepare vision images: {', '.join(found_datasets)} → 224×224 arrays"
            )
        # Also check legacy path
        elif (DATA_RAW / "roboflow").exists() and any((DATA_RAW / "roboflow").iterdir()):
            run(
                [sys.executable, str(SCRIPTS_DIR / "prepare_vision.py"),
                 "--dataset", "xtned",
                 "--output", str(DATA_PROCESSED / "vision")],
                "Prepare Roboflow images (legacy path) → 224×224 arrays"
            )
        else:
            print("  ⏭️  Skipping vision prep (no Roboflow data found)")


def combine():
    """Combine multiple datasets with class balancing."""
    print("\n🔀  Combining Datasets")
    run(
        [sys.executable, str(SCRIPTS_DIR / "combine_datasets.py"), "all"],
        "Combine & balance audio + vision datasets"
    )


def train():
    """Train models (runs locally, but Colab recommended)."""
    print("\n🏋️   Training Models")
    print("💡  Recommended: Use Google Colab notebooks for GPU training:")
    print(f"    Audio: {BASE_DIR / 'notebooks' / '03_train_acoustic.ipynb'}")
    print(f"    Vision: {BASE_DIR / 'notebooks' / '04_train_vision.ipynb'}")
    print()

    models_audio = BASE_DIR / "models" / "acoustic"
    models_vision = BASE_DIR / "models" / "vision"

    run(
        [sys.executable, str(AUDIO_DIR / "train_knocknet.py"),
         "--data", str(DATA_PROCESSED / "audio"),
         "--model", "cnn", "--feature", "mel",
         "--epochs", "100", "--batch", "32",
         "--output", str(models_audio)],
        "Train KnockNet-lite (CNN on Mel Spectrogram)"
    )

    run(
        [sys.executable, str(VISION_DIR / "train_cnn.py"),
         "--data", str(DATA_PROCESSED / "vision"),
         "--model", "mobilenet",
         "--epochs", "50", "--fine-tune-epochs", "20",
         "--output", str(models_vision)],
        "Train MobileNetV2 vision model"
    )


def export():
    """Export trained TFLite models to frontend + backend."""
    print("\n📦  Exporting Models")
    FRONTEND_MODELS.mkdir(parents=True, exist_ok=True)

    # Acoustic model
    acoustic_src = BASE_DIR / "models" / "acoustic" / "knocknet_lite.tflite"
    acoustic_labels = BASE_DIR / "models" / "acoustic" / "labels.txt"

    if acoustic_src.exists():
        shutil.copy2(str(acoustic_src), str(FRONTEND_MODELS / "knocknet_lite.tflite"))
        print(f"  ✅ Frontend: knocknet_lite.tflite")
        if acoustic_labels.exists():
            shutil.copy2(str(acoustic_labels), str(FRONTEND_MODELS / "acoustic_labels.txt"))

        # Backend
        backend_dir = BASE_DIR / ".." / "backend" / "models" / "acoustic"
        backend_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(acoustic_src), str(backend_dir / "knocknet_lite.tflite"))
        if acoustic_labels.exists():
            shutil.copy2(str(acoustic_labels), str(backend_dir / "labels.txt"))
        print(f"  ✅ Backend: knocknet_lite.tflite")
    else:
        print(f"  ⚠️  Acoustic model not found: {acoustic_src}")
        print("      Run train step first, or download from Colab.")

    # Vision model
    vision_src = BASE_DIR / "models" / "vision" / "durian_cnn.tflite"
    vision_labels = BASE_DIR / "models" / "vision" / "labels.txt"

    if vision_src.exists():
        shutil.copy2(str(vision_src), str(FRONTEND_MODELS / "durian_cnn.tflite"))
        print(f"  ✅ Frontend: durian_cnn.tflite")
        if vision_labels.exists():
            shutil.copy2(str(vision_labels), str(FRONTEND_MODELS / "vision_labels.txt"))
    else:
        print(f"  ⚠️  Vision model not found: {vision_src}")
        print("      Run train step first, or download from Colab.")

    print(f"\n  Frontend models dir: {FRONTEND_MODELS.resolve()}")
    print("  After copying models, rebuild frontend: cd frontend && npm run build")


def all_steps():
    """Full pipeline."""
    validate()
    download("all")
    prepare("all")
    combine()
    train()
    export()
    print("\n🎉  Complete pipeline finished!")
    print("   Rebuild frontend: cd frontend && npm run build")


COMMANDS = {
    "validate": lambda: validate(),
    "download": lambda: download("all"),
    "download audio": lambda: download("audio"),
    "download vision": lambda: download("vision"),
    "prepare": lambda: prepare("all"),
    "combine": lambda: combine(),
    "train": lambda: train(),
    "export": lambda: export(),
    "all": lambda: all_steps(),
}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    # Support: python run_pipeline.py download audio
    cmd_key = " ".join(sys.argv[1:]).lower()

    # Try exact match first, then prefix match
    handler = COMMANDS.get(cmd_key)
    if handler is None:
        # Try just the first argument
        handler = COMMANDS.get(sys.argv[1].lower())

    if handler is None:
        print(f"Unknown command: {cmd_key}")
        print(f"Available: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    handler()


if __name__ == "__main__":
    main()
