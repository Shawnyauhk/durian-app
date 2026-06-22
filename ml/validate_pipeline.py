#!/usr/bin/env python3
"""
validate_pipeline.py — 端到端管線驗證腳本

使用合成數據（無需下載真實數據集）測試完整 pipeline:
  1. 環境依賴檢查
  2. 目錄結構驗證
  3. Zenodo API 可達性測試
  4. GitHub Dalvii API 可達性測試
  5. CSV 標籤解析測試
  6. 合成音頻 → 特徵提取 → prepare_audio 流程
  7. 合成圖像 → prepare_vision 流程
  8. 合成數據 → train_knocknet (2 epochs 快速測試)
  9. 合成數據 → train_cnn (2 epochs 快速測試)
  10. TFLite 導出測試
  11. 反饋 API 測試 (若後端在線)

用法:
  python ml/validate_pipeline.py           # 完整驗證
  python ml/validate_pipeline.py --quick   # 只做環境和API檢查(不訓練)
  python ml/validate_pipeline.py --train   # 包含快速模型訓練測試
"""
import os
import sys
import json
import time
import shutil
import argparse
import tempfile
import traceback
import urllib.request
import urllib.error
from pathlib import Path

BASE_DIR = Path(__file__).parent
SCRIPTS_DIR = BASE_DIR / "scripts"
AUDIO_DIR_ML = BASE_DIR / "audio"
VISION_DIR_ML = BASE_DIR / "vision"

# Results tracking
results: dict[str, str] = {}  # test_name → "PASS" | "FAIL" | "SKIP" | "WARN"
details: dict[str, str] = {}  # test_name → detail message


def check(name: str, condition: bool, detail: str = "", warn_only: bool = False):
    """Record a test result."""
    status = "PASS" if condition else ("WARN" if warn_only else "FAIL")
    results[name] = status
    details[name] = detail
    icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️", "WARN": "⚠️"}[status]
    print(f"  {icon} [{status}] {name}")
    if detail and not condition:
        print(f"       {detail}")


def skip(name: str, reason: str = ""):
    results[name] = "SKIP"
    details[name] = reason
    print(f"  ⏭️ [SKIP] {name} — {reason}")


# ============================================================
# 1. Environment Check
# ============================================================

def check_environment():
    print("\n[1] Environment & Dependencies")
    print("-" * 40)

    # Python version
    py_ver = sys.version_info
    check("Python version >= 3.10", py_ver >= (3, 10),
          f"Got {py_ver.major}.{py_ver.minor}")

    # Required packages
    packages = {
        "numpy": "numpy",
        "librosa": "librosa",
        "soundfile": "soundfile",
        "sklearn": "sklearn",
        "PIL": "Pillow",
        "tensorflow": "tensorflow",
    }
    for import_name, pkg_name in packages.items():
        try:
            __import__(import_name)
            check(f"Package: {pkg_name}", True)
        except ImportError:
            check(f"Package: {pkg_name}", False,
                  f"Run: pip install {pkg_name}", warn_only=(import_name == "tensorflow"))

    # Check TFLite availability separately
    try:
        import tensorflow as tf
        check("TensorFlow version", True, f"v{tf.__version__}")
        # Check TFLite converter
        converter_ok = hasattr(tf.lite, "TFLiteConverter")
        check("TFLite Converter", converter_ok)
    except Exception as e:
        check("TensorFlow import", False, str(e), warn_only=True)


# ============================================================
# 2. Directory Structure
# ============================================================

def check_directories():
    print("\n[2] Project Directory Structure")
    print("-" * 40)

    expected = [
        BASE_DIR / "scripts" / "download_zenodo.py",
        BASE_DIR / "scripts" / "download_dalvii.py",
        BASE_DIR / "scripts" / "download_roboflow.py",
        BASE_DIR / "scripts" / "prepare_audio.py",
        BASE_DIR / "scripts" / "prepare_vision.py",
        BASE_DIR / "scripts" / "combine_datasets.py",
        BASE_DIR / "audio" / "train_knocknet.py",
        BASE_DIR / "audio" / "augment.py",
        BASE_DIR / "vision" / "train_cnn.py",
        BASE_DIR / "vision" / "augment.py",
        BASE_DIR / "run_pipeline.py",
        BASE_DIR / "TRAINING_PLAN.md",
        BASE_DIR / "notebooks" / "03_train_acoustic.ipynb",
        BASE_DIR / "notebooks" / "04_train_vision.ipynb",
    ]

    for path in expected:
        check(f"File: {path.relative_to(BASE_DIR)}", path.exists())

    # Data dirs
    for d_name in ["data/raw/zenodo", "data/raw/dalvii_audio", "data/raw/roboflow",
                   "data/processed/audio", "data/processed/vision", "models"]:
        d = BASE_DIR / d_name
        d.mkdir(parents=True, exist_ok=True)
        check(f"Dir: {d_name} (created if missing)", True)

    # Check Zenodo CSV (the one small file we downloaded)
    csv_path = BASE_DIR / "data" / "raw" / "zenodo" / "durian_characteristics_cleaned.csv"
    check("Zenodo CSV downloaded", csv_path.exists(),
          f"Run: python scripts/download_zenodo.py", warn_only=True)


# ============================================================
# 3. API Connectivity
# ============================================================

def check_apis():
    print("\n[3] API Connectivity")
    print("-" * 40)

    tests = [
        ("Zenodo API", "https://zenodo.org/api/records/18603796", 15),
        ("GitHub API (Dalvii)", "https://api.github.com/repos/Dalvii/durian-maturity-classification/contents/AUDIO_DATA", 15),
    ]

    for name, url, timeout in tests:
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "DurianAI-Validator/1.0")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                if isinstance(data, list):
                    detail = f"{len(data)} items"
                elif "files" in data:
                    detail = f"{len(data['files'])} files"
                else:
                    detail = "OK"
                check(name, True, detail)
        except urllib.error.URLError as e:
            check(name, False, str(e), warn_only=True)
        except Exception as e:
            check(name, False, str(e), warn_only=True)


# ============================================================
# 4. CSV Label Parsing
# ============================================================

def check_csv_parsing():
    print("\n[4] CSV Label Parsing")
    print("-" * 40)

    csv_path = BASE_DIR / "data" / "raw" / "zenodo" / "durian_characteristics_cleaned.csv"
    if not csv_path.exists():
        skip("Zenodo CSV parsing", "CSV not downloaded")
        return

    try:
        import csv
        from collections import Counter

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        check("CSV loads without error", True, f"{len(rows)} rows")

        # Check expected columns
        expected_cols = ["Maturity_Stage", "Class", "Ripeness", "Code", "Actual_Ripening_Status"]
        has_all = all(col in reader.fieldnames for col in expected_cols)
        check("CSV has expected columns", has_all)

        # Check label distribution
        ripeness_counts = Counter(r["Ripeness"] for r in rows)
        unripe = ripeness_counts.get("Unripe", 0)
        ripe = ripeness_counts.get("Ripe", 0)
        overripe = ripeness_counts.get("Overripe", 0)
        balanced = unripe == ripe == overripe
        check("CSV labels balanced (63/63/63)", balanced,
              f"Got: Unripe={unripe}, Ripe={ripe}, Overripe={overripe}")

        # Test Code parsing
        test_codes = [
            ("IM_CA_UN_1", "unripe"),
            ("M_CB_RI_92", "ripe"),
            ("OM_CC_OR_189", "overripe"),
        ]
        sys.path.insert(0, str(SCRIPTS_DIR))
        from prepare_audio import get_zenodo_label_from_code
        all_pass = True
        for code, expected in test_codes:
            got = get_zenodo_label_from_code(code)
            if got != expected:
                all_pass = False
                print(f"       Code {code}: expected '{expected}', got '{got}'")
        check("Code → label mapping", all_pass)

    except Exception as e:
        check("CSV parsing", False, traceback.format_exc())


# ============================================================
# 5. Synthetic Audio Pipeline Test
# ============================================================

def check_audio_pipeline():
    print("\n[5] Audio Feature Extraction (Synthetic)")
    print("-" * 40)

    try:
        import numpy as np
        import librosa
        import soundfile as sf
    except ImportError:
        skip("Audio pipeline", "librosa/soundfile not installed")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic WAV files (2 classes × 5 files)
        sr = 16000
        for cls, freq in [("75-85%", 300), ("95-Ripe", 600)]:  # Dalvii naming
            cls_dir = tmpdir / "dalvii_audio"
            cls_dir.mkdir(exist_ok=True)
            for i in range(5):
                # Knock sound: short burst + decay
                t = np.linspace(0, 0.5, int(0.5 * sr))
                signal = np.sin(2 * np.pi * freq * t) * np.exp(-t * 10)
                signal = signal.astype(np.float32)
                filename = f"20240929_00000{i}_Dona_{cls}output.wav"
                sf.write(str(cls_dir / filename), signal, sr)

        check("Synthetic WAV creation", True, "10 files (5 × 2 classes)")

        # Test load_and_segment
        sys.path.insert(0, str(SCRIPTS_DIR))
        from prepare_audio import load_and_segment, extract_features, get_dalvii_label
        wav = list((tmpdir / "dalvii_audio").glob("*.wav"))[0]

        segments = load_and_segment(str(wav))
        check("load_and_segment", len(segments) > 0, f"{len(segments)} segments from {wav.name}")

        # Test feature extraction
        feat_mel = extract_features(segments[0], "mel")
        check("extract_features (mel)", feat_mel.shape[0] == 40,
              f"Shape: {feat_mel.shape}")

        feat_mfcc = extract_features(segments[0], "mfcc")
        check("extract_features (mfcc)", feat_mfcc.shape[0] == 120,
              f"Shape: {feat_mfcc.shape} (40 MFCC × 3: original+Δ+ΔΔ)")

        # Test label parsing
        label = get_dalvii_label(wav.name)
        check("get_dalvii_label", label in {"unripe", "ripe"},
              f"Got: '{label}' from '{wav.name}'")

        # Test full process_and_save
        from prepare_audio import process_and_save, RAW_DIR, PROCESSED_DIR as PROC_DIR
        import prepare_audio as pa_module

        # Temporarily redirect raw/processed dirs
        orig_raw = pa_module.RAW_DIR
        orig_proc = pa_module.PROCESSED_DIR
        pa_module.RAW_DIR = tmpdir
        pa_module.PROCESSED_DIR = tmpdir / "processed"

        try:
            process_and_save("dalvii", tmpdir / "processed", "mel")
            npz_files = list((tmpdir / "processed").glob("*.npz"))
            check("prepare_audio full pipeline", len(npz_files) >= 3,
                  f"Generated: {[f.name for f in npz_files]}")
        except Exception as e:
            check("prepare_audio full pipeline", False, str(e))
        finally:
            pa_module.RAW_DIR = orig_raw
            pa_module.PROCESSED_DIR = orig_proc


# ============================================================
# 6. Synthetic Vision Pipeline Test
# ============================================================

def check_vision_pipeline():
    print("\n[6] Vision Feature Extraction (Synthetic)")
    print("-" * 40)

    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        skip("Vision pipeline", "Pillow not installed")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic images (Roboflow folder structure)
        colors = {"Ripe": (255, 200, 0), "Unripe": (0, 180, 50)}  # yellow/green
        for split in ["train", "valid", "test"]:
            for cls, color in colors.items():
                d = tmpdir / "roboflow" / split / cls
                d.mkdir(parents=True, exist_ok=True)
                for i in range(6):
                    # Create a 64×64 solid color image
                    img = Image.new("RGB", (64, 64), color)
                    img.save(str(d / f"durian_{cls}_{i:02d}.jpg"))

        check("Synthetic image creation", True, f"36 images (2 classes × 3 splits × 6 each)")

        # Test prepare_vision
        sys.path.insert(0, str(SCRIPTS_DIR))
        import prepare_vision as pv_module

        orig_raw = pv_module.RAW_DIR
        pv_module.RAW_DIR = tmpdir

        try:
            splits = pv_module.process_roboflow(tmpdir / "processed", size=64)
            check("process_roboflow", "train" in splits,
                  f"Splits: {list(splits.keys())}")

            if "train" in splits:
                X, y = splits["train"]
                check("Image array shape", X.shape[1:] == (64, 64, 3),
                      f"Shape: {X.shape}")
                check("Labels are unified", all(l in {"unripe", "ripe"} for l in y),
                      f"Labels: {set(y)}")
        except Exception as e:
            check("prepare_vision full pipeline", False, str(e))
        finally:
            pv_module.RAW_DIR = orig_raw


# ============================================================
# 7. Quick Model Training Test
# ============================================================

def check_model_training():
    print("\n[7] Quick Model Training Test (2 epochs, synthetic data)")
    print("-" * 40)

    try:
        import tensorflow as tf
        import numpy as np
    except ImportError:
        skip("Model training", "TensorFlow not installed")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ── Acoustic CNN ──
        try:
            sys.path.insert(0, str(AUDIO_DIR_ML))
            from train_knocknet import build_cnn_model, export_tflite
            import tensorflow.keras as keras

            # Tiny synthetic dataset: (N, 40, 32, 1) mel spectrograms
            N = 30
            X = np.random.rand(N, 40, 32, 1).astype(np.float32)
            y = np.array([0, 1, 2] * (N // 3), dtype=np.int32)[:N]

            model = build_cnn_model((40, 32, 1), 3)
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            model.fit(X, y, epochs=2, batch_size=10, verbose=0)
            test_loss, test_acc = model.evaluate(X, y, verbose=0)
            check("Acoustic CNN build & train", True, f"test_acc={test_acc:.2f}")

            # TFLite export
            acoustic_out = tmpdir / "acoustic"
            acoustic_out.mkdir()
            export_tflite(model, str(acoustic_out), ["unripe", "ripe", "overripe"])
            tflite_path = acoustic_out / "knocknet_lite.tflite"
            check("Acoustic TFLite export", tflite_path.exists(),
                  f"Size: {tflite_path.stat().st_size/1024:.1f} KB" if tflite_path.exists() else "Not found")

        except Exception as e:
            check("Acoustic CNN pipeline", False, str(e))

        # ── Vision CNN ──
        try:
            sys.path.insert(0, str(VISION_DIR_ML))
            from train_cnn import build_simple_cnn, export_tflite as export_vision_tflite

            # Tiny synthetic dataset: (N, 64, 64, 3) images
            N = 30
            X = np.random.rand(N, 64, 64, 3).astype(np.float32)
            y = np.array([0, 1, 2] * (N // 3), dtype=np.int32)[:N]

            model = build_simple_cnn(3, img_size=64)
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            model.fit(X, y, epochs=2, batch_size=10, verbose=0)
            test_loss, test_acc = model.evaluate(X, y, verbose=0)
            check("Vision CNN build & train", True, f"test_acc={test_acc:.2f}")

            # TFLite export
            vision_out = tmpdir / "vision"
            vision_out.mkdir()
            export_vision_tflite(model, str(vision_out))
            tflite_path = vision_out / "durian_cnn.tflite"
            check("Vision TFLite export", tflite_path.exists(),
                  f"Size: {tflite_path.stat().st_size/1024:.1f} KB" if tflite_path.exists() else "Not found")

        except Exception as e:
            check("Vision CNN pipeline", False, str(e))


# ============================================================
# 8. Summary Report
# ============================================================

def print_summary():
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "WARN": 0}
    for name, status in sorted(results.items()):
        icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️", "WARN": "⚠️"}[status]
        counts[status] += 1

    print(f"\n  ✅ PASS: {counts['PASS']}")
    print(f"  ❌ FAIL: {counts['FAIL']}")
    print(f"  ⚠️  WARN: {counts['WARN']}")
    print(f"  ⏭️  SKIP: {counts['SKIP']}")
    print()

    if counts["FAIL"] > 0:
        print("Failed tests:")
        for name, status in results.items():
            if status == "FAIL":
                print(f"  ❌ {name}: {details.get(name, '')}")

    if counts["FAIL"] == 0 and counts["WARN"] == 0:
        print("🎉 All tests passed! Pipeline is ready.")
    elif counts["FAIL"] == 0:
        print("✅ No critical failures. Some warnings need attention.")
    else:
        print("❌ Some tests failed. Fix issues before running the full pipeline.")

    return counts["FAIL"] == 0


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Validate DurianAI ML pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: only env + API checks (no synthetic training)")
    parser.add_argument("--train", action="store_true",
                        help="Include quick model training test (requires TensorFlow)")
    args = parser.parse_args()

    print("DurianAI ML Pipeline Validator")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print(f"Base dir: {BASE_DIR}")

    check_environment()
    check_directories()
    check_apis()
    check_csv_parsing()

    if not args.quick:
        check_audio_pipeline()
        check_vision_pipeline()

    if args.train:
        check_model_training()
    elif not args.quick:
        skip("Model training", "Use --train flag to enable quick training test")

    ok = print_summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
