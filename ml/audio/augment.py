"""
augment.py — 音頻數據增強
用於擴充榴槤敲擊音數據集，提升模型泛化能力
"""
import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

TARGET_SR = 16000


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Time stretch audio by rate (0.8=slower, 1.2=faster)."""
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Pitch shift audio by n_steps semitones."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def add_noise(y: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Add Gaussian noise."""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


def add_background_noise(y: np.ndarray, bg_path: str, snr_db: float = 20) -> np.ndarray:
    """Mix with background noise at specified SNR."""
    bg, _ = librosa.load(bg_path, sr=TARGET_SR, mono=True)
    if len(bg) < len(y):
        bg = np.pad(bg, (0, len(y) - len(bg)))
    else:
        bg = bg[:len(y)]

    y_power = np.mean(y ** 2)
    bg_power = np.mean(bg ** 2)
    if bg_power > 0:
        bg = bg * np.sqrt(y_power / bg_power) * (10 ** (-snr_db / 20))

    return y + bg


def shift_time(y: np.ndarray, shift_max: float = 0.3) -> np.ndarray:
    """Random time shift."""
    shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)


def change_volume(y: np.ndarray, gain_range: tuple = (0.8, 1.2)) -> np.ndarray:
    """Random volume change."""
    gain = np.random.uniform(*gain_range)
    return y * gain


def augment_audio_file(
    input_path: str,
    output_dir: str,
    n_augmentations: int = 5,
):
    """Generate augmented versions of an audio file."""
    y, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)
    basename = Path(input_path).stem

    augmentations = [
        lambda: time_stretch(y, np.random.uniform(0.8, 1.2)),
        lambda: pitch_shift(y, sr, np.random.uniform(-2, 2)),
        lambda: add_noise(y, np.random.uniform(0.002, 0.01)),
        lambda: shift_time(y, np.random.uniform(0.1, 0.3)),
        lambda: change_volume(y),
    ]

    for i in range(n_augmentations):
        # Randomly select 1-3 augmentations to combine
        n_ops = np.random.randint(1, 4)
        ops = np.random.choice(len(augmentations), size=n_ops, replace=False)

        augmented = y.copy()
        for op_idx in sorted(ops, reverse=True):
            augmented = augmentations[op_idx]()

        # Clip to valid range
        augmented = np.clip(augmented, -1.0, 1.0)

        out_path = os.path.join(output_dir, f"{basename}_aug{i}.wav")
        sf.write(out_path, augmented, TARGET_SR)


def augment_dataset(input_dir: str, output_dir: str, n_augmentations: int = 5):
    """Augment all WAV files in a directory."""
    os.makedirs(output_dir, exist_ok=True)

    wav_files = list(Path(input_dir).rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files in {input_dir}")

    # Preserve subdirectory structure
    for wav_path in wav_files:
        rel_dir = wav_path.parent.relative_to(input_dir)
        out_subdir = os.path.join(output_dir, str(rel_dir))
        os.makedirs(out_subdir, exist_ok=True)

        # Copy original
        import shutil
        shutil.copy2(str(wav_path), out_subdir)

        # Generate augmentations
        try:
            augment_audio_file(str(wav_path), out_subdir, n_augmentations)
        except Exception as e:
            print(f"  [FAIL] {wav_path}: {e}")

    total = len(list(Path(output_dir).rglob("*.wav")))
    print(f"Output: {total} WAV files (original + augmented) in {output_dir}")


if __name__ == "__main__":
    import sys
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "ml/data/raw/dalvii_audio"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "ml/data/augmented/audio"
    n_aug = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    augment_dataset(input_dir, output_dir, n_aug)
