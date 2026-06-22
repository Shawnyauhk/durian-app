"""
augment.py — 圖像數據增強
用於擴充榴槤照片數據集
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random

IMG_SIZE = 224


def augment_image(img: Image.Image) -> list[Image.Image]:
    """Generate augmented versions of an image."""
    augmented = [img]  # Include original

    # Horizontal flip
    augmented.append(img.transpose(Image.FLIP_LEFT_RIGHT))

    # Rotation (±15°)
    for angle in [-15, 15]:
        rotated = img.rotate(angle, fillcolor=(0, 0, 0), expand=False)
        augmented.append(rotated)

    # Brightness variation
    for factor in [0.8, 1.2]:
        enhancer = ImageEnhance.Brightness(img)
        augmented.append(enhancer.enhance(factor))

    # Contrast variation
    for factor in [0.8, 1.3]:
        enhancer = ImageEnhance.Contrast(img)
        augmented.append(enhancer.enhance(factor))

    # Color jitter
    enhancer = ImageEnhance.Color(img)
    augmented.append(enhancer.enhance(0.7))  # Desaturate
    augmented.append(enhancer.enhance(1.3))  # Saturate

    # Gaussian blur
    augmented.append(img.filter(ImageFilter.GaussianBlur(radius=1)))

    # Random crop + resize (simulating different distances)
    w, h = img.size
    crop_ratio = random.uniform(0.8, 0.95)
    left = int(w * (1 - crop_ratio) / 2)
    top = int(h * (1 - crop_ratio) / 2)
    cropped = img.crop((left, top, w - left, h - top))
    augmented.append(cropped.resize((w, h), Image.BILINEAR))

    return augmented


def augment_dataset(input_dir: str, output_dir: str, max_per_image: int = 5):
    """Augment images in a directory, preserving class structure."""
    os.makedirs(output_dir, exist_ok=True)

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in Path(input_dir).rglob("*") if f.suffix.lower() in image_exts]
    print(f"Found {len(image_files)} images in {input_dir}")

    for img_path in image_files:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [FAIL] {img_path}: {e}")
            continue

        # Preserve directory structure
        rel_path = img_path.relative_to(input_dir)
        out_subdir = os.path.join(output_dir, str(rel_path.parent))
        os.makedirs(out_subdir, exist_ok=True)

        # Copy original
        import shutil
        shutil.copy2(str(img_path), os.path.join(out_subdir, img_path.name))

        # Generate random augmentations
        all_augs = augment_image(img)
        # Select max_per_image random augmentations (excluding original)
        random_augs = random.sample(all_augs[1:], min(max_per_image, len(all_augs) - 1))

        for i, aug_img in enumerate(random_augs):
            aug_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
            aug_path = os.path.join(out_subdir, aug_name)
            aug_img.save(aug_path)

    total = sum(1 for _ in Path(output_dir).rglob("*") if _.suffix.lower() in image_exts)
    print(f"Output: {total} images (original + augmented) in {output_dir}")


if __name__ == "__main__":
    import sys
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "ml/data/raw/roboflow"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "ml/data/augmented/vision"
    max_per = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    augment_dataset(input_dir, output_dir, max_per)
