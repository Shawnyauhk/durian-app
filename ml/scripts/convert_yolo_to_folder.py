"""
convert_yolo_to_folder.py — Convert YOLOv8 format to classification folder structure

YOLOv8 dataset: split/{images,labels}/ files
Folder structure: split/class_name/short_name.jpg

Label mapping: 0=defective(excluded), 1=mature→ripe, 2=immature→unripe
"""
import os
import shutil

SRC = r"c:\tmp\mutruity"
DST = r"C:\Users\Shawn\WorkBuddy\2026-06-11-12-46-13\durian-app\ml\data\raw\roboflow_mutruity"
VALID_CLASSES = {1: "mature", 2: "immature"}  # class_id → original class name

def convert_split(split_name):
    """Convert one split (train/valid/test) to folder structure."""
    images_dir = os.path.join(SRC, split_name, "images")
    labels_dir = os.path.join(SRC, split_name, "labels")
    
    if not os.path.isdir(images_dir):
        print(f"  [SKIP] {split_name} not found")
        return
    
    # Count images per class
    class_counts = {"mature": 0, "immature": 0}
    
    for img_file in os.listdir(images_dir):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        
        # Corresponding label file
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)
        
        if not os.path.isfile(label_path):
            continue
        
        # Read annotations and find majority class
        with open(label_path, "r") as f:
            lines = f.readlines()
        
        class_ids = [int(line.strip().split()[0]) for line in lines if line.strip()]
        if not class_ids:
            continue
        
        # Majority class (exclude defective=0)
        valid_ids = [cid for cid in class_ids if cid in VALID_CLASSES]
        if not valid_ids:
            continue  # Skip images with only defective labels
        
        majority_class = max(set(valid_ids), key=valid_ids.count)
        class_name = VALID_CLASSES[majority_class]
        
        # Copy to destination with short filename
        src_path = os.path.join(images_dir, img_file)
        dst_class_dir = os.path.join(DST, split_name, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)
        
        # Use index-based short name
        idx = class_counts[class_name]
        ext = os.path.splitext(img_file)[1]
        dst_name = f"{class_name}_{idx:04d}{ext}"
        dst_path = os.path.join(dst_class_dir, dst_name)
        
        shutil.copy2(src_path, dst_path)
        class_counts[class_name] += 1
    
    print(f"  {split_name}: mature={class_counts['mature']}, immature={class_counts['immature']}")

def main():
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    
    for split in ["train", "valid", "test"]:
        convert_split(split)
    
    # Total count
    total = sum(len(files) for _, _, files in os.walk(DST))
    print(f"\n✅ Total: {total} images converted to folder structure")
    print(f"   Location: {DST}")

if __name__ == "__main__":
    main()
