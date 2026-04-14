import os
import shutil
from pathlib import Path
 
# ============================================================
# EDIT THIS if your folder name is different
# ============================================================
SOURCE_DIR = Path("chest x-ray")
 
TARGET_DIR = Path("data")
 
NORMAL_NAMES  = {"NORMAL", "Normal", "normal"}
DISEASE_NAMES = {
    "BACTERIA", "Bacteria", "bacteria",
    "BACTERIAL", "Bacterial", "bacterial",
    "VIRUS",    "Virus",    "virus",
    "VIRAL",    "Viral",    "viral",
    "PNEUMONIA","Pneumonia","pneumonia",
}
 
 
def is_image(path):
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
 
 
def is_normal_by_filename(name):
    return "normal" in name.lower()
 
 
def copy_images(src_folder, dest_folder, label):
    dest_folder.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in src_folder.iterdir():
        if img.is_file() and is_image(img):
            shutil.copy2(img, dest_folder / img.name)
            count += 1
    if count:
        print(f"  [OK] [{label}] Copied {count} images -> {dest_folder}")
    return count
 
 
def organize_split(source_split, target_split, split_name):
    print(f"\n-- {split_name.upper()} --")
 
    normal_dest  = target_split / "Normal"
    disease_dest = target_split / "Disease"
 
    if not source_split.exists():
        print(f"  [SKIP] Folder not found: {source_split}")
        return
 
    subfolders = [f for f in source_split.iterdir() if f.is_dir()]
 
    if subfolders:
        for folder in subfolders:
            name = folder.name
            if name in NORMAL_NAMES:
                copy_images(folder, normal_dest, f"Normal <- {name}")
            elif name in DISEASE_NAMES:
                copy_images(folder, disease_dest, f"Disease <- {name}")
            else:
                print(f"  [WARN] Unknown folder '{name}' - SKIPPED")
    else:
        print(f"  [INFO] No subfolders - guessing class from filename ...")
        normal_dest.mkdir(parents=True, exist_ok=True)
        disease_dest.mkdir(parents=True, exist_ok=True)
        n_count = d_count = 0
        for img in source_split.iterdir():
            if img.is_file() and is_image(img):
                if is_normal_by_filename(img.name):
                    shutil.copy2(img, normal_dest / img.name)
                    n_count += 1
                else:
                    shutil.copy2(img, disease_dest / img.name)
                    d_count += 1
        print(f"  [OK] Normal:  {n_count} images -> {normal_dest}")
        print(f"  [OK] Disease: {d_count} images -> {disease_dest}")
 
 
def print_summary():
    print("\n" + "=" * 46)
    print("  FINAL DATASET SUMMARY")
    print("=" * 46)
    total = 0
    for split in ["train", "test"]:
        for cls in ["Normal", "Disease"]:
            folder = TARGET_DIR / split / cls
            if folder.exists():
                count = len([f for f in folder.iterdir() if is_image(f)])
                print(f"  data/{split}/{cls:<10}  ->  {count:>5} images")
                total += count
    print(f"  {'-'*38}")
    print(f"  TOTAL                ->  {total:>5} images")
    print("=" * 46)
 
 
def main():
    print("=" * 50)
    print("  organize_dataset.py  -  Medical AI Project")
    print("=" * 50)
 
    if not SOURCE_DIR.exists():
        print(f"\n[ERROR] Cannot find folder: '{SOURCE_DIR}'")
        print(f"\n  Files/folders found in current directory:")
        for item in sorted(Path(".").iterdir()):
            marker = "[DIR] " if item.is_dir() else "[FILE]"
            print(f"    {marker}  {item.name}")
        return
 
    print(f"\n[OK] Source folder found: {SOURCE_DIR.resolve()}")
    print(f"[OK] Target folder:       {TARGET_DIR.resolve()}")
 
    organize_split(SOURCE_DIR / "train", TARGET_DIR / "train", "train")
    organize_split(SOURCE_DIR / "test",  TARGET_DIR / "test",  "test")
 
    val_source = SOURCE_DIR / "val"
    if val_source.exists():
        print(f"\n  [INFO] Found val/ folder - merging into train/ ...")
        organize_split(val_source, TARGET_DIR / "train", "val merged into train")
 
    print_summary()
    print("\n[DONE] Your data/ folder is ready.")
    print("  Next step:  python main.py\n")
 
 
if __name__ == "__main__":
    main()
