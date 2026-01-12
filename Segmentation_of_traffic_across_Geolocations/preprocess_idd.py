from glob import glob
import os
import shutil
import random
import csv

import json
from PIL import Image, ImageDraw
import numpy as np

random.seed(92837646584)



base_dir = "datasets/idd_segmentation"
output_dir = "datasets/IDD_Segmentation/data"

folders = ["leftImg8bit", "gtFine"]

for folder in folders:
    train_path = os.path.join(base_dir, folder, "train")
    val_path = os.path.join(base_dir, folder, "val")
    out_path = os.path.join(output_dir, folder)

    os.makedirs(out_path, exist_ok=True)

    def copy_nested_files(src_dir, dst_dir):
        if not os.path.exists(src_dir):
            print(f"Skipping missing directory: {src_dir}")
            return

        for root, dirs, files in os.walk(src_dir):
            rel_path = os.path.relpath(root, src_dir)
            target_dir = os.path.join(dst_dir, rel_path)
            os.makedirs(target_dir, exist_ok=True)

            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_dir, file)

                if os.path.exists(dst_file):
                    name, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(dst_file):
                        dst_file = os.path.join(target_dir, f"{name}_{counter}{ext}")
                        counter += 1

                shutil.copy2(src_file, dst_file)

    copy_nested_files(train_path, out_path)
    copy_nested_files(val_path, out_path)

print("Merging complete! All files (including nested folders) copied to:", output_dir)




def copy_random_subfolders(
    source_dir: str,
    masks_dir: str,
    dest1: str, dest1_masks: str, x: int,
    dest2: str, dest2_masks: str, y: int,
    dest_rest: str, dest_rest_masks: str
):
    """
    Copy subfolders from source_dir (and corresponding ones from masks_dir) into three destinations:
      - X random subfolders to dest1 (+ dest1_masks)
      - Y random subfolders to dest2 (+ dest2_masks)
      - Remaining to dest_rest (+ dest_rest_masks)
    """

    for d in [dest1, dest1_masks, dest2, dest2_masks, dest_rest, dest_rest_masks]:
        os.makedirs(d, exist_ok=True)

    subfolders = [f for f in os.listdir(source_dir)
                  if os.path.isdir(os.path.join(source_dir, f))]

    if len(subfolders) < x + y:
        raise ValueError("Not enough subfolders to sample the requested amounts.")

    sampled_1 = random.sample(subfolders, x)
    remaining = [f for f in subfolders if f not in sampled_1]

    sampled_2 = random.sample(remaining, y)
    remaining = [f for f in remaining if f not in sampled_2]

    def copy_folder(src_root, dest_root, folder_name):
        src_path = os.path.join(src_root, folder_name)
        dest_path = os.path.join(dest_root, folder_name)
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        shutil.copytree(src_path, dest_path)

    for folder in sampled_1:
        copy_folder(source_dir, dest1, folder)
        copy_folder(masks_dir, dest1_masks, folder)

    for folder in sampled_2:
        copy_folder(source_dir, dest2, folder)
        copy_folder(masks_dir, dest2_masks, folder)

    for folder in remaining:
        copy_folder(source_dir, dest_rest, folder)
        copy_folder(masks_dir, dest_rest_masks, folder)

    print(f"Copied {len(sampled_1)} folders to {dest1} (+ masks)")
    print(f"Copied {len(sampled_2)} folders to {dest2} (+ masks)")
    print(f"Copied {len(remaining)} folders to {dest_rest} (+ masks)")


if __name__ == "__main__":
    copy_random_subfolders(
        source_dir="datasets/idd_segmentation/data/leftImg8bit",
        masks_dir="datasets/idd_segmentation/data/gtFine",

        dest1="datasets/idd_segmentation/data/train/images",
        dest1_masks="datasets/idd_segmentation/data/train/masks",
        x=36,

        dest2="datasets/idd_segmentation/data/val/images",
        dest2_masks="datasets/idd_segmentation/data/val/masks",
        y=12,

        dest_rest="datasets/idd_segmentation/data/test/images",
        dest_rest_masks="datasets/idd_segmentation/data/test/masks"
    )




def flatten_and_truncate(folders):
    """
    Flattens all subdirectories into the main folder, renaming each file to:
        <first 6 chars>_<parent_folder><original_extension>
    Raises an error if duplicate filenames would occur.
    """
    for main_dir in folders:
        main_dir = main_dir.rstrip("/\\")
        print(f"Processing folder: {main_dir}")

        planned_moves = []
        seen_names = set()

        for root, _, files in os.walk(main_dir, topdown=False):
            for fname in files:
                src_path = os.path.join(root, fname)
                name, ext = os.path.splitext(fname)
                truncated = name[:6]

                if root == main_dir:
                    parent_name = os.path.basename(main_dir)
                else:
                    parent_name = os.path.basename(root)

                new_name = f"{truncated}_{parent_name}{ext}"
                dst_path = os.path.join(main_dir, new_name)

                if new_name in seen_names or os.path.exists(dst_path):
                    raise FileExistsError(f"Duplicate filename would occur: {dst_path}")

                seen_names.add(new_name)
                planned_moves.append((src_path, dst_path))

        for src, dst in planned_moves:
            shutil.move(src, dst)

        for root, dirs, _ in os.walk(main_dir, topdown=False):
            for d in dirs:
                dir_path = os.path.join(root, d)
                try:
                    os.rmdir(dir_path)
                except OSError:
                    pass 

        print(f"✅ Flattened and renamed files in: {main_dir} ({len(planned_moves)} files)")

# Example usage
folders = [
    "datasets/idd_segmentation/data/train/images",
    "datasets/idd_segmentation/data/val/images",
    "datasets/idd_segmentation/data/test/images",
    "datasets/idd_segmentation/data/train/masks",
    "datasets/idd_segmentation/data/val/masks",
    "datasets/idd_segmentation/data/test/masks"
]

flatten_and_truncate(folders)




LABEL_TO_VALUE = {
    "car": 1,
    "bus": 2,
    "truck": 3,
    "motorcycle": 4,
    "rider": 5,
    "bicycle": 6,
    "person": 7,
    "road": 8,
    "traffic light": 9,
    "traffic sign": 10,
}

VALID_CLASSES = set(LABEL_TO_VALUE.keys())

def filter_json_objects(json_path: str):
    """Remove objects not in VALID_CLASSES and overwrite the JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)

    filtered_objects = [
        obj for obj in data.get("objects", [])
        if obj.get("draw", True) and obj.get("label", "") in VALID_CLASSES
    ]

    data["objects"] = filtered_objects

    with open(json_path, "w") as f:
        json.dump(data, f)

def json_to_png(json_path: str, output_folder: str):
    """Convert a single JSON file to a greyscale PNG mask."""
    with open(json_path, "r") as f:
        data = json.load(f)

    width, height = int(data["imgWidth"]), int(data["imgHeight"])
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)

    for obj in data.get("objects", []):
        label = obj.get("label", "")
        polygon = obj.get("polygon", [])
        if not polygon:
            continue
        color = LABEL_TO_VALUE.get(label, 0) 
        polygon_tuples = [tuple(point) for point in polygon]
        draw.polygon(polygon_tuples, fill=color)

    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}.png")
    img.save(output_path)

def process_folder(json_folder: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    total = 0
    for file_name in os.listdir(json_folder):
        if not file_name.endswith(".json"):
            continue
        total += 1
        json_path = os.path.join(json_folder, file_name)

        filter_json_objects(json_path)

        json_to_png(json_path, output_folder)

    print(f"Processed {total} JSON files in '{json_folder}' → PNGs saved in '{output_folder}'")

folders = [
    "datasets/idd_segmentation/data/train/masks",
    "datasets/idd_segmentation/data/val/masks",
    "datasets/idd_segmentation/data/test/masks"
]

output_folders = [
    "datasets/idd_segmentation/data/train/masks_png",
    "datasets/idd_segmentation/data/val/masks_png",
    "datasets/idd_segmentation/data/test/masks_png"
]

for src, dst in zip(folders, output_folders):
    process_folder(src, dst)
















def sample_images(src_root, dst_root, num_samples=50, move=False):
    """
    Randomly sample a number of images from a single folder and
    copy/move them to a destination folder.

    Returns:
        sampled (list): List of full paths of sampled images.
    """


    os.makedirs(dst_root, exist_ok=True)

    image_paths = glob(os.path.join(src_root, "*"))
    if len(image_paths) == 0:
        print(f"No images found in {src_root}")
        return []

    sample_count = min(num_samples, len(image_paths))
    sampled = random.sample(image_paths, sample_count)

    for img_path in sampled:
        if move:
            shutil.move(img_path, dst_root)
        else:
            shutil.copy2(img_path, dst_root)

    print(f"✅ {sample_count} images {'moved' if move else 'copied'} from {src_root} to {dst_root}")

    return sampled 




def copy_corresponding_gt_images(sampled_images, gt_root, gt_dst_root, move=False):
    """
    Find and copy/move ground-truth (GT) images matching sampled images.

    GT files have the same name but replace '_leftImg8bit' with '_gtFine_labelIds'.
    After copying/moving, the GT files are renamed to match the original image name.

    Args:
        sampled_images (list): List of sampled image paths (returned by sample_images()).
        gt_root (str): Folder containing all GT images.
        gt_dst_root (str): Destination folder for copied GT images.
        move (bool): Whether to move instead of copy (default: False).
    """
    os.makedirs(gt_dst_root, exist_ok=True)
    copied = 0

    for img_path in sampled_images:
        original_name = os.path.basename(img_path)
        gt_name = original_name.replace("_leftImg8bit", "_gtFine_labelIds")
        gt_path = os.path.join(gt_root, gt_name)
        dst_path = os.path.join(gt_dst_root, original_name)  # <-- rename to original

        print(gt_path)
        if os.path.exists(gt_path):
            if move:
                shutil.move(gt_path, dst_path)
            else:
                shutil.copy2(gt_path, dst_path)
            copied += 1
        else:
            print(f"GT not found for: {original_name}")

    print(f"{copied} GT images {'moved' if move else 'copied'} to {gt_dst_root}")



train_src = "datasets/idd_segmentation/data/train/images"
train_dst = "finetune-sam/datasets/idd/train/images"
train_gt_root = "datasets/idd_segmentation/data/train/masks_png"
train_gt_dst = "finetune-sam/datasets/idd/train/masks"

sampled_train = sample_images(train_src, train_dst, num_samples=750, move=False)
copy_corresponding_gt_images(sampled_train, train_gt_root, train_gt_dst, move=False)


val_src = "datasets/idd_segmentation/data/val/images"
val_dst = "finetune-sam/datasets/idd/val/images"
val_gt_root = "datasets/idd_segmentation/data/val/masks_png"
val_gt_dst = "finetune-sam/datasets/idd/val/masks"

sampled_val = sample_images(val_src, val_dst, num_samples=250, move=False)
copy_corresponding_gt_images(sampled_val, val_gt_root, val_gt_dst, move=False)


test_src = "datasets/idd_segmentation/data/test/images"
test_dst = "finetune-sam/datasets/idd/test/images"
test_gt_root = "datasets/idd_segmentation/data/test/masks_png"
test_gt_dst = "finetune-sam/datasets/idd/test/masks"

sampled_test = sample_images(test_src, test_dst, num_samples=1000, move=False)
copy_corresponding_gt_images(sampled_test, test_gt_root, test_gt_dst, move=False)



dataset_name="idd"
splits = ["train", "val", "test"]


output_base = "finetune-SAM/datasets/"+dataset_name

def generate_csv(split_name):
    img_dir = os.path.join(output_base, split_name, "images")
    mask_dir = os.path.join(output_base, split_name, "masks")
    csv_path = os.path.join(output_base, f"{split_name}.csv")
    
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for img_name in sorted(os.listdir(img_dir)):
            if img_name.lower().endswith(".png"):
                img_path = dataset_name + "/" + os.path.join(f"{split_name}/images", img_name).replace("\\", "/")
                mask_path = dataset_name + "/" + os.path.join(f"{split_name}/masks", img_name).replace("\\", "/")
                writer.writerow([img_path, mask_path])
    print(f"{split_name}.csv generated with {len(os.listdir(img_dir))} entries.")

for split in splits:
    generate_csv(split)



