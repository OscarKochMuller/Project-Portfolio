import os
import shutil
import random
from PIL import Image
import csv
import numpy as np


random.seed(5723782397856)

dataset_name="bdd"

image_train_dir = "datasets/"+dataset_name+"/images/train"
image_val_dir = "datasets/"+dataset_name+"/images/val"
mask_train_dir = "datasets/"+dataset_name+"/labels/train"
mask_val_dir = "datasets/"+dataset_name+"/labels/val"

combined_image_dir = "datasets/"+dataset_name+"/data/images"
combined_mask_dir = "datasets/"+dataset_name+"/data/masks"

os.makedirs(combined_image_dir, exist_ok=True)
os.makedirs(combined_mask_dir, exist_ok=True)

def copy_files(src_dir, dst_dir):
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        dst_file = os.path.join(dst_dir, filename)
        if os.path.exists(dst_file):
            base, ext = os.path.splitext(filename)
            count = 1
            while os.path.exists(dst_file):
                dst_file = os.path.join(dst_dir, f"{base}_{count}{ext}")
                count += 1
        shutil.copy2(src_file, dst_file)

copy_files(image_train_dir, combined_image_dir)
copy_files(image_val_dir, combined_image_dir)

for src_dir in [mask_train_dir, mask_val_dir]:
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        base, ext = os.path.splitext(filename)
        new_name = base[:-9] + ext
        dst_file = os.path.join(combined_mask_dir, new_name)
        if os.path.exists(dst_file):
            count = 1
            while os.path.exists(dst_file):
                dst_file = os.path.join(combined_mask_dir, f"{base[:-9]}_{count}{ext}")
                count += 1
        shutil.copy2(src_file, dst_file)

print("Train and val images & masks merged successfully.")

output_base = "finetune-SAM/datasets/"+dataset_name
splits = ["train", "val", "test"]
split_sizes = {"train": 750, "val": 250, "test": 1000}

for split in splits:
    os.makedirs(os.path.join(output_base, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_base, split, "masks"), exist_ok=True)

all_images = [f for f in os.listdir(combined_image_dir) if f.lower().endswith(".jpg")]
all_images.sort()
random.shuffle(all_images)

train_images = all_images[:split_sizes["train"]]
val_images = all_images[split_sizes["train"]:split_sizes["train"] + split_sizes["val"]]
test_images = all_images[split_sizes["train"] + split_sizes["val"]:
                         split_sizes["train"] + split_sizes["val"] + split_sizes["test"]]

def convert_image_jpg_to_png(src_path, dst_path):
    with Image.open(src_path) as img:
        img.save(dst_path, format="PNG")

def copy_images_and_masks(image_list, split_name):
    img_dst_dir = os.path.join(output_base, split_name, "images")
    mask_dst_dir = os.path.join(output_base, split_name, "masks")
    
    for img_name in image_list:
        src_img_path = os.path.join(combined_image_dir, img_name)
        dst_img_name = os.path.splitext(img_name)[0] + ".png"
        dst_img_path = os.path.join(img_dst_dir, dst_img_name)
        convert_image_jpg_to_png(src_img_path, dst_img_path)
        
        mask_name = os.path.splitext(img_name)[0] + ".png" 
        src_mask_path = os.path.join(combined_mask_dir, mask_name)
        dst_mask_path = os.path.join(mask_dst_dir, mask_name)
        if os.path.exists(src_mask_path):
            shutil.copy2(src_mask_path, dst_mask_path)
        else:
            print(f"Warning: Mask not found for {img_name}, expected {mask_name}")

copy_images_and_masks(train_images, "train")
copy_images_and_masks(val_images, "val")
copy_images_and_masks(test_images, "test")

print("Dataset split complete:")
print(f"Train: {len(train_images)}")
print(f"Val:   {len(val_images)}")
print(f"Test:  {len(test_images)}")






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





def apply_gray_mapping_to_folders(folder_list, gray_mapping, output_root=None, overwrite=False):
    """
    Apply a grayscale pixel value mapping to all PNG images in a list of folders.
    All pixel values NOT in gray_mapping will be set to 0.

    Args:
        folder_list (list[str]): List of folder paths containing PNG images.
        gray_mapping (dict[int, int]): Mapping of old grayscale values to new ones.
        output_root (str, optional): Root folder to save mapped images.
                                     If None and overwrite=False, saves in "<folder>_mapped".
        overwrite (bool): If True, replaces original images.
    """
    lut = np.zeros(256, dtype=np.uint8)
    for old_val, new_val in gray_mapping.items():
        if 0 <= old_val < 256:
            lut[old_val] = new_val

    for folder in folder_list:
        if overwrite:
            out_dir = folder
        else:
            out_dir = output_root or f"{folder}_mapped"
            os.makedirs(out_dir, exist_ok=True)

        for file_name in os.listdir(folder):
            if file_name.lower().endswith(".png"):
                input_path = os.path.join(folder, file_name)
                output_path = os.path.join(out_dir, file_name)

                img = Image.open(input_path).convert("L")
                np_img = np.array(img)

                mapped_img = Image.fromarray(lut[np_img])

                mapped_img.save(output_path)

        print(f"✅ Processed folder: {folder} → {out_dir}")


folders = [
    "finetune-sam/datasets/"+dataset_name+"/train/masks",
    "finetune-sam/datasets/"+dataset_name+"/val/masks",
    "finetune-sam/datasets/"+dataset_name+"/test/masks"
]

gray_map = {
    13: 1, 
    15: 2,
    14: 3,
    17: 4,
    12: 5,
    18: 6,
    11: 7,
    0: 8,
    6: 9,
    7: 10
}

apply_gray_mapping_to_folders(folders, gray_map, overwrite=True)