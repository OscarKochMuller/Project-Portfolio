import os
import shutil
import random
from PIL import Image
import csv
from glob import glob
import numpy as np


random.seed(3653739485445)


def delete_folder(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path) 
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

delete_folder("datasets/city/gtFine_trainvaltest/gtfine/test")
delete_folder("datasets/city/leftImg8bit_trainvaltest/leftImg8bit/test")

src_folder = "datasets/city/gtFine_trainvaltest/gtfine/val"
dst_folder = "datasets/city/leftImg8bit_trainvaltest/leftImg8bit/val"

os.makedirs(dst_folder, exist_ok=True) 


def move_folder(src,dst):
    for filename in os.listdir(src):
        src_path = os.path.join(src, filename)
        dst_path = os.path.join(dst, filename)
        try:
            shutil.move(src_path, dst_path)
            print(f"Moved: {src_path} → {dst_path}")
        except Exception as e:
            print(f"Failed to move {src_path}. Reason: {e}")




move_folder("datasets/city/gtFine_trainvaltest/gtfine/val", "datasets/city/gtFine_trainvaltest/gtfine/train")
move_folder("datasets/city/leftImg8bit_trainvaltest/leftImg8bit/val","datasets/city/leftImg8bit_trainvaltest/leftImg8bit/train")




def split_and_move_cases(
    image_root,
    mask_root,
    dest_a_images,
    dest_a_masks,
    dest_b_images,
    dest_b_masks,
    x,
    y,
    copy_instead=False
):
    """
    Randomly moves (or copies) X and Y folders (and their corresponding masks)
    from source directories to two destination pairs.

    Args:
        image_root (str): Source folder containing image subfolders.
        mask_root (str): Source folder containing GT mask subfolders (same names).
        dest_a_images (str): Destination for first X image folders.
        dest_a_masks (str): Destination for first X mask folders.
        dest_b_images (str): Destination for next Y image folders.
        dest_b_masks (str): Destination for next Y mask folders.
        x (int): Number of folders to move to destination A.
        y (int): Number of folders to move to destination B.
        copy_instead (bool): If True, copies instead of moves.
    """

    os.makedirs(dest_a_images, exist_ok=True)
    os.makedirs(dest_a_masks, exist_ok=True)
    os.makedirs(dest_b_images, exist_ok=True)
    os.makedirs(dest_b_masks, exist_ok=True)

    all_folders = [f for f in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, f))]

    random.shuffle(all_folders)

    total = len(all_folders)
    print(f"Found {total} folders in {image_root}")

    if total < x + y:
        print(f"Not enough folders! Found {total}, need {x + y}")
        return

    to_a = all_folders[:x]
    to_b = all_folders[x:x+y]

    move_fn = shutil.copytree if copy_instead else shutil.move

    def move_pair(case_name, dst_img_root, dst_mask_root):
        src_img = os.path.join(image_root, case_name)
        src_mask = os.path.join(mask_root, case_name)
        dst_img = os.path.join(dst_img_root, case_name)
        dst_mask = os.path.join(dst_mask_root, case_name)

        if not os.path.exists(src_img):
            print(f"Missing image folder: {src_img}")
            return
        if not os.path.exists(src_mask):
            print(f"Missing mask folder: {src_mask}")
            return

        move_fn(src_img, dst_img)
        move_fn(src_mask, dst_mask)
        print(f"{'Copied' if copy_instead else 'Moved'}: {case_name}")

    print(f"\n Moving {len(to_a)} folders to A...")
    for c in to_a:
        move_pair(c, dest_a_images, dest_a_masks)

    print(f"\n Moving {len(to_b)} folders to B...")
    for c in to_b:
        move_pair(c, dest_b_images, dest_b_masks)

    print(f"\n Done! {len(to_a)} → A, {len(to_b)} → B (randomly selected).")








split_and_move_cases(
    image_root="datasets/city/leftImg8bit_trainvaltest/leftImg8bit/train",
    mask_root="datasets/city/gtFine_trainvaltest/gtfine/train",
    dest_a_images="datasets/city/leftImg8bit_trainvaltest/leftImg8bit/val",
    dest_a_masks="datasets/city/gtFine_trainvaltest/gtfine/val",
    dest_b_images="datasets/city/leftImg8bit_trainvaltest/leftImg8bit/test",
    dest_b_masks="datasets/city/gtFine_trainvaltest/gtfine/test",
    x=2,   
    y=10, 
    copy_instead=False
)



main_dirs = ["datasets/city/gtFine_trainvaltest/gtfine/val", "datasets/city/gtFine_trainvaltest/gtfine/test", "datasets/city/gtFine_trainvaltest/gtfine/train", "datasets/city/leftImg8bit_trainvaltest/leftImg8bit/val", "datasets/city/leftImg8bit_trainvaltest/leftImg8bit/test", "datasets/city/leftImg8bit_trainvaltest/leftImg8bit/train"]

for main_dir in main_dirs:
    for root, dirs, files in os.walk(main_dir, topdown=False):
        for name in files:
            src_path = os.path.join(root, name)
            dst_path = os.path.join(main_dir, name)
            
            if os.path.exists(dst_path):
                raise FileExistsError(f"Duplicate file found: {dst_path}")
            
            shutil.move(src_path, dst_path)
        
        for name in dirs:
            dir_path = os.path.join(root, name)
            os.rmdir(dir_path)



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

    print(f"{sample_count} images {'moved' if move else 'copied'} from {src_root} to {dst_root}")

    return sampled  

src_root = "datasets/city/leftImg8bit_trainvaltest/leftImg8bit/train"
dst_root = "finetune-sam/datasets/city/train/images"


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
        dst_path = os.path.join(gt_dst_root, original_name) 

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


train_src = "datasets/city/leftImg8bit_trainvaltest/leftImg8bit/train"
train_dst = "finetune-sam/datasets/city/train/images"
train_gt_root = "datasets/city/gtFine_trainvaltest/gtfine/train"
train_gt_dst = "finetune-sam/datasets/city/train/masks"

sampled_train = sample_images(train_src, train_dst, num_samples=750, move=False)
copy_corresponding_gt_images(sampled_train, train_gt_root, train_gt_dst, move=False)


val_src = "datasets/city/leftImg8bit_trainvaltest/leftImg8bit/val"
val_dst = "finetune-sam/datasets/city/val/images"
val_gt_root = "datasets/city/gtFine_trainvaltest/gtfine/val"
val_gt_dst = "finetune-sam/datasets/city/val/masks"

sampled_val = sample_images(val_src, val_dst, num_samples=250, move=False)
copy_corresponding_gt_images(sampled_val, val_gt_root, val_gt_dst, move=False)


test_src = "datasets/city/leftImg8bit_trainvaltest/leftImg8bit/test"
test_dst = "finetune-sam/datasets/city/test/images"
test_gt_root = "datasets/city/gtFine_trainvaltest/gtfine/test"
test_gt_dst = "finetune-sam/datasets/city/test/masks"

sampled_test = sample_images(test_src, test_dst, num_samples=1000, move=False)
copy_corresponding_gt_images(sampled_test, test_gt_root, test_gt_dst, move=False)




dataset_name="city"
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

        print(f"Processed folder: {folder} → {out_dir}")


folders = [
    "finetune-sam/datasets/"+dataset_name+"/train/masks",
    "finetune-sam/datasets/"+dataset_name+"/val/masks",
    "finetune-sam/datasets/"+dataset_name+"/test/masks"
]

gray_map = {
    26: 1, 
    28: 2,
    27: 3,
    32: 4,
    25: 5,
    33: 6,
    24: 7,
    7: 8,
    19: 9,
    20: 10
}

apply_gray_mapping_to_folders(folders, gray_map, overwrite=True)