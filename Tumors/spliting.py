import shutil
import os

def split_dataset(input_dir_images, input_dir_masks, output_base_dir, train_ratio=0.8):
    """Split dataset into train/val sets."""
    image_dirs = sorted(os.listdir(input_dir_images))
    mask_dirs = sorted(os.listdir(input_dir_masks))

    assert len(image_dirs) == len(mask_dirs), "Mismatch between images and masks!"

    # Create train/val folders
    train_images_dir = os.path.join(output_base_dir, "train/images")
    train_masks_dir = os.path.join(output_base_dir, "train/masks")
    val_images_dir = os.path.join(output_base_dir, "val/images")
    val_masks_dir = os.path.join(output_base_dir, "val/masks")

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_masks_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_masks_dir, exist_ok=True)

    # Split data
    num_train = int(len(image_dirs) * train_ratio)

    for i, (img_folder, mask_folder) in enumerate(zip(image_dirs, mask_dirs)):
        if i < num_train:
            shutil.move(os.path.join(input_dir_images, img_folder), train_images_dir)
            shutil.move(os.path.join(input_dir_masks, mask_folder), train_masks_dir)
        else:
            shutil.move(os.path.join(input_dir_images, img_folder), val_images_dir)
            shutil.move(os.path.join(input_dir_masks, mask_folder), val_masks_dir)


# Example usage
split_dataset(
    r"C:\Users\jayab\Downloads\dataset_png\images",
    r"C:\Users\jayab\Downloads\dataset_png\masks",
    r"C:\Users\jayab\Downloads\processed_dataset"
)
