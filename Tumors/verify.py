import os

# Directories
output_dir_images = r"C:\Users\jayab\Downloads\dataset_png\images"
output_dir_masks = r"C:\Users\jayab\Downloads\dataset_png\masks"

# List all subfolders
image_folders = sorted(os.listdir(output_dir_images))
mask_folders = sorted(os.listdir(output_dir_masks))

# Verify correspondence
for img_folder, mask_folder in zip(image_folders, mask_folders):
    img_path = os.path.join(output_dir_images, img_folder)
    mask_path = os.path.join(output_dir_masks, mask_folder)

    img_files = sorted(os.listdir(img_path))
    mask_files = sorted(os.listdir(mask_path))

    print(f"Image Folder: {img_folder}, Mask Folder: {mask_folder}")
    print(f"Number of Images: {len(img_files)}, Number of Masks: {len(mask_files)}")
    print(f"Match: {len(img_files) == len(mask_files)}")
