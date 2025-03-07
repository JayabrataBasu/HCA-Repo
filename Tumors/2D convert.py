import numpy as np
from PIL import Image
import os
import SimpleITK as sitk
from config import volume_files, data_dir, segmentation_files


def nii_to_png(nii_file, output_dir):
    """Convert a NIfTI file to PNG slices."""
    os.makedirs(output_dir, exist_ok=True)
    img = sitk.GetArrayFromImage(sitk.ReadImage(nii_file))

    # Normalize image data to [0, 255]
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0

    # Save each slice as a PNG file
    for i in range(img.shape[0]):  # Iterate over axial slices
        slice_img = img[i].astype(np.uint8)
        Image.fromarray(slice_img).save(os.path.join(output_dir, f"slice_{i:03d}.png"))


# Define output directories for images and masks
output_dir_images = r"C:\Users\jayab\Downloads\dataset_png\images"
output_dir_masks = r"C:\Users\jayab\Downloads\dataset_png\masks"

os.makedirs(output_dir_images, exist_ok=True)
os.makedirs(output_dir_masks, exist_ok=True)

# Process all volumes and segmentations
for vol_file, seg_file in zip(volume_files, segmentation_files):
    print(f"Processing Volume: {vol_file}, Segmentation: {seg_file}")

    # Convert volume to PNGs
    nii_to_png(
        os.path.join(data_dir, vol_file),
        os.path.join(output_dir_images, vol_file.split('.')[0])
    )

    # Convert segmentation to PNGs
    nii_to_png(
        os.path.join(data_dir, seg_file),
        os.path.join(output_dir_masks, seg_file.split('.')[0])
    )

print("All volumes and segmentations have been converted to PNG format.")
