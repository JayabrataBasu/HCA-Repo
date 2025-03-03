import SimpleITK as sitk
import os

data_dir = r"C:\Users\jayab\Downloads\dataset"

# List all volume and segmentation files
volume_files = sorted([f for f in os.listdir(data_dir) if f.startswith("volume") and f.endswith(".nii")])
segmentation_files = sorted([f for f in os.listdir(data_dir) if f.startswith("segmentation") and f.endswith(".nii")])

# Verify file integrity using SimpleITK
for vol_file, seg_file in zip(volume_files, segmentation_files):
    vol_path = os.path.join(data_dir, vol_file)
    seg_path = os.path.join(data_dir, seg_file)

    try:
        vol_img = sitk.ReadImage(vol_path)
        seg_img = sitk.ReadImage(seg_path)
        print(f"Volume: {vol_file}, Segmentation: {seg_file}, Status: Valid")
    except Exception as e:
        print(f"Volume: {vol_file}, Segmentation: {seg_file}, Error: {e}")
