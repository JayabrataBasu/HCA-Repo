import os
import SimpleITK as sitk

# Define the directory containing the NIfTI files
data_dir = r"C:\Users\jayab\Downloads\dataset"

# List all volume and segmentation files
volume_files = sorted([f for f in os.listdir(data_dir) if f.startswith("volume") and f.endswith(".nii")])
segmentation_files = sorted([f for f in os.listdir(data_dir) if f.startswith("segmentation") and f.endswith(".nii")])

# Verify shape correspondence
correspondence = []
for vol, seg in zip(volume_files, segmentation_files):
    vol_path = os.path.join(data_dir, vol)
    seg_path = os.path.join(data_dir, seg)

    # Load the NIfTI files using SimpleITK
    vol_img = sitk.ReadImage(vol_path)
    seg_img = sitk.ReadImage(seg_path)

    # Check if shapes match
    vol_size = vol_img.GetSize()
    seg_size = seg_img.GetSize()
    correspondence.append((vol, seg, vol_size == seg_size))

# Print results
for vol, seg, match in correspondence:
    print(f"Volume: {vol}, Segmentation: {seg}, Shape Match: {match}")
