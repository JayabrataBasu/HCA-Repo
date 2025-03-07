import os

# Define the directory containing the NIfTI files
data_dir = r"C:\Users\jayab\Downloads\dataset"

# List all volume and segmentation files
volume_files = sorted([f for f in os.listdir(data_dir) if f.startswith("volume") and f.endswith(".nii")])
segmentation_files = sorted([f for f in os.listdir(data_dir) if f.startswith("segmentation") and f.endswith(".nii")])
