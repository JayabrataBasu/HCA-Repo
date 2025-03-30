import os
import pandas as pd

# Base directory where files are stored
base_dir = r"C:\Softwares\All Programs\HCA\Duke_DLDS\Series_Classification\Series_Classification"

def count_files_by_label(base_dir):
    label_counts = {}

    # Iterate over each label folder in the base directory
    for label in os.listdir(base_dir):
        label_path = os.path.join(base_dir, label)

        # Check if it's a directory
        if os.path.isdir(label_path):
            file_count = sum(len(files) for _, _, files in os.walk(label_path))  # Count files recursively
            label_counts[label] = file_count

    return label_counts

# Get the counts
label_file_counts = count_files_by_label(base_dir)

# Print the results
for label, count in label_file_counts.items():
    print(f"Label: {label}, Total Files: {count}")

# Optionally, save the counts to a CSV



