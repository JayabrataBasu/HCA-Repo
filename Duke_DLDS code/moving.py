import os
import shutil
import pandas as pd

# Path to the CSV file and base directory
csv_path = r"C:\Softwares\All Programs\HCA\Duke_DLDS\SeriesClassificationKey.csv"
base_dir = r"C:\Softwares\All Programs\HCA\Duke_DLDS\Series_Classification\Series_Classification"

def restructure_folders(csv_path, base_dir):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Iterate over each row in the CSV
    for _, row in df.iterrows():
        dlds = str(row['DLDS']).zfill(4)  # Ensure DLDS folder name is zero-padded (e.g., 0001)
        series_number = str(row['Series'])
        label = str(row['Label'])

        # Define paths
        source_folder = os.path.join(base_dir, dlds, series_number)
        label_series_folder = os.path.join(base_dir, label, f"{series_number}_DLDS_{dlds}")  # Unique folder name

        # Create the new label/series folder if it doesn't exist
        os.makedirs(label_series_folder, exist_ok=True)

        # Copy all contents of the source folder to the label/series folder
        if os.path.exists(source_folder):
            for item in os.listdir(source_folder):
                item_path = os.path.join(source_folder, item)
                shutil.copy2(item_path, label_series_folder)  # Copy files instead of moving
            print(f"Copied contents of {source_folder} -> {label_series_folder}")
        else:
            print(f"Folder not found: {source_folder}")

# Run the function
restructure_folders(csv_path, base_dir)
