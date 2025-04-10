import os
import pandas as pd

# Base directory where files are stored
base_dir = r"C:\Softwares\All Programs\HCA\Duke_DLDS\Series_Classification"
csv_path = r"C:\Softwares\All Programs\HCA\Duke_DLDS\SeriesClassificationKey.csv"

def count_files_by_label():
    # Read the classification key
    df = pd.read_csv(csv_path)
    label_counts = {}
    
    # Initialize counts for all labels
    unique_labels = df['Label'].unique()
    for label in unique_labels:
        label_counts[label] = 0
    
    # Iterate through each row in classification key
    for _, row in df.iterrows():
        patient_id = f"{int(row['DLDS']):04d}"
        series_id = str(row['Series'])
        label = row['Label']
        
        # Construct path to series directory
        series_path = os.path.join(base_dir, patient_id, series_id)
        
        # Count files if directory exists
        if os.path.exists(series_path):
            file_count = len([f for f in os.listdir(series_path) if os.path.isfile(os.path.join(series_path, f))])
            label_counts[label] += file_count

    # Sort labels alphabetically
    sorted_counts = {k: v for k, v in sorted(label_counts.items())}
    return sorted_counts

# Get the counts
label_file_counts = count_files_by_label()

# Print results
print("File counts for each label:")
sum=0
for label, count in label_file_counts.items():
    print(f"Label {label} has {count} files")
    sum+=count
print(f"total files: {sum}")

# Save to CSV
df_output = pd.DataFrame(list(label_file_counts.items()), columns=['Label', 'File_Count'])
df_output.to_csv('label_counts.csv', index=False)