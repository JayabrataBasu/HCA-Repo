import pandas as pd

# Load the CSV file
file_path = "C:/Softwares/All Programs/HCA/Duke_DLDS/SeriesClassificationKey.csv"  # Ensure the file is in the same directory
df = pd.read_csv(file_path)

# Sort the dataframe by 'Label' column
df_sorted = df.sort_values(by="Label")

# Save the sorted dataframe
sorted_file_path = "Sorted_SeriesClassificationKey.csv"
df_sorted.to_csv(sorted_file_path, index=False)

print(f"Sorted file saved as {sorted_file_path}")
