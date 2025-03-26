import os
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from tqdm import tqdm
from scipy.stats import skew, kurtosis


# Define base path
base_path = r'C:\Users\jayab\Duke_DLDS'
output_path = os.path.join(base_path, 'texture_analysis_output_of_series_classification')
os.makedirs(output_path, exist_ok=True)

segmentation_path = os.path.join(base_path, 'Series_Classification', 'Series_Classification')


# Create timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load the series classification key
series_class_df = pd.read_csv(os.path.join(base_path, 'SeriesClassificationKey.csv'))

# Create a dictionary to map patient/series to labels
series_labels = {}
for _, row in series_class_df.iterrows():
    patient_id = f"{int(row['DLDS']):04d}"
    series_id = str(row['Series'])
    label = row['Label']
    series_labels[(patient_id, series_id)] = label


# Function to extract texture features from a single image
def extract_texture_features(image_array):
    # Normalize image to 8-bit range for texture analysis
    image_8bit = np.clip(((image_array - image_array.min()) * 255 /
                          (image_array.max() - image_array.min())), 0, 255).astype(np.uint8)

    # First-order statistics
    mean = np.mean(image_array)
    std = np.std(image_array)
    skewness = skew(image_array.flatten())
    kurt = kurtosis(image_array.flatten())

    # GLCM features
    distances = [1, 3, 5]  # Multiple distances for more robust analysis
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(image_8bit, distances, angles, 256, symmetric=True, normed=True)

    contrast = np.mean(graycoprops(glcm, 'contrast'))
    dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    ASM = np.mean(graycoprops(glcm, 'ASM'))

    # LBP features
    lbp = local_binary_pattern(image_8bit, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)

    return {
        'mean': mean,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurt,
        'contrast': contrast,
        'dissimilarity': dissimilarity,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation,
        'ASM': ASM,
        'lbp_hist_1': lbp_hist[0],
        'lbp_hist_2': lbp_hist[1],
        'lbp_hist_3': lbp_hist[2],
        'lbp_hist_4': lbp_hist[3],
        'lbp_hist_5': lbp_hist[4],
        'lbp_hist_6': lbp_hist[5],
        'lbp_hist_7': lbp_hist[6],
        'lbp_hist_8': lbp_hist[7],
        'lbp_hist_9': lbp_hist[8],
        'lbp_hist_10': lbp_hist[9]
    }


# Process all valid combinations
all_features = []
processed_count = 0
visualization_count = 0

print("Starting texture analysis...")
for patient_id, series_id in tqdm(series_labels.keys(), desc="Processing patients"):
    # Check if the folders exist
    image_folder = os.path.join(segmentation_path, patient_id, series_id)

    if not os.path.exists(image_folder):
        continue

    # Get list of image files
    try:
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.dicom')])
    except Exception as e:
        print(f"Error listing files for patient {patient_id}, series {series_id}: {e}")
        continue

    if not image_files:
        continue

    # Process middle slice for more representative liver tissue
    middle_idx = len(image_files) // 2
    image_path = os.path.join(image_folder, image_files[middle_idx])

    try:
        # Read DICOM file
        image_dicom = pydicom.dcmread(image_path)
        image_array = image_dicom.pixel_array

        # Extract texture features
        feature_dict = extract_texture_features(image_array)

        # Add patient and series info
        feature_dict['patient_id'] = patient_id
        feature_dict['series_id'] = series_id
        feature_dict['slice_id'] = image_files[middle_idx]
        feature_dict['label'] = series_labels.get((patient_id, series_id), "Unknown")

        all_features.append(feature_dict)

        # Create visualization for first 5 processed images
        if visualization_count < 5:
            # Create a figure
            plt.figure(figsize=(15, 5))

            # Display the original image
            plt.subplot(1, 3, 1)
            plt.title(f'Original - Patient {patient_id}, Series {series_id}')
            plt.imshow(image_array, cmap='gray')
            plt.axis('off')

            # Display GLCM visualization
            plt.subplot(1, 3, 2)
            plt.title('GLCM Features')
            # Normalize for visualization
            image_8bit = ((image_array - image_array.min()) * 255 /
                          (image_array.max() - image_array.min())).astype(np.uint8)
            glcm = graycomatrix(image_8bit, [1], [0], 256, symmetric=True, normed=True)
            plt.imshow(np.log1p(glcm[:, :, 0, 0]), cmap='viridis')
            plt.colorbar(label='Log(GLCM)')
            plt.axis('off')

            # Display LBP visualization
            plt.subplot(1, 3, 3)
            plt.title('LBP Features')
            lbp = local_binary_pattern(image_8bit, 8, 1, method='uniform')
            plt.imshow(lbp, cmap='jet')
            plt.colorbar(label='LBP Value')
            plt.axis('off')

            # Save the figure
            output_filename = f'texture_viz_{patient_id}_{series_id}_{timestamp}.png'
            plt.savefig(os.path.join(output_path, output_filename), dpi=300, bbox_inches='tight')
            plt.close()

            visualization_count += 1

        processed_count += 1

    except Exception as e:
        print(f"Error processing patient {patient_id}, series {series_id}: {e}")

# Save all features to CSV
if all_features:
    features_df = pd.DataFrame(all_features)
    output_file = os.path.join(output_path, f'texture_features_{timestamp}.csv')
    features_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

    # Generate summary statistics
    summary_stats = features_df.describe()
    summary_file = os.path.join(output_path, f'texture_summary_stats_{timestamp}.csv')
    summary_stats.to_csv(summary_file)
    print(f"Summary statistics saved to {summary_file}")

    # Create feature correlation heatmap
    plt.figure(figsize=(12, 10))
    numeric_features = features_df.select_dtypes(include=[np.number])
    corr = numeric_features.corr()
    plt.imshow(corr, cmap='coolwarm')
    plt.colorbar(label='Correlation')
    plt.title('Feature Correlation Matrix')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.tight_layout()
    corr_file = os.path.join(output_path, f'feature_correlation_{timestamp}.png')
    plt.savefig(corr_file, dpi=300, bbox_inches='tight')
    plt.close()

    # Create boxplots for key features
    key_features = ['contrast', 'homogeneity', 'energy', 'correlation']
    plt.figure(figsize=(12, 8))
    features_df[key_features].boxplot()
    plt.title('Distribution of Key Texture Features')
    plt.tight_layout()
    boxplot_file = os.path.join(output_path, f'feature_boxplots_{timestamp}.png')
    plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Processed {processed_count} series successfully")
else:
    print("No features were extracted.")

print(f"Processing complete! All outputs saved to {output_path}")
