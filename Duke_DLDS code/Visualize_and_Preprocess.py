import os
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import SimpleITK as sitk
from radiomics import featureextractor

# Define base path
base_path = r'C:\Users\jayab\Duke_DLDS'
# Create an output folder if it doesn't exist
output_path = os.path.join(base_path, 'output')
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Created output directory: {output_path}")

segmentation_path = os.path.join(base_path, 'Segmentation', 'Segmentation')

# Create timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load the segmentation key to get valid combinations
key_df = pd.read_csv(os.path.join(base_path, 'SegmentationKey.csv'))

# Create a list of valid patient/series combinations
valid_combinations = []
for _, row in key_df.iterrows():
    patient_id = f"{int(row['DLDS']):04d}"  # Format as 4-digit string with leading zeros
    series_id = str(row['Series'])
    valid_combinations.append((patient_id, series_id))

print(f"Found {len(valid_combinations)} valid patient/series combinations")


# Method 1: Visualize DICOM Images
def visualize_images():
    print("Starting Method 1: Visualization of DICOM Images")

    processed_count = 0
    results = []

    # Process only valid combinations
    for patient_id, series_id in valid_combinations:
        # Check if the folders exist
        image_folder = os.path.join(segmentation_path, patient_id, series_id, 'images')
        mask_folder = os.path.join(segmentation_path, patient_id, series_id, 'masks')

        if not (os.path.exists(image_folder) and os.path.exists(mask_folder)):
            continue

        # Get list of files - CHANGED .dcm to .dicom
        try:
            image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.dicom')])
            mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.dicom')])
        except Exception as e:
            print(f"Error listing files for patient {patient_id}, series {series_id}: {e}")
            continue

        if not image_files or not mask_files:
            continue

        # Load the first image and mask
        try:
            image_path = os.path.join(image_folder, image_files[0])
            mask_path = os.path.join(mask_folder, mask_files[0])

            # Read DICOM files
            image_dicom = pydicom.dcmread(image_path)
            mask_dicom = pydicom.dcmread(mask_path)

            # Convert to numpy arrays
            image_array = image_dicom.pixel_array
            mask_array = mask_dicom.pixel_array

            # Create a figure
            plt.figure(figsize=(12, 6))

            # Display the image
            plt.subplot(1, 3, 1)
            plt.title(f'MRI Image - Patient {patient_id}, Series {series_id}')
            plt.imshow(image_array, cmap='gray')
            plt.axis('off')

            # Display the mask
            plt.subplot(1, 3, 2)
            plt.title('Liver Mask')
            plt.imshow(mask_array, cmap='gray')
            plt.axis('off')

            # Display the overlay
            plt.subplot(1, 3, 3)
            plt.title('Overlay')
            plt.imshow(image_array, cmap='gray')
            plt.imshow(mask_array > 0, cmap='jet', alpha=0.3)
            plt.axis('off')

            # Save the figure
            output_filename = f'visualization_{patient_id}_{series_id}_{timestamp}.png'
            plt.savefig(os.path.join(output_path, output_filename), dpi=300, bbox_inches='tight')
            plt.close()

            # Store results
            results.append({
                'patient_id': patient_id,
                'series_id': series_id,
                'image_shape': image_array.shape,
                'mask_shape': mask_array.shape,
                'mask_percentage': (np.sum(mask_array > 0) / mask_array.size) * 100
            })

            processed_count += 1
            print(f"Processed patient {patient_id}, series {series_id}")

            # Process only 5 patients for demonstration
            if processed_count >= 5:
                break

        except Exception as e:
            print(f"Error processing patient {patient_id}, series {series_id}: {e}")

    # Save summary to CSV
    if results:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(os.path.join(output_path, f'visualization_summary_{timestamp}.csv'), index=False)
        print(f"Visualization complete. Processed {processed_count} series.")
    else:
        print("No images were processed.")


# Method 2: Radiomics Feature Extraction
def extract_radiomics():
    print("Starting Method 2: Radiomics Feature Extraction")

    # Initialize radiomics feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()

    processed_count = 0
    all_features = []

    # Process only valid combinations
    for patient_id, series_id in valid_combinations:
        # Check if the folders exist
        image_folder = os.path.join(segmentation_path, patient_id, series_id, 'images')
        mask_folder = os.path.join(segmentation_path, patient_id, series_id, 'masks')

        if not (os.path.exists(image_folder) and os.path.exists(mask_folder)):
            continue

        # Get list of files - CHANGED .dcm to .dicom
        try:
            image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.dicom')])
            mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.dicom')])
        except Exception as e:
            print(f"Error listing files for patient {patient_id}, series {series_id}: {e}")
            continue

        if not image_files or not mask_files:
            continue

        # Load the first image and mask
        try:
            image_path = os.path.join(image_folder, image_files[0])
            mask_path = os.path.join(mask_folder, mask_files[0])

            # Read DICOM files
            image_dicom = pydicom.dcmread(image_path)
            mask_dicom = pydicom.dcmread(mask_path)

            # Convert to numpy arrays
            image_array = image_dicom.pixel_array.astype(np.float32)
            mask_array = mask_dicom.pixel_array.astype(np.int32)

            # Convert to SimpleITK images
            image = sitk.GetImageFromArray(image_array)
            mask = sitk.GetImageFromArray(mask_array)

            # Extract features
            features = extractor.execute(image, mask)

            # Convert to dictionary (exclude diagnostic features)
            feature_dict = {k: v for k, v in features.items() if not k.startswith('diagnostics_')}

            # Add patient and series info
            feature_dict['patient_id'] = patient_id
            feature_dict['series_id'] = series_id
            feature_dict['slice_id'] = image_files[0]

            all_features.append(feature_dict)

            processed_count += 1
            print(f"Extracted features for patient {patient_id}, series {series_id}")

            # Process only 10 patients for demonstration
            if processed_count >= 10:
                break

        except Exception as e:
            print(f"Error extracting features for patient {patient_id}, series {series_id}: {e}")

    # Save all features to CSV
    if all_features:
        features_df = pd.DataFrame(all_features)
        output_file = os.path.join(output_path, f'radiomics_features_{timestamp}.csv')
        features_df.to_csv(output_file, index=False)
        print(f"Features saved to {output_file}")

        # Also save a summary with key features
        key_features = ['patient_id', 'series_id', 'slice_id']

        # Add important radiomics features
        important_features = [
            'original_firstorder_Mean',
            'original_firstorder_Entropy',
            'original_glcm_JointEntropy',
            'original_glcm_Contrast',
            'original_glszm_SizeZoneNonUniformity',
            'original_shape_Sphericity',
            'original_shape_VoxelVolume'
        ]

        for feature in important_features:
            if feature in features_df.columns:
                key_features.append(feature)

        summary_df = features_df[key_features]
        summary_file = os.path.join(output_path, f'radiomics_summary_{timestamp}.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary saved to {summary_file}")
    else:
        print("No features were extracted.")


# Run both methods
if __name__ == "__main__":
    # Run Method 1: Visualization
    visualize_images()

    # Run Method 2: Radiomics Feature Extraction
    extract_radiomics()

    print(f"Processing complete! All outputs saved to {output_path}")