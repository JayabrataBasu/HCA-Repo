import os
import pydicom
import numpy as np
import pandas as pd
from datetime import datetime
import SimpleITK as sitk
from radiomics import featureextractor

# Define base path
base_path = r'C:\Users\jayab\Duke_DLDS'
output_path = base_path  # Save outputs to the same folder

# Define paths for images and masks
segmentation_path = os.path.join(base_path, 'Segmentation', 'Segmentation')

# Create a timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Initialize radiomics feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.enableAllFeatures()  # Enable all available features


# Function to convert DICOM to SimpleITK image
def dicom_to_sitk(dicom_path):
    dicom = pydicom.dcmread(dicom_path)
    array = dicom.pixel_array.astype(np.float32)
    image = sitk.GetImageFromArray(array)
    return image


# Function to extract radiomics features
def extract_features(image_path, mask_path):
    try:
        # Convert DICOM to SimpleITK images
        image = dicom_to_sitk(image_path)
        mask = dicom_to_sitk(mask_path)

        # Extract features
        features = extractor.execute(image, mask)

        # Convert to dictionary (exclude the diagnostic features)
        feature_dict = {k: v for k, v in features.items() if not k.startswith('diagnostics_')}

        return feature_dict
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


# Find all patient folders
patient_folders = [f for f in os.listdir(segmentation_path) if os.path.isdir(os.path.join(segmentation_path, f))]

# Process patients
all_features = []
processed_patients = 0

for patient_id in patient_folders:
    patient_path = os.path.join(segmentation_path, patient_id)
    series_folders = [f for f in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, f))]

    for series_id in series_folders:
        series_path = os.path.join(patient_path, series_id)
        image_folder = os.path.join(series_path, 'images')
        mask_folder = os.path.join(series_path, 'masks')

        # Check if folders exist
        if not os.path.exists(image_folder) or not os.path.exists(mask_folder):
            continue

        # Get list of files
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.dicom')])
        mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.dicom')])

        if not image_files or not mask_files:
            continue

        # Process first slice only for demonstration
        image_path = os.path.join(image_folder, image_files[0])
        mask_path = os.path.join(mask_folder, mask_files[0])

        # Extract features
        features = extract_features(image_path, mask_path)

        if features:
            # Add patient and series info
            features['patient_id'] = patient_id
            features['series_id'] = series_id
            features['slice_id'] = image_files[0]

            all_features.append(features)
            print(f"Extracted features for patient {patient_id}, series {series_id}")

            processed_patients += 1

            # Process only 10 patients for demonstration
            if processed_patients >= 10:
                break

    if processed_patients >= 10:
        break

# Save all features to CSV
if all_features:
    features_df = pd.DataFrame(all_features)
    output_file = os.path.join(output_path, f'radiomics_features_{timestamp}.csv')
    features_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

    # Also save a summary with key features only
    key_features = ['patient_id', 'series_id', 'slice_id']

    # Add some important radiomics features (if available)
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
