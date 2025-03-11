# This program analyzes MRI images to classify them into different sequence types
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import cv2
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the correct file paths
root_dir = r'C:\Users\jayab\Duke_DLDS\Series_Classification\Series_Classification'
csv_path = r'C:\Users\jayab\Duke_DLDS\SeriesClassificationKey.csv'
output_folder = r'C:\Users\jayab\Duke_DLDS\output_folder'

# Create the output folder if it doesn't exist yet
os.makedirs(output_folder, exist_ok=True)


# This function reads the CSV file to get labels for each image series
def load_series_labels(csv_path):
    # Create an empty dictionary to store the labels
    series_labels = {}

    # Open and read the CSV file
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row

        # For each row in the CSV, store the series ID and its label
        for row in reader:
            # The CSV has three columns: DLDS, Series, and Label
            dlds, series, label = row
            # Create a key in format "patient_id/series_id"
            # Add leading zeros to patient ID to match folder structure
            patient_id = f"{int(dlds):04d}"
            series_id = f"{patient_id}/{series}"
            series_labels[series_id] = label

    return series_labels


# This function loads and processes the MRI images
def load_dicom_images(root_dir, series_labels, max_images_per_series=5):
    import pydicom  # Library for reading DICOM medical images

    X = []  # Will store the image data
    y = []  # Will store the labels
    series_ids = []  # Will store which series each image came from

    # Loop through each patient folder
    for patient_id in tqdm(os.listdir(root_dir)):
        patient_dir = os.path.join(root_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue

        # Loop through each series folder for this patient
        for series_id in os.listdir(patient_dir):
            series_dir = os.path.join(patient_dir, series_id)
            if not os.path.isdir(series_dir):
                continue

            # Get the label for this series from our CSV data
            full_series_id = f"{patient_id}/{series_id}"
            if full_series_id not in series_labels:
                continue

            label = series_labels[full_series_id]

            # Find all DICOM files in this series
            dicom_files = []
            for file in os.listdir(series_dir):
                file_path = os.path.join(series_dir, file)
                if os.path.isfile(file_path):
                    dicom_files.append(file)

            if not dicom_files:
                continue

            # Select middle slices (usually the most informative)
            middle_idx = len(dicom_files) // 2
            start_idx = max(0, middle_idx - max_images_per_series // 2)
            selected_files = dicom_files[start_idx:start_idx + max_images_per_series]

            # Process each selected DICOM file
            for dicom_file in selected_files:
                try:
                    # Read the DICOM file
                    file_path = os.path.join(series_dir, dicom_file)
                    ds = pydicom.dcmread(file_path, force=True)

                    # Convert to a regular image array
                    img = ds.pixel_array

                    # Resize to a standard size (128x128 pixels)
                    img = cv2.resize(img, (128, 128))

                    # Normalize the pixel values (make them between 0 and 1)
                    img = img / np.max(img) if np.max(img) > 0 else img

                    # Flatten the image (turn 2D image into 1D array)
                    img_flat = img.flatten()

                    # Add this image and its label to our lists
                    X.append(img_flat)
                    y.append(label)
                    series_ids.append(full_series_id)
                except Exception as e:
                    print(f"Error processing {dicom_file}: {e}")

    return np.array(X), np.array(y), series_ids


# This function saves our results and creates visualizations
def save_results(clf, le, X_test, y_test, y_pred):
    # Save a text report showing how well our model performed
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    with open(os.path.join(output_folder, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # Create and save a confusion matrix (shows predictions vs actual labels)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'))

    # If our model has feature importance, save that too
    if hasattr(clf, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        feature_importance = clf.feature_importances_
        # Plot top 20 features
        n_features = min(20, len(feature_importance))
        indices = np.argsort(feature_importance)[-n_features:]
        plt.barh(range(n_features), feature_importance[indices])
        plt.yticks(range(n_features), [f'Feature {i}' for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'feature_importance.png'))


# Main program
def main():
    print("Loading series labels from CSV...")
    series_labels = load_series_labels(csv_path)
    print(f"Loaded {len(series_labels)} series labels from CSV.")

    print("Loading and preprocessing DICOM images...")
    X, y, series_ids = load_dicom_images(root_dir, series_labels)

    if len(X) == 0:
        print("No valid images found. Check path and DICOM files.")
        return

    print(f"Loaded {len(X)} images from {len(set(series_ids))} series.")

    # Save a summary of what data we loaded
    with open(os.path.join(output_folder, 'data_summary.txt'), 'w') as f:
        f.write(f"Total images loaded: {len(X)}\n")
        f.write(f"Total unique series: {len(set(series_ids))}\n")
        f.write(f"Label distribution:\n")
        for label, count in zip(*np.unique(y, return_counts=True)):
            f.write(f"  {label}: {count}\n")

    # Convert text labels to numbers for the machine learning model
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split our data into training set (to teach the model) and testing set (to evaluate it)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("Training Random Forest classifier...")
    # Create and train a Random Forest model (good for image classification)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Test how well our model works on the test data
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")

    # Save our results and visualizations
    save_results(clf, le, X_test, y_test, y_pred)

    # Save the trained model and other important files
    joblib.dump(clf, os.path.join(output_folder, 'mri_sequence_classifier.joblib'))
    joblib.dump(le, os.path.join(output_folder, 'label_encoder.joblib'))

    # Save which images were used for training and testing (for reproducibility)
    np.save(os.path.join(output_folder, 'X_train_indices.npy'), np.arange(len(X))[train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42, stratify=y_encoded
    )[0]])
    np.save(os.path.join(output_folder, 'X_test_indices.npy'), np.arange(len(X))[train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42, stratify=y_encoded
    )[1]])

    print(f"All outputs saved to {output_folder}")


# Run the main program
if __name__ == "__main__":
    main()
