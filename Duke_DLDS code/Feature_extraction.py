import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pydicom
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from torchvision import models
import pickle
import time
from PIL import Image
import random
import torchvision.transforms.functional as TF


class FeatureExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        print(f"Initializing FeatureExtractor on device: {device}")
        self.device = device

        print("Loading DenseNet121 model...")
        self.densenet = models.densenet121(weights='IMAGENET1K_V1')
        self.densenet.classifier = nn.Identity()  # Remove classification layer
        self.densenet.to(device)
        self.densenet.eval()
        print("✓ DenseNet121 loaded successfully")

        print("Loading ResNet50 model...")
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.resnet.fc = nn.Identity()  # Remove classification layer
        self.resnet.to(device)
        self.resnet.eval()
        print("✓ ResNet50 loaded successfully")
        print("Feature extractor initialized and ready")

    def extract_texture_features(self, image):
        """Extract GLCM texture features from grayscale image"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = cv2.resize(image, (224, 224))
        # Quantize image to 16 levels (0-15)
        image_norm = (image * 15).astype(np.uint8)  # Changed from 255 to 15

        distances = [1, 3, 5]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        glcm = graycomatrix(image_norm, distances=distances, angles=angles,
                            levels=16, symmetric=True, normed=True)

        features = {}
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        for prop in properties:
            features[prop] = graycoprops(glcm, prop).mean()

        return features

    def extract_statistical_features(self, image):
        """Extract statistical features from image"""
        if image.max() > 1.0:
            image = image / 255.0

        features = {
            'mean': np.mean(image),
            'std': np.std(image),
            'variance': np.var(image),
            'min': np.min(image),
            'max': np.max(image),
            'median': np.median(image),
            'skewness': self._calculate_skewness(image),
            'kurtosis': self._calculate_kurtosis(image)
        }

        hist, bins = np.histogram(image.flatten(), bins=10, range=(0, 1))
        hist = hist / np.sum(hist)  # Normalize
        for i, h in enumerate(hist):
            features[f'hist_bin_{i}'] = h

        return features

    def _calculate_skewness(self, image):
        """Calculate skewness of image intensity"""
        mean = np.mean(image)
        std = np.std(image)
        if std == 0:
            return 0
        skewness = np.mean(((image - mean) / std) ** 3)
        return skewness

    def _calculate_kurtosis(self, image):
        """Calculate kurtosis of image intensity"""
        mean = np.mean(image)
        std = np.std(image)
        if std == 0:
            return 0
        kurtosis = np.mean(((image - mean) / std) ** 4) - 3
        return kurtosis

    def extract_deep_features(self, image, model_name='densenet'):
        """Extract deep features using pre-trained models"""
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)

        if image.max() > 1.0:
            image = image / 255.0

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            if model_name == 'densenet':
                features = self.densenet(image_tensor)
            elif model_name == 'resnet':
                features = self.resnet(image_tensor)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

        features = features.cpu().numpy().flatten()

        return features

    def extract_all_features(self, image):
        """Extract all features from an image"""
        if image.max() > 1.0:
            image = image / 255.0

        texture_features = self.extract_texture_features(image)
        statistical_features = self.extract_statistical_features(image)

        densenet_features = self.extract_deep_features(image, 'densenet')
        resnet_features = self.extract_deep_features(image, 'resnet')

        all_features = {
            **texture_features,
            **statistical_features,
            'densenet_features': densenet_features,
            'resnet_features': resnet_features
        }

        return all_features


class DICOMDataset(Dataset):
    def __init__(self, root_dir, series_labels, transform=None, augment=False, extract_features=False):
        self.root_dir = root_dir
        self.series_labels = series_labels
        self.transform = transform
        self.augment = augment
        self.extract_features = extract_features
        self.samples = []
        self.feature_extractor = FeatureExtractor() if extract_features else None
        self.features_cache = {}  # Cache for extracted features

        for patient_id in tqdm(os.listdir(root_dir), desc="Loading dataset"):
            patient_dir = os.path.join(root_dir, patient_id)
            if not os.path.isdir(patient_dir):
                continue

            for series_id in os.listdir(patient_dir):
                series_dir = os.path.join(patient_dir, series_id)
                if not os.path.isdir(series_dir):
                    continue

                full_series_id = f"{patient_id}/{series_id}"
                if full_series_id not in series_labels:
                    continue

                dicom_files = []
                for file in os.listdir(series_dir):
                    file_path = os.path.join(series_dir, file)
                    if os.path.isfile(file_path):
                        dicom_files.append(file_path)

                if dicom_files:
                    middle_idx = len(dicom_files) // 2
                    self.samples.append((dicom_files[middle_idx], series_labels[full_series_id]))

    def __len__(self):
        return len(self.samples)

    def apply_augmentations(self, img):
        img_np = img.numpy().transpose(1, 2, 0)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            img_pil = TF.rotate(img_pil, angle)

        if random.random() > 0.5:
            img_pil = TF.hflip(img_pil)

        if random.random() > 0.5:
            img_pil = TF.vflip(img_pil)

        if random.random() > 0.5:
            brightness_factor = random.uniform(0.85, 1.15)
            img_pil = TF.adjust_brightness(img_pil, brightness_factor)

        if random.random() > 0.5:
            contrast_factor = random.uniform(0.85, 1.15)
            img_pil = TF.adjust_contrast(img_pil, contrast_factor)

        img_np = np.array(img_pil).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))

        return img_tensor

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            # Load DICOM file
            ds = pydicom.dcmread(img_path, force=True)

            # Convert pixel array to float32 and normalize properly
            img = ds.pixel_array.astype(np.float32)

            # Normalize to 0-255 range
            if img.max() != img.min():  # Avoid division by zero
                img = ((img - img.min()) / (img.max() - img.min())) * 255.0
            else:
                img = np.zeros_like(img)

            # Convert to uint8 for OpenCV compatibility
            img = img.astype(np.uint8)

            # Convert to 3 channels if grayscale
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=2)

            # Resize to 224x224
            img = cv2.resize(img, (224, 224))

            # Normalize to [0, 1] for neural network processing
            img = img.astype(np.float32) / 255.0

            # Extract features if needed
            if self.extract_features:
                if img_path not in self.features_cache:
                    self.features_cache[img_path] = self.feature_extractor.extract_all_features(img)
                features = self.features_cache[img_path]

            # Convert to PyTorch tensor
            img = torch.from_numpy(img).permute(2, 0, 1)

            # Apply augmentations if enabled
            if self.augment:
                img = self.apply_augmentations(img)

            # Apply any additional transformations
            if self.transform:
                img = self.transform(img)

            if self.extract_features:
                return img, label, features
            else:
                return img, label

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            img = torch.zeros((3, 224, 224), dtype=torch.float32)
            if self.extract_features:
                features = {'error': True}
                return img, label, features
            else:
                return img, label


def extract_and_analyze_features(dataset_path, labels_path, output_folder):
    print("\n" + "=" * 50)
    print("STARTING FEATURE EXTRACTION AND ANALYSIS")
    print("=" * 50)

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Load labels
    df = pd.read_csv(labels_path)
    series_labels = {f"{int(row['DLDS']):04d}/{row['Series']}": row['Label'] for _, row in df.iterrows()}

    # Create dataset
    dataset = DICOMDataset(dataset_path, series_labels, extract_features=True)

    # Feature extraction
    all_features = {}
    all_labels = []
    successful = 0
    errors = 0

    for idx in tqdm(range(len(dataset)), desc="Processing images"):
        try:
            _, label, features = dataset[idx]
            if 'error' in features:
                errors += 1
                continue

            all_features[idx] = features
            all_labels.append(label)
            successful += 1

        except Exception as e:
            errors += 1
            print(f"Error processing sample {idx}: {e}")

    # ========== CRUCIAL VALIDATION CHECKS ==========
    if successful == 0:
        raise RuntimeError(
            "All feature extractions failed! Check:\n"
            "1. DICOM file paths and permissions\n"
            "2. Image normalization in DICOMDataset\n"
            "3. GLCM parameters in FeatureExtractor"
        )

    # Create DataFrame
    feature_rows = []
    for idx, features in all_features.items():
        row = {k: v for k, v in features.items() if k not in ['densenet_features', 'resnet_features']}
        feature_rows.append(row)

    feature_df = pd.DataFrame(feature_rows)
    feature_df['label'] = all_labels

    if feature_df.empty:
        raise ValueError("Feature DataFrame is empty - check extraction logic")

    # Save features
    feature_df.to_csv(os.path.join(output_folder, 'extracted_features.csv'), index=False)

    # Feature importance analysis
    X = feature_df.drop('label', axis=1)
    y = feature_df['label']

    # Convert labels to numeric indices
    label_to_idx = {label: idx for idx, label in enumerate(y.unique())}
    y_numeric = y.map(label_to_idx)

    # Feature selection
    selector = SelectKBest(f_classif, k=10)
    try:
        selector.fit(X, y_numeric)
    except ValueError as e:
        raise RuntimeError(f"Feature selection failed: {e}\nCheck for NaN/inf values in features") from e

    # Save results
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': selector.scores_
    }).sort_values('Importance', ascending=False)

    # Visualization
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance['Feature'][:15], feature_importance['Importance'][:15])
    plt.savefig(os.path.join(output_folder, 'feature_importance.png'))

    return feature_importance

def main():
    print("\n" + "*" * 70)
    print("*" + " " * 24 + "FEATURE EXTRACTION" + " " * 24 + "*")
    print("*" + " " * 18 + "FOR MEDICAL IMAGE CLASSIFICATION" + " " * 18 + "*")
    print("*" * 70)

    # Configuration parameters - update these paths for your environment
    dataset_path = r'C:\Softwares\All Programs\HCA\Duke_DLDS\Series_Classification'  # Path to your DICOM dataset
    labels_path = r'C:\Softwares\All Programs\HCA\Duke_DLDS\SeriesClassificationKey.csv'  # Path to your CSV with labels
    output_folder = r'C:\Softwares\All Programs\HCA\Duke_DLDS\Feature_extraction'  # Where to save results

    print(f"\nConfiguration:")
    print(f"  Dataset path: {dataset_path}")
    print(f"  Labels path: {labels_path}")
    print(f"  Output folder: {output_folder}")

    # Check if paths exist
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return

    if not os.path.exists(labels_path):
        print(f"ERROR: Labels path does not exist: {labels_path}")
        return

    print("\nStarting feature extraction and analysis process...")

    # Record start time
    start_time = time.time()

    # Extract and analyze features
    feature_importance = extract_and_analyze_features(dataset_path, labels_path, output_folder)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\nProcess completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

    print("\nTop 10 Most Important Features:")
    for i, (feature, importance) in enumerate(zip(feature_importance['Feature'][:10],
                                                  feature_importance['Importance'][:10])):
        print(f"{i + 1}. {feature}: {importance:.4f}")

    print("\nYou can find all results in the output folder:")
    print(f"  {output_folder}")


if __name__ == "__main__":
    main()
