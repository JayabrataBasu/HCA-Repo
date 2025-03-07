import os
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import cv2

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

# Define paths
base_path = r'C:\Users\jayab\Duke_DLDS'
segmentation_path = os.path.join(base_path, 'Segmentation', 'Segmentation')
output_dir = os.path.join(base_path, 'alexnet_output')
os.makedirs(output_dir, exist_ok=True)

# Create timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load the segmentation key to get valid combinations
key_df = pd.read_csv(os.path.join(base_path, 'SegmentationKey.csv'))

# Create a list of valid patient/series combinations
valid_combinations = []
for _, row in key_df.iterrows():
    patient_id = f"{int(row['DLDS']):04d}"
    series_id = str(row['Series'])
    valid_combinations.append((patient_id, series_id))

print(f"Found {len(valid_combinations)} valid patient/series combinations")


# Function to load and preprocess DICOM images
def load_and_preprocess_image(patient_id, series_id):
    image_folder = os.path.join(segmentation_path, patient_id, series_id, 'images')

    if not os.path.exists(image_folder):
        return None, None

    try:
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.dicom')])
        if not image_files:
            return None, None

        # Use middle slice for better representation
        middle_idx = len(image_files) // 2
        image_path = os.path.join(image_folder, image_files[middle_idx])

        # Read DICOM
        dicom = pydicom.dcmread(image_path)
        image = dicom.pixel_array

        # Normalize to 0-255
        image = ((image - image.min()) * 255 / (image.max() - image.min())).astype(np.uint8)

        # Convert to 3-channel (RGB) by repeating the grayscale image
        image_rgb = np.stack([image] * 3, axis=2)

        # Resize to AlexNet input size (224x224)
        image_resized = cv2.resize(image_rgb, (224, 224))

        return image_resized, image_files[middle_idx]
    except Exception as e:
        print(f"Error loading image for patient {patient_id}, series {series_id}: {e}")
        return None, None


# Custom Dataset class
class LiverDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Data augmentation and normalization
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
])

# Load and prepare data
print("Loading and preprocessing images...")
images = []
labels = []
metadata = []

# For demonstration, we'll use a simple binary classification:
# 1 = Has fibrosis features (based on having valid segmentation)
# 0 = No/minimal fibrosis features

# Track patients with valid segmentation (from radiomics results)
valid_segmentation = ['0002_7', '0002_8', '0081_10']

for patient_id, series_id in tqdm(valid_combinations):
    image, slice_id = load_and_preprocess_image(patient_id, series_id)

    if image is not None:
        images.append(image)

        # Assign label based on whether this patient/series had valid segmentation
        patient_series = f"{patient_id}_{series_id}"
        label = 1 if patient_series in valid_segmentation else 0
        labels.append(label)

        metadata.append({
            'patient_id': patient_id,
            'series_id': series_id,
            'slice_id': slice_id,
            'label': label
        })

print(f"Loaded {len(images)} images")

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Training set: {X_train.shape[0]} images")
print(f"Validation set: {X_val.shape[0]} images")

# Create datasets
train_dataset = LiverDataset(X_train, y_train, transform=data_transforms)
val_dataset = LiverDataset(X_val, y_val, transform=data_transforms)

# Create dataloaders with small batch size for CPU
batch_size = 4  # Small batch size for CPU
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize AlexNet with pretrained weights
model = models.alexnet(pretrained=True)

# Modify the classifier for binary classification
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 1)

# Use sigmoid for binary classification
model.classifier.add_module('7', nn.Sigmoid())

# Use CPU for training
device = torch.device('cpu')
model = model.to(device)

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Reduce learning rate when plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': []
    }

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.float().to(device).view(-1, 1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        running_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.float().to(device).view(-1, 1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = running_loss / len(val_loader.dataset)

        # Calculate AUC
        val_auc = roc_auc_score(val_labels, val_preds)

        # Update learning rate
        scheduler.step(epoch_val_loss)

        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_auc'].append(val_auc)

        print(f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val AUC: {val_auc:.4f}")

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, f'alexnet_best_model_{timestamp}.pth'))
            print("Saved best model")

    return model, history


# Train the model with fewer epochs for CPU
print("Starting training...")
num_epochs = 5  # Reduced for CPU training
model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs)

# Save final model
torch.save(model.state_dict(), os.path.join(output_dir, f'alexnet_final_model_{timestamp}.pth'))

# Save training history
history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(output_dir, f'training_history_{timestamp}.csv'), index=False)

# Plot training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['val_auc'], label='Validation AUC')
plt.title('AUC Curve')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'training_curves_{timestamp}.png'))

# Evaluate model on validation set
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.numpy())

# Convert to binary predictions
binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]

# Calculate confusion matrix
cm = confusion_matrix(all_labels, binary_preds)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0, 1], ['No Fibrosis', 'Fibrosis'])
plt.yticks([0, 1], ['No Fibrosis', 'Fibrosis'])

# Add text annotations
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'confusion_matrix_{timestamp}.png'))

# Save metadata with predictions
metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(os.path.join(output_dir, f'metadata_{timestamp}.csv'), index=False)

print(f"Processing complete! All outputs saved to {output_dir}")
