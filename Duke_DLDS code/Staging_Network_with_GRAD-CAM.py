import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pydicom
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Paths (Update these based on your local paths)
root_dir = r'C:\Softwares\All Programs\HCA\Duke_DLDS\Series_Classification\Series_Classification'
csv_path = r'C:\Softwares\All Programs\HCA\Duke_DLDS\SeriesClassificationKey.csv'
output_folder = r'C:\Softwares\All Programs\HCA\Duke_DLDS\GradCAM_output'

os.makedirs(output_folder, exist_ok=True)


# Load labels from CSV file
def load_series_labels(csv_path):
    df = pd.read_csv(csv_path)
    return {f"{int(row['DLDS']):04d}/{row['Series']}": row['Label'] for _, row in df.iterrows()}


labels_dict = load_series_labels(csv_path)
class_names = sorted(list(set(labels_dict.values())))
label_to_idx = {label: idx for idx, label in enumerate(class_names)}


# Custom Dataset class for loading DICOM images
class MRIDataset(Dataset):
    def __init__(self, root_dir, labels_dict, transform=None):
        self.samples = []
        self.labels_dict = labels_dict
        self.transform = transform

        for patient_id in os.listdir(root_dir):
            patient_dir = os.path.join(root_dir, patient_id)
            if not os.path.isdir(patient_dir):
                continue

            for series_id in os.listdir(patient_dir):
                series_dir = os.path.join(patient_dir, series_id)
                full_series_id = f"{patient_id}/{series_id}"
                if full_series_id not in labels_dict:
                    continue

                dicom_files = [f for f in os.listdir(series_dir) if f.endswith(('.dicom', '.dcm'))]
                if dicom_files:
                    middle_file = dicom_files[len(dicom_files) // 2]
                    img_path = os.path.join(series_dir, middle_file)
                    self.samples.append((img_path, labels_dict[full_series_id]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        ds = pydicom.dcmread(img_path)
        img = ds.pixel_array.astype(np.float32)

        # Normalize and convert to RGB
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
        img_rgb = np.stack([img] * 3, axis=-1)
        img_rgb = cv2.resize(img_rgb, (224, 224))

        if self.transform:
            img_transformed = self.transform(img_rgb)
        else:
            img_transformed = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float()

        return img_transformed, label_to_idx[label], img_rgb  # Return original image for visualization


# Data transforms
transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = MRIDataset(root_dir=root_dir, labels_dict=labels_dict,
                     transform=transform_pipeline)

# Split dataset
train_indices, test_indices = train_test_split(
    range(len(dataset)), test_size=0.2,
    stratify=[dataset.samples[i][1] for i in range(len(dataset))],
    random_state=42
)

# Create data loaders
train_loader = DataLoader(dataset, batch_size=16,
                          sampler=torch.utils.data.SubsetRandomSampler(train_indices))
test_loader = DataLoader(dataset, batch_size=8,
                         sampler=torch.utils.data.SubsetRandomSampler(test_indices))

# Model definition (ResNet50 pretrained)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop with loss tracking
train_losses = []
model.train()
for epoch in range(5):
    epoch_loss = 0.0
    for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/5"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)

    avg_loss = epoch_loss / len(train_loader.dataset)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

# Save training curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), train_losses, marker='o')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(range(1, 6))
plt.savefig(os.path.join(output_folder, "training_loss.png"))
plt.close()

# Save model
torch.save(model.state_dict(), os.path.join(output_folder, 'resnet50.pth'))

# Evaluation and ROC-AUC calculation
model.eval()
all_preds_probs = []
all_true_labels = []

with torch.no_grad():
    for images, labels, _ in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs.cpu(), dim=1).numpy()
        all_preds_probs.extend(probs)
        all_true_labels.extend(labels.numpy())

roc_auc_ovr = roc_auc_score(all_true_labels, all_preds_probs, multi_class='ovr')
print(f"Overall ROC-AUC Score: {roc_auc_ovr:.4f}")

# Grad-CAM visualization
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

# Create Grad-CAM output directory
gradcam_dir = os.path.join(output_folder, "gradcam_results")
os.makedirs(gradcam_dir, exist_ok=True)

# Process first 20 test samples
for idx, (images, labels, orig_images) in enumerate(test_loader):
    if idx >= 20 // images.size(0):  # Process up to 20 samples
        break

    # Get Grad-CAM heatmaps
    grayscale_cams = cam(input_tensor=images.to(device))

    for i in range(images.size(0)):
        # Denormalize image
        img_np = orig_images[i].astype(np.float32) / 255.0

        # Get heatmap and overlay
        heatmap = grayscale_cams[i]
        visualization = show_cam_on_image(img_np, heatmap, use_rgb=True)

        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        ax1.imshow(img_np)
        ax1.set_title(f"Original\n{class_names[labels[i]]}")
        ax1.axis('off')

        # Heatmap
        ax2.imshow(heatmap, cmap='jet')
        ax2.set_title("Grad-CAM Heatmap")
        ax2.axis('off')

        # Overlay
        ax3.imshow(visualization)
        ax3.set_title("Grad-CAM Overlay")
        ax3.axis('off')

        # Save visualization
        plt.tight_layout()
        plt.savefig(os.path.join(gradcam_dir, f"gradcam_{idx * images.size(0) + i}_{class_names[labels[i]]}.png"))
        plt.close()

# Generate and save confusion matrix
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))
plt.close()
