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
root_dir = r'C:\Users\jayab\Duke_DLDS\Series_Classification\Series_Classification'
csv_path = r'C:\Users\jayab\Duke_DLDS\SeriesClassificationKey.csv'
output_folder = r'C:\Users\jayab\Duke_DLDS\GradCAM_output'

os.makedirs(output_folder, exist_ok=True)

# Load labels from CSV file
def load_series_labels(csv_path):
    df = pd.read_csv(csv_path)
    series_labels = {}
    for _, row in df.iterrows():
        patient_id = f"{int(row['DLDS']):04d}"
        series_id = f"{patient_id}/{row['Series']}"
        series_labels[series_id] = row['Label']
    return series_labels

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

                dicom_files = [f for f in os.listdir(series_dir) if f.endswith('.dicom') or f.endswith('.dcm')]
                if len(dicom_files) == 0:
                    continue

                middle_file = dicom_files[len(dicom_files) // 2]
                img_path = os.path.join(series_dir, middle_file)
                self.samples.append((img_path, labels_dict[full_series_id]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        ds = pydicom.dcmread(img_path)
        img = ds.pixel_array.astype(np.float32)

        # Normalize image to [0,1]
        img -= np.min(img)
        img /= np.max(img)

        # Convert grayscale to RGB (3 channels)
        img_rgb = np.stack([img] * 3, axis=-1)

        # Resize image to 224x224 pixels (standard size)
        img_rgb = cv2.resize(img_rgb, (224, 224))

        if self.transform:
            img_rgb = self.transform(img_rgb)

        return img_rgb, label_to_idx[label]

# Data transforms and loaders
transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

dataset = MRIDataset(root_dir=root_dir, labels_dict=labels_dict,
                     transform=transform_pipeline)

train_indices, test_indices = train_test_split(
    range(len(dataset)), test_size=0.2,
    stratify=[sample[1] for sample in dataset.samples],
    random_state=42)

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

# Training loop
model.train()
for epoch in range(5):
    total_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/5"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), os.path.join(output_folder, 'resnet50.pth'))

# Evaluation and ROC-AUC calculation
model.eval()
all_preds_probs, all_true_labels = [], []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs.cpu(), dim=1).numpy()
        all_preds_probs.extend(probs)
        all_true_labels.extend(labels.numpy())

roc_auc_ovr = roc_auc_score(all_true_labels, all_preds_probs, multi_class='ovr')
print(f"Overall ROC-AUC Score: {roc_auc_ovr:.4f}")

# Grad-CAM visualization
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

for i, (img, label) in enumerate(test_loader):
    grayscale_cam = cam(input_tensor=img.to(device))[0, :]
    rgb_img = np.transpose(img[0].cpu().numpy(), (1, 2, 0))
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    visualization = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.title(f"GradCAM: {class_names[label[0]]}")
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, f"gradcam_{i}_{class_names[label[0]]}.png"))
    if i >= 19:
        break
