import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import pydicom
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms


# Set paths
root_dir = r'C:\Softwares\All Programs\HCA\Duke_DLDS\Series_Classification\Series_Classification'
csv_path = r'C:\Softwares\All Programs\HCA\Duke_DLDS\SeriesClassificationKey.csv'
output_folder = r'C:\Softwares\All Programs\HCA\Duke_DLDS\ViT_improved_output'

# Create output directory
os.makedirs(output_folder, exist_ok=True)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load series labels
def load_series_labels(csv_path):
    df = pd.read_csv(csv_path)
    return {f"{int(row['DLDS']):04d}/{row['Series']}": row['Label'] for _, row in df.iterrows()}

# Enhanced DICOM Dataset with Advanced Augmentations
class DICOMDataset(Dataset):
    def __init__(self, root_dir, series_labels, transform=None):
        self.transform = transform
        self.samples = []
        for patient_id in tqdm(os.listdir(root_dir), desc="Loading dataset"):
            patient_dir = os.path.join(root_dir, patient_id)
            if os.path.isdir(patient_dir):
                for series_id in os.listdir(patient_dir):
                    full_id = f"{patient_id}/{series_id}"
                    if full_id in series_labels:
                        series_dir = os.path.join(patient_dir, series_id)
                        dicom_files = [os.path.join(series_dir, f) for f in os.listdir(series_dir)
                                       if os.path.isfile(os.path.join(series_dir, f))]
                        if dicom_files:
                            self.samples.append((dicom_files[len(dicom_files) // 2], series_labels[full_id]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            dicom_file = self.samples[idx][0]
            img = pydicom.dcmread(dicom_file).pixel_array
            img = cv2.resize(np.stack([img] * 3, axis=2) if img.ndim == 2 else img, (224, 224))
            img = torch.from_numpy(img.astype(np.float32) / np.max(img))  # Fix: Use .astype() instead of .ast()
            img = img.permute(2, 0, 1)  # Change to (C, H, W) format
            if self.transform:
                img = self.transform(img)
            return img, self.samples[idx][1]
        except Exception as e:
            print(f"Error loading image {self.samples[idx][0]}: {e}")
            return torch.zeros((3, 224, 224)), self.samples[idx][1]

# Improved Vision Transformer Components
class ShiftedPatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_chans=3, embed_dim=512):
        super().__init__()
        self.shift = patch_size // 2
        self.proj_main = nn.Conv2d(in_chans, embed_dim // 2, kernel_size=patch_size, stride=patch_size)
        self.proj_shift_x = nn.Conv2d(in_chans, embed_dim // 8, kernel_size=patch_size, stride=patch_size)
        self.proj_shift_y = nn.Conv2d(in_chans, embed_dim // 8, kernel_size=patch_size, stride=patch_size)
        self.proj_shift_xy = nn.Conv2d(in_chans, embed_dim // 8, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Linear(embed_dim // 2 + 3 * (embed_dim // 8), embed_dim)  # Final projection

    def forward(self, x):
        # Original projection
        main = self.proj_main(x)

        # Shifted projections
        sx = self.proj_shift_x(x[:, :, self.shift:, self.shift:])
        sy = self.proj_shift_y(x[:, :, :-self.shift, :-self.shift])
        sxy = self.proj_shift_xy(x[:, :, self.shift:, :-self.shift])

        # Pad shifted patches to match the size of the main patch
        sx = torch.nn.functional.pad(sx, (0, 1, 0, 1))  # Pad to match dimensions
        sy = torch.nn.functional.pad(sy, (0, 1, 0, 1))  # Pad to match dimensions
        sxy = torch.nn.functional.pad(sxy, (0, 1, 0, 1))  # Pad to match dimensions

        # Concatenate all patches
        x = torch.cat([main, sx, sy, sxy], dim=1).flatten(2).transpose(1, 2)
        return self.proj(x)  # Project to embed_dim


class LocalitySelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gamma = nn.Parameter(torch.ones(1))
        self.register_buffer("base_mask", ~torch.eye(197).bool())  # For 224x224 images

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, **kwargs):
        # Dynamically create the mask based on the input batch size
        batch_size = query.size(0)
        mask = self.base_mask.unsqueeze(0).expand(batch_size, -1, -1).to(query.device)

        # Pass the mask to the attention module
        attn_out, _ = self.attn(
            query,
            key,
            value,
            attn_mask=mask,
            **kwargs
        )
        return query + self.gamma * attn_out
class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=17, embed_dim=512, depth=12,
                 num_heads=8, mlp_ratio=3., dropout=0.2):
        super().__init__()
        self.patch_embed = ShiftedPatchEmbedding(image_size, patch_size, 3, embed_dim)
        num_patches = (image_size // patch_size) ** 2

        # 2D Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer with Locality Self-Attention
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(embed_dim, num_heads, int(embed_dim * mlp_ratio), dropout,
                                       activation='gelu', batch_first=True)
            for _ in range(depth)
        ])
        for block in self.blocks:
            block.self_attn = LocalitySelfAttention(embed_dim, num_heads)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        return self.head(self.norm(x[:, 0]))

# Training Infrastructure
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, device='cuda', epochs=30, label_to_idx=None):
    model.to(device)
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    for epoch in range(epochs):
        # Training
        model.train()
        loss_sum, correct, total = 0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), torch.tensor([label_to_idx[l] for l in labels]).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), torch.tensor([label_to_idx[l] for l in labels]).to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                val_correct += (torch.max(outputs, 1)[1] == labels).sum().item()
                val_total += labels.size(0)

        # Update metrics
        train_loss = loss_sum / len(train_loader.dataset)
        train_acc = correct / total
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch + 1}: "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")

        if scheduler:
            scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_folder, 'best_model.pth'))

    torch.save(model.state_dict(), os.path.join(output_folder, 'final_model.pth'))
# Evaluation and Visualization
def visualize_attention(model, test_loader, device='cuda', num_images=5):
    model.eval()
    os.makedirs(os.path.join(output_folder, 'attention'), exist_ok=True)
    inputs, labels = next(iter(test_loader))

    for i in range(min(num_images, len(inputs))):
        img = inputs[i].unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(model.blocks[0], 'self_attn'):
                attns = [block.self_attn(img)[1] for block in model.blocks]  # Get attention weights
            else:
                attns = []

        plt.figure(figsize=(20, 10))
        plt.subplot(1, len(attns) + 1, 1)
        plt.imshow(inputs[i].permute(1, 2, 0).cpu().numpy())
        plt.title(f"Original\nTrue: {labels[i]}")

        for j, attn in enumerate(attns):
            plt.subplot(1, len(attns) + 1, j + 2)
            plt.imshow(attn[0, 0, 1:].reshape(14, 14).cpu().numpy(), cmap='jet')
            plt.title(f"Layer {j + 1} Attention")

        plt.savefig(os.path.join(output_folder, 'attention', f'attn_{i}.png'))
        plt.close()

def main():
    # Data Preparation
    series_labels = load_series_labels(csv_path)
    dataset = DICOMDataset(root_dir, series_labels, transform=True)
    label_to_idx = {l: i for i, l in enumerate(sorted(set(dataset.samples[i][1] for i in range(len(dataset)))))}

    # Split dataset
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=[s[1] for s in dataset.samples])
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25,
                                          stratify=[dataset.samples[i][1] for i in train_idx])

    # Create loaders
    train_loader = DataLoader(dataset, batch_size=16, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    val_loader = DataLoader(dataset, batch_size=16, sampler=torch.utils.data.SubsetRandomSampler(val_idx))
    test_loader = DataLoader(dataset, batch_size=16, sampler=torch.utils.data.SubsetRandomSampler(test_idx))

    # Model Configuration
    model = VisionTransformer(
        num_classes=len(label_to_idx),
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=3.0,
        dropout=0.2
    ).to(device)

    # Layer-specific Optimization
    optimizer = optim.AdamW([
        {'params': model.patch_embed.parameters(), 'weight_decay': 0.03},
        {'params': model.blocks.parameters(), 'weight_decay': 0.05},
        {'params': model.head.parameters(), 'weight_decay': 0.01}
    ], lr=0.0001)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Training
    model, history, best_acc = train_model(
        model, train_loader, val_loader, nn.CrossEntropyLoss(),
        optimizer, scheduler, device, epochs=30, label_to_idx=label_to_idx
    )

    # Evaluation
    model.load_state_dict(torch.load(os.path.join(output_folder, 'best_model.pth')))
    visualize_attention(model, test_loader, device)

if __name__ == "__main__":
    main()