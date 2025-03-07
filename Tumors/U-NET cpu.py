import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np


# Define U-Net model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x_out = self.decoder(x1)
        return x_out


# Dataset class for loading images and masks
class LiverDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0

        # Add channel dimension (for PyTorch)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image), torch.tensor(mask)


# Training function
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)


# Validation function
def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()

    return val_loss / len(val_loader)


# Main script to train and validate the model
if __name__ == "__main__":
    # Directories for training and validation datasets
    train_image_dir = r"C:\Users\jayab\Downloads\processed_dataset\train\images"
    train_mask_dir = r"C:\Users\jayab\Downloads\processed_dataset\train\masks"
    val_image_dir = r"C:\Users\jayab\Downloads\processed_dataset\val\images"
    val_mask_dir = r"C:\Users\jayab\Downloads\processed_dataset\val\masks"

    # Hyperparameters
    batch_size = 4
    learning_rate = 1e-3
    num_epochs = 10

    # Device configuration (CPU-only system)
    device = torch.device("cpu")

    # Load datasets and create DataLoaders
    train_dataset = LiverDataset(train_image_dir, train_mask_dir)
    val_dataset = LiverDataset(val_image_dir, val_mask_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, loss function (BCE), and optimizer (Adam)
    model = UNet().to(device)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for segmentation tasks
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        val_loss = validate_model(model, val_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    print("Training complete.")

torch.save(model.state_dict(), "unet_model.pth")
print("Model saved as unet_model.pth.")
# Load the model for inference
model = UNet()
model.load_state_dict(torch.load("unet_model.pth"))
model.eval()
print("Model loaded for inference.")
# Example inference on a single image