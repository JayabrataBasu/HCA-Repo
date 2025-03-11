import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pydicom
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
root_dir = r'C:\Users\jayab\Duke_DLDS\Series_Classification\Series_Classification'
csv_path = r'C:\Users\jayab\Duke_DLDS\SeriesClassificationKey.csv'
output_folder = r'C:\Users\jayab\Duke_DLDS\Densenet_output'

# Create output directory
os.makedirs(output_folder, exist_ok=True)


# Load series labels from CSV
def load_series_labels(csv_path):
    df = pd.read_csv(csv_path)
    series_labels = {}

    for _, row in df.iterrows():
        # Format patient ID with leading zeros
        patient_id = f"{int(row['DLDS']):04d}"
        series_id = f"{patient_id}/{row['Series']}"
        series_labels[series_id] = row['Label']

    return series_labels


# Custom Dataset for DICOM images
class DICOMDataset(Dataset):
    def __init__(self, root_dir, series_labels, transform=None):
        self.root_dir = root_dir
        self.series_labels = series_labels
        self.transform = transform
        self.samples = []

        # Collect all valid samples
        for patient_id in tqdm(os.listdir(root_dir), desc="Loading dataset"):
            patient_dir = os.path.join(root_dir, patient_id)
            if not os.path.isdir(patient_dir):
                continue

            for series_id in os.listdir(patient_dir):
                series_dir = os.path.join(patient_dir, series_id)
                if not os.path.isdir(series_dir):
                    continue

                # Check if this series has a label
                full_series_id = f"{patient_id}/{series_id}"
                if full_series_id not in series_labels:
                    continue

                # Find all DICOM files in this series
                dicom_files = []
                for file in os.listdir(series_dir):
                    file_path = os.path.join(series_dir, file)
                    if os.path.isfile(file_path):
                        dicom_files.append(file_path)

                if dicom_files:
                    # Use the middle slice as representative of the series
                    middle_idx = len(dicom_files) // 2
                    self.samples.append((dicom_files[middle_idx], series_labels[full_series_id]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            # Load DICOM file
            ds = pydicom.dcmread(img_path, force=True)
            img = ds.pixel_array

            # Convert to 3 channels (DenseNet expects 3 channels)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=2)

            # Resize to 224x224 (standard input size for many CNNs)
            img = cv2.resize(img, (224, 224))

            # Normalize to [0, 1]
            img = img / np.max(img) if np.max(img) > 0 else img

            # Convert to PyTorch tensor and permute dimensions to [C, H, W]
            img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)

            # Apply any additional transformations
            if self.transform:
                img = self.transform(img)

            return img, label

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            # Return a blank image in case of error
            img = torch.zeros((3, 224, 224), dtype=torch.float32)
            return img, label


# DenseNet architecture components
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        if self.drop_rate > 0:
            new_features = nn.functional.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=17):
        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def densenet121(num_classes=17):
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=num_classes)


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            # Convert string labels to indices
            label_indices = torch.tensor([label_to_idx[label] for label in labels], dtype=torch.long)

            inputs = inputs.to(device)
            label_indices = label_indices.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, label_indices)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += label_indices.size(0)
            correct += (predicted == label_indices).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                # Convert string labels to indices
                label_indices = torch.tensor([label_to_idx[label] for label in labels], dtype=torch.long)

                inputs = inputs.to(device)
                label_indices = label_indices.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, label_indices)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += label_indices.size(0)
                val_correct += (predicted == label_indices).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

        # Save the best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), os.path.join(output_folder, 'best_model.pth'))

    # Save the final model
    torch.save(model.state_dict(), os.path.join(output_folder, 'final_model.pth'))

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'training_history.png'))

    return model, history


# Evaluate the model
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            # Convert string labels to indices
            label_indices = torch.tensor([label_to_idx[label] for label in labels], dtype=torch.long)

            inputs = inputs.to(device)
            label_indices = label_indices.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label_indices.cpu().numpy())

    # Convert indices back to labels for the report
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    pred_labels = [idx_to_label[i] for i in all_preds]
    true_labels = [idx_to_label[i] for i in all_labels]

    # Generate classification report
    report = classification_report(true_labels, pred_labels)
    with open(os.path.join(output_folder, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(label_to_idx.keys()),
                yticklabels=sorted(label_to_idx.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'))

    return report, cm


def main():
    # Load series labels
    print("Loading series labels from CSV...")
    series_labels = load_series_labels(csv_path)
    print(f"Loaded {len(series_labels)} series labels.")

    # Create dataset
    print("Creating dataset...")
    dataset = DICOMDataset(root_dir, series_labels)
    print(f"Dataset created with {len(dataset)} samples.")

    # Create label to index mapping
    global label_to_idx
    unique_labels = sorted(set(label for _, label in dataset.samples))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    # Save label mapping
    with open(os.path.join(output_folder, 'label_mapping.txt'), 'w') as f:
        for label, idx in label_to_idx.items():
            f.write(f"{label}: {idx}\n")

    # Split dataset into train, validation, and test sets
    train_idx, test_idx = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42,
        stratify=[label for _, label in dataset.samples]
    )

    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.25, random_state=42,  # 0.25 * 0.8 = 0.2 of original data
        stratify=[dataset.samples[i][1] for i in train_idx]
    )

    # Create data loaders
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    train_loader = DataLoader(dataset, batch_size=8, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=8, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=8, sampler=test_sampler)

    # Initialize model
    print("Initializing DenseNet-121 model...")
    model = densenet121(num_classes=len(label_to_idx))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train the model
    print("Starting training...")
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=15, device=device
    )

    # Load the best model for evaluation
    model.load_state_dict(torch.load(os.path.join(output_folder, 'best_model.pth')))

    # Evaluate the model
    print("Evaluating model...")
    report, cm = evaluate_model(model, test_loader, device=device)
    print("Evaluation complete. Results saved to output folder.")


if __name__ == "__main__":
    main()
