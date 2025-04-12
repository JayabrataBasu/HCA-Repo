import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pydicom
import cv2
import random
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score,f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.models as models
from itertools import cycle

# Set paths
root_dir = r'C:\Softwares\All Programs\HCA\Duke_DLDS\Series_Classification'
csv_path = r'C:\Softwares\All Programs\HCA\Duke_DLDS\SeriesClassificationKey.csv'
output_folder = r'C:\Softwares\All Programs\HCA\Duke_DLDS\Ensemble_new_output'

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
    def apply_augmentations(self, img):
        """Apply various augmentations to the image tensor"""
        # Convert tensor to PIL image for transformations
        img_np = img.numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

        # Random rotation (±15°) - simulates different patient positioning
        angle = random.uniform(-15, 15)
        img_pil = TF.rotate(img_pil, angle)

        # Random horizontal flip - simulates different viewing perspectives
        img_pil = TF.hflip(img_pil)

        # Random vertical flip - provides additional orientation variation
        img_pil = TF.vflip(img_pil)

        # Random brightness adjustment - simulates different exposure levels
        brightness_factor = random.uniform(0.85, 1.15)
        img_pil = TF.adjust_brightness(img_pil, brightness_factor)

        # Random contrast adjustment - simulates different tissue contrast
        contrast_factor = random.uniform(0.85, 1.15)
        img_pil = TF.adjust_contrast(img_pil, contrast_factor)

        # Convert back to tensor
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))  # [H, W, C] -> [C, H, W]

        return img_tensor
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            # Load DICOM file
            ds = pydicom.dcmread(img_path, force=True)
            img = ds.pixel_array

            # Convert to 3 channels (models expect 3 channels)
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


# Create ensemble model class
class EnsembleModel(nn.Module):
    def __init__(self, models, num_classes):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.classifier = nn.Linear(len(models) * num_classes, num_classes)

    def forward(self, x):
        # Get outputs from all models
        outputs = [model(x) for model in self.models]

        # Concatenate outputs
        concat_outputs = torch.cat(outputs, dim=1)

        # Final classification layer
        return self.classifier(concat_outputs)


# Create attention-based ensemble model class
class AttentionEnsembleModel(nn.Module):
    def __init__(self, models, num_classes):
        super(AttentionEnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.num_classes = num_classes

        # Attention mechanism for model weighting
        self.attention = nn.Sequential(
            nn.Linear(self.num_models * num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_models),
            nn.Softmax(dim=1)
        )

        # Final classifier for feature-level fusion
        self.classifier = nn.Sequential(
            nn.Linear(self.num_models * num_classes, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Get outputs from all models
        individual_outputs = [model(x) for model in self.models]

        # Stack outputs for attention mechanism
        stacked_outputs = torch.stack(individual_outputs, dim=1)  # [batch, num_models, num_classes]
        batch_size = stacked_outputs.size(0)

        # Concatenate all outputs for attention mechanism input
        concat_outputs = torch.cat(individual_outputs, dim=1)  # [batch, num_models*num_classes]

        # Generate attention weights
        attention_weights = self.attention(concat_outputs)  # [batch, num_models]

        # Apply attention weights to individual model outputs
        weighted_outputs = stacked_outputs * attention_weights.unsqueeze(-1)  # [batch, num_models, num_classes]

        # For weighted voting approach - weighted sum of softmax probabilities
        weighted_voting = torch.sum(
            torch.softmax(stacked_outputs, dim=2) * attention_weights.unsqueeze(-1),
            dim=1
        )  # [batch, num_classes]

        # For feature-level fusion - process concatenated features through classifier
        feature_fusion = self.classifier(concat_outputs)  # [batch, num_classes]

        # Return both results for potential ensemble
        return feature_fusion, weighted_voting, attention_weights


# Function to create individual models
def create_model(model_name, num_classes):
    if model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


# Training function for individual models
def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, num_epochs=20, device='cuda'):
    model.to(device)
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Set gradient accumulation steps for resource-intensive models
    accumulation_steps = 4 if model_name == 'vit_b_16' else 1

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Reset gradients at the beginning of each epoch
        optimizer.zero_grad()

        batch_count = 0

        for inputs, labels in tqdm(train_loader, desc=f"{model_name} - Epoch {epoch + 1}/{num_epochs} - Training"):
            # Convert string labels to indices
            label_indices = torch.tensor([label_to_idx[label] for label in labels], dtype=torch.long)

            inputs = inputs.to(device)
            label_indices = label_indices.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, label_indices) / accumulation_steps  # Normalize loss

            # Backward pass
            loss.backward()

            # Update weights after accumulation_steps or at the end of the epoch
            batch_count += 1
            if batch_count % accumulation_steps == 0 or batch_count == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            # Statistics (use original loss value for logging)
            running_loss += loss.item() * accumulation_steps * inputs.size(0)
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
            for inputs, labels in tqdm(val_loader, desc=f"{model_name} - Epoch {epoch + 1}/{num_epochs} - Validation"):
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

        print(f"{model_name} - Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

        # Save the best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), os.path.join(output_folder, f'best_{model_name}.pth'))

    # Save the final model
    torch.save(model.state_dict(), os.path.join(output_folder, f'final_{model_name}.pth'))

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{model_name}_training_history.png'))

    return model, history, best_val_acc


# Training function for ensemble model
def train_ensemble(ensemble_model, train_loader, val_loader, criterion, optimizer, num_epochs=5, device='cuda'):
    ensemble_model.to(device)
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # Training phase
        ensemble_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Ensemble - Epoch {epoch + 1}/{num_epochs} - Training"):
            # Convert string labels to indices
            label_indices = torch.tensor([label_to_idx[label] for label in labels], dtype=torch.long)

            inputs = inputs.to(device)
            label_indices = label_indices.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = ensemble_model(inputs)
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
        ensemble_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Ensemble - Epoch {epoch + 1}/{num_epochs} - Validation"):
                # Convert string labels to indices
                label_indices = torch.tensor([label_to_idx[label] for label in labels], dtype=torch.long)

                inputs = inputs.to(device)
                label_indices = label_indices.to(device)

                outputs = ensemble_model(inputs)
                loss = criterion(outputs, label_indices)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += label_indices.size(0)
                val_correct += (predicted == label_indices).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)

        print(f"Ensemble - Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

        # Save the best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(ensemble_model.state_dict(), os.path.join(output_folder, 'best_ensemble.pth'))

    # Save the final model
    torch.save(ensemble_model.state_dict(), os.path.join(output_folder, 'final_ensemble.pth'))

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Ensemble - Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Ensemble - Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'ensemble_training_history.png'))

    return ensemble_model, history, best_val_acc


# Training function for attention-based ensemble model
def train_attention_ensemble(ensemble_model, train_loader, val_loader, criterion, optimizer,
                             num_epochs=20, device='cuda', alpha=0.5):
    """
    Train the attention-based ensemble model

    Parameters:
    - ensemble_model: The attention-based ensemble model
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data
    - criterion: Loss function
    - optimizer: Optimizer
    - num_epochs: Number of epochs to train
    - device: Device to use (cuda/cpu)
    - alpha: Weight parameter for combining feature fusion and weighted voting losses
            (alpha=0 is only weighted voting, alpha=1 is only feature fusion)
    """

    ensemble_model.to(device)
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
               'fusion_acc': [], 'voting_acc': []}

    for epoch in range(num_epochs):
        # Training phase
        ensemble_model.train()
        running_loss = 0.0
        correct_fusion = 0
        correct_voting = 0
        total = 0

        # Track average attention weights for analysis
        epoch_attention_weights = torch.zeros(ensemble_model.num_models).to(device)
        batch_count = 0

        for inputs, labels in tqdm(train_loader,
                                   desc=f"Attention Ensemble - Epoch {epoch + 1}/{num_epochs} - Training"):
            # Convert string labels to indices
            label_indices = torch.tensor([label_to_idx[label] for label in labels], dtype=torch.long)

            inputs = inputs.to(device)
            label_indices = label_indices.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            fusion_output, voting_output, attention_weights = ensemble_model(inputs)

            # Calculate loss for both outputs and combine them
            fusion_loss = criterion(fusion_output, label_indices)
            voting_loss = criterion(voting_output, label_indices)
            loss = alpha * fusion_loss + (1 - alpha) * voting_loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)

            # Track accuracy for both outputs
            _, fusion_preds = torch.max(fusion_output, 1)
            _, voting_preds = torch.max(voting_output, 1)

            total += label_indices.size(0)
            correct_fusion += (fusion_preds == label_indices).sum().item()
            correct_voting += (voting_preds == label_indices).sum().item()

            # Track attention weights
            epoch_attention_weights += attention_weights.sum(dim=0)
            batch_count += 1

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_fusion_acc = correct_fusion / total
        epoch_voting_acc = correct_voting / total
        epoch_acc = max(epoch_fusion_acc, epoch_voting_acc)  # Use the better accuracy

        # Calculate average attention weights
        avg_attention = epoch_attention_weights / batch_count

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['fusion_acc'].append(epoch_fusion_acc)
        history['voting_acc'].append(epoch_voting_acc)

        # Validation phase
        ensemble_model.eval()
        val_loss = 0.0
        val_correct_fusion = 0
        val_correct_voting = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader,
                                       desc=f"Attention Ensemble - Epoch {epoch + 1}/{num_epochs} - Validation"):
                # Convert string labels to indices
                label_indices = torch.tensor([label_to_idx[label] for label in labels], dtype=torch.long)

                inputs = inputs.to(device)
                label_indices = label_indices.to(device)

                # Forward pass
                fusion_output, voting_output, _ = ensemble_model(inputs)

                # Calculate combined loss
                fusion_loss = criterion(fusion_output, label_indices)
                voting_loss = criterion(voting_output, label_indices)
                loss = alpha * fusion_loss + (1 - alpha) * voting_loss

                val_loss += loss.item() * inputs.size(0)

                # Track accuracy for both outputs
                _, fusion_preds = torch.max(fusion_output, 1)
                _, voting_preds = torch.max(voting_output, 1)

                val_total += label_indices.size(0)
                val_correct_fusion += (fusion_preds == label_indices).sum().item()
                val_correct_voting += (voting_preds == label_indices).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_fusion_acc = val_correct_fusion / val_total
        val_epoch_voting_acc = val_correct_voting / val_total
        val_epoch_acc = max(val_epoch_fusion_acc, val_epoch_voting_acc)  # Use the better accuracy

        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)

        # Print epoch results
        print(f"Attention Ensemble - Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Fusion Acc: {epoch_fusion_acc:.4f}, Voting Acc: {epoch_voting_acc:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, "
              f"Val Fusion Acc: {val_epoch_fusion_acc:.4f}, Val Voting Acc: {val_epoch_voting_acc:.4f}")

        # Print attention weights
        model_names = ['densenet121', 'resnet50', 'efficientnet_b0']
        print("Model attention weights:")
        for i, weight in enumerate(avg_attention.detach().cpu().numpy()):
            print(f"  {model_names[i]}: {weight:.4f}")

        # Save the best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(ensemble_model.state_dict(), os.path.join(output_folder, 'best_attention_ensemble.pth'))

            # Save attention weights
            weights_df = pd.DataFrame({
                'model': model_names,
                'weight': avg_attention.detach().cpu().numpy()
            })
            weights_df.to_csv(os.path.join(output_folder, 'best_attention_weights.csv'), index=False)

    # Save the final model
    torch.save(ensemble_model.state_dict(), os.path.join(output_folder, 'final_attention_ensemble.pth'))

    # Plot training history
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Attention Ensemble - Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Attention Ensemble - Overall Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(history['fusion_acc'], label='Fusion Accuracy')
    plt.plot(history['voting_acc'], label='Weighted Voting Accuracy')
    plt.title('Attention Ensemble - Training Methods Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Create a bar chart of final attention weights
    plt.subplot(2, 2, 4)
    plt.bar(model_names, avg_attention.detach().cpu().numpy())
    plt.title('Model Attention Weights')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'attention_ensemble_training_history.png'))

    return ensemble_model, history, best_val_acc


# Evaluate the model
def evaluate_model(model, model_name, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # Store probabilities for ROC curve

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            # Convert string labels to indices
            label_indices = torch.tensor([label_to_idx[label] for label in labels], dtype=torch.long)

            inputs = inputs.to(device)
            label_indices = label_indices.to(device)

            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label_indices.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # Convert indices back to labels for the report
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    pred_labels = [idx_to_label[i] for i in all_preds]
    true_labels = [idx_to_label[i] for i in all_labels]

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Calculate AUC-ROC
    all_probs = np.array(all_probs)
    all_labels_array = np.array(all_labels)
    n_classes = len(label_to_idx)

    # Binarize labels for ROC calculation
    y_bin = label_binarize(all_labels_array, classes=range(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Store ROC data for later plotting
    roc_data = {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'n_classes': n_classes
    }

    # Plot ROC curves
    plot_roc_curves(model_name, roc_data, label_to_idx)

    # Save metrics to file
    metrics_file_path = os.path.join(output_folder, f'{model_name}_metrics.txt')
    with open(metrics_file_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Micro-average AUC-ROC: {roc_auc['micro']:.4f}\n")
        f.write("\nAUC-ROC for each class:\n")
        for i in range(n_classes):
            f.write(f"{idx_to_label[i]}: {roc_auc[i]:.4f}\n")

    # Generate bar chart for metrics
    generate_metrics_barchart(model_name, accuracy, precision, recall, f1, roc_auc["micro"])

    # Generate classification report
    report = classification_report(true_labels, pred_labels)
    with open(os.path.join(output_folder, f'{model_name}_classification_report.txt'), 'w') as f:
        f.write(report)

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(label_to_idx.keys()),
                yticklabels=sorted(label_to_idx.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{model_name}_confusion_matrix.png'))

    # Return metrics with the report and confusion matrix
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': roc_auc["micro"],
        'roc_data': roc_data
    }

    return report, cm, metrics


# Evaluate the attention ensemble model
def evaluate_attention_ensemble(model, test_loader, device='cuda'):
    model.eval()
    all_fusion_preds = []
    all_voting_preds = []
    all_labels = []
    all_fusion_probs = []
    all_voting_probs = []
    all_attention_weights = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating Attention Ensemble"):
            # Convert string labels to indices
            label_indices = torch.tensor([label_to_idx[label] for label in labels], dtype=torch.long)

            inputs = inputs.to(device)
            label_indices = label_indices.to(device)

            fusion_output, voting_output, attention_weights = model(inputs)
            fusion_probs = torch.softmax(fusion_output, dim=1)
            voting_probs = voting_output  # Already contains weighted softmax probabilities

            _, fusion_predicted = torch.max(fusion_output, 1)
            _, voting_predicted = torch.max(voting_output, 1)

            all_fusion_preds.extend(fusion_predicted.cpu().numpy())
            all_voting_preds.extend(voting_predicted.cpu().numpy())
            all_labels.extend(label_indices.cpu().numpy())
            all_fusion_probs.extend(fusion_probs.cpu().numpy())
            all_voting_probs.extend(voting_probs.cpu().numpy())
            all_attention_weights.extend(attention_weights.cpu().numpy())

    # Convert indices back to labels for the report
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    fusion_pred_labels = [idx_to_label[i] for i in all_fusion_preds]
    voting_pred_labels = [idx_to_label[i] for i in all_voting_preds]
    true_labels = [idx_to_label[i] for i in all_labels]

    # Calculate metrics for both methods
    fusion_metrics = calculate_metrics("attention_ensemble_fusion", all_fusion_preds, all_labels, all_fusion_probs)
    voting_metrics = calculate_metrics("attention_ensemble_voting", all_voting_preds, all_labels, all_voting_probs)

    # Plot attention weight distribution
    plot_attention_weights(np.array(all_attention_weights))

    # Return the better of the two methods
    if fusion_metrics['accuracy'] >= voting_metrics['accuracy']:
        return "fusion", fusion_pred_labels, true_labels, fusion_metrics
    else:
        return "voting", voting_pred_labels, true_labels, voting_metrics


# Helper function to calculate all metrics
def calculate_metrics(model_name, all_preds, all_labels, all_probs):
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Calculate AUC-ROC
    all_probs = np.array(all_probs)
    all_labels_array = np.array(all_labels)
    n_classes = len(label_to_idx)

    # Binarize labels for ROC calculation
    y_bin = label_binarize(all_labels_array, classes=range(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Store ROC data for later plotting
    roc_data = {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'n_classes': n_classes
    }

    # Plot ROC curves
    plot_roc_curves(model_name, roc_data, label_to_idx)

    # Generate metrics barchart
    generate_metrics_barchart(model_name, accuracy, precision, recall, f1, roc_auc["micro"])

    # Return metrics dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': roc_auc["micro"],
        'roc_data': roc_data
    }


# Function to plot ROC curves
def plot_roc_curves(model_name, roc_data, label_to_idx):
    fpr = roc_data['fpr']
    tpr = roc_data['tpr']
    roc_auc = roc_data['roc_auc']
    n_classes = roc_data['n_classes']

    # Plot all ROC curves
    plt.figure(figsize=(12, 8))

    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    # Plot ROC curves for all classes
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'limegreen', 'purple', 'red',
                    'yellow', 'brown', 'pink', 'gray', 'olive', 'cyan'])

    idx_to_label = {v: k for k, v in label_to_idx.items()}

    for i, color in zip(range(n_classes), colors):
        if i in roc_auc:  # Check if class i exists in roc_auc
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {idx_to_label[i]} (AUC = {roc_auc[i]:.2f})')

    # Plot settings
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_folder, f'{model_name}_roc_curves.png'))
    plt.close()


# Function to plot attention weights
def plot_attention_weights(attention_weights):
    model_names = ['densenet121', 'resnet50', 'efficientnet_b0']

    plt.figure(figsize=(12, 10))

    # Plot average attention weights
    plt.subplot(2, 1, 1)
    avg_weights = attention_weights.mean(axis=0)
    plt.bar(model_names, avg_weights)
    plt.title('Average Attention Weights Across Test Dataset')
    plt.ylabel('Weight')
    plt.ylim(0, max(1.0, avg_weights.max() * 1.1))

    # Plot distribution of weights using violin plots
    plt.subplot(2, 1, 2)
    plt.violinplot([attention_weights[:, i] for i in range(attention_weights.shape[1])],
                   showmedians=True)
    plt.xticks(range(1, len(model_names) + 1), model_names)
    plt.title('Distribution of Attention Weights')
    plt.ylabel('Weight')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'attention_weights_analysis.png'))
    plt.close()


# Function to generate a bar chart of metrics
def generate_metrics_barchart(model_name, accuracy, precision, recall, f1, auc_roc):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
    values = [accuracy, precision, recall, f1, auc_roc]
    colors = ['blue', 'green', 'orange', 'red', 'purple']

    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics, values, color=colors)

    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f'{value:.4f}',
                 ha='center',
                 va='bottom')

    plt.title(f'{model_name} Performance Metrics')
    plt.ylim(0, 1.1)  # Set y-axis limit
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{model_name}_metrics_barchart.png'))
    plt.close()


# Function to generate a combined bar chart comparing all models
def generate_combined_metrics_barchart(metrics_results):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
    model_names = list(metrics_results.keys())

    # Setup the plot
    fig, ax = plt.figure(figsize=(15, 8)), plt.axes()

    # Calculate bar positions
    x = np.arange(len(metrics))
    width = 0.2  # Width of bars

    # Plot bars for each model
    for i, model_name in enumerate(model_names):
        model_metrics = [
            metrics_results[model_name]['accuracy'],
            metrics_results[model_name]['precision'],
            metrics_results[model_name]['recall'],
            metrics_results[model_name]['f1'],
            metrics_results[model_name]['auc_roc']
        ]

        # Create bars with offset
        bars = ax.bar(x + (i - len(model_names) / 2 + 0.5) * width,
                      model_metrics,
                      width,
                      label=model_name)

        # Add value labels on top of bars
        for bar, value in zip(bars, model_metrics):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{value:.4f}',
                    ha='center',
                    va='bottom',
                    fontsize=8)

    # Configure the plot
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'all_models_metrics_comparison.png'))
    plt.close()


# Function to plot combined ROC curves for all models
def plot_combined_roc_curves(metrics_results):
    plt.figure(figsize=(12, 8))

    colors = ['deeppink', 'blue', 'green', 'red']
    linestyles = ['-', '--', '-.', ':']

    for (model_name, metrics), color, linestyle in zip(metrics_results.items(), colors, linestyles):
        if 'roc_data' in metrics:
            roc_data = metrics['roc_data']
            # Plot micro-average ROC curve
            plt.plot(
                roc_data['fpr']["micro"],
                roc_data['tpr']["micro"],
                label=f'{model_name} (AUC = {metrics["auc_roc"]:.4f})',
                color=color,
                linestyle=linestyle,
                linewidth=2
            )

    # Plot random chance line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # Configure the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Model Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'combined_roc_curves.png'))
    plt.close()


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

    # Create data loaders with adjusted batch sizes

    # Create common data loaders with smallest batch size for test and validation
    val_loader = DataLoader(dataset, batch_size=4, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=4, sampler=test_sampler)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize individual models
    print("Initializing models...")
    model_names = ['densenet121', 'resnet50', 'efficientnet_b0']
    models = {}
    best_val_accs = {}

    # Train individual models
    metrics_results = {}  # Dictionary to store metrics for all models

    for model_name in model_names:
        print(f"\n=== Training {model_name} ===")
        model = create_model(model_name, len(label_to_idx))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
        model_train_loader = DataLoader(dataset,batch_size=8, sampler=train_sampler)



        try:
            model, history, best_val_acc = train_model(
                model, model_name, model_train_loader, val_loader, criterion, optimizer,
                num_epochs=10, device=device
            )

            models[model_name] = model
            best_val_accs[model_name] = best_val_acc

            # Evaluate individual model
            print(f"Evaluating {model_name}...")
            report, cm, metrics = evaluate_model(model, model_name, test_loader, device=device)
            metrics_results[model_name] = metrics

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"ERROR: GPU out of memory while training {model_name}. Skipping this model.")
                torch.cuda.empty_cache()  # Clear cache
            else:
                raise e

    # Create and train ensemble model
    print("\n=== Creating Ensemble Model ===")
    # Load the best version of each model
    ensemble_models = []
    for model_name in model_names:
        model = create_model(model_name, len(label_to_idx))
        model.load_state_dict(torch.load(os.path.join(output_folder, f'best_{model_name}.pth')))
        # Freeze individual model parameters
        for param in model.parameters():
            param.requires_grad = False
        ensemble_models.append(model)

    ensemble_model = EnsembleModel(ensemble_models, len(label_to_idx))

    # Create train loader for ensemble model with appropriate batch size
    ensemble_train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler)

    # Train only the ensemble classifier layer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ensemble_model.classifier.parameters(), lr=0.001)

    ensemble_model, ensemble_history, ensemble_best_val_acc = train_ensemble(
        ensemble_model, ensemble_train_loader, val_loader, criterion, optimizer,
        num_epochs=5, device=device
    )

    # Evaluate ensemble model
    print("Evaluating ensemble model...")
    report, cm, ensemble_metrics = evaluate_model(ensemble_model, "ensemble", test_loader, device=device)
    metrics_results["ensemble"] = ensemble_metrics

    # Create and train attention-based ensemble model
    print("\n=== Creating Attention-Based Ensemble Model ===")
    # Load the best version of each model
    attention_ensemble_models = []
    for model_name in model_names:
        model = create_model(model_name, len(label_to_idx))
        model.load_state_dict(torch.load(os.path.join(output_folder, f'best_{model_name}.pth')))
        # Freeze individual model parameters
        for param in model.parameters():
            param.requires_grad = False
        attention_ensemble_models.append(model)

    attention_ensemble = AttentionEnsembleModel(attention_ensemble_models, len(label_to_idx))

    # Train the attention ensemble model
    criterion = nn.CrossEntropyLoss()
    # Only train the attention mechanism and classifier
    trainable_params = list(attention_ensemble.attention.parameters()) + list(
        attention_ensemble.classifier.parameters())
    optimizer = optim.Adam(trainable_params, lr=0.001)

    attention_ensemble, attention_history, attention_best_val_acc = train_attention_ensemble(
        attention_ensemble, ensemble_train_loader, val_loader, criterion, optimizer,
        num_epochs=20, device=device, alpha=0.5  # Equally weight feature fusion and weighted voting
    )

    # Evaluate attention ensemble model
    print("Evaluating attention ensemble model...")
    best_method, pred_labels, true_labels, attention_metrics = evaluate_attention_ensemble(
        attention_ensemble, test_loader, device=device)
    metrics_results["attention_ensemble"] = attention_metrics

    # Compare performance of all models including the new attention-based ensemble
    print("\n=== Performance Comparison ===")
    print(
        f"DenseNet121: {best_val_accs['densenet121']:.4f}, Test Accuracy: {metrics_results['densenet121']['accuracy']:.4f}, AUC-ROC: {metrics_results['densenet121']['auc_roc']:.4f}")
    print(
        f"ResNet50: {best_val_accs['resnet50']:.4f}, Test Accuracy: {metrics_results['resnet50']['accuracy']:.4f}, AUC-ROC: {metrics_results['resnet50']['auc_roc']:.4f}")
    print(
        f"EfficientNet-B0: {best_val_accs['efficientnet_b0']:.4f}, Test Accuracy: {metrics_results['efficientnet_b0']['accuracy']:.4f}, AUC-ROC: {metrics_results['efficientnet_b0']['auc_roc']:.4f}")
    print(
        f"Basic Ensemble: {ensemble_best_val_acc:.4f}, Test Accuracy: {metrics_results['ensemble']['accuracy']:.4f}, AUC-ROC: {metrics_results['ensemble']['auc_roc']:.4f}")
    print(
        f"Attention Ensemble ({best_method}): {attention_best_val_acc:.4f}, Test Accuracy: {metrics_results['attention_ensemble']['accuracy']:.4f}, AUC-ROC: {metrics_results['attention_ensemble']['auc_roc']:.4f}")

    # Generate combined metrics bar chart for all models
    generate_combined_metrics_barchart(metrics_results)

    # Generate combined ROC curves for model comparison
    plot_combined_roc_curves(metrics_results)

    print("\nTraining and evaluation complete. All outputs saved to output folder.")


if __name__ == "__main__":
    main()