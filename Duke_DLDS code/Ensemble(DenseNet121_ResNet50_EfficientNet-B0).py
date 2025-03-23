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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.models as models

# Set paths
root_dir = r'C:\Users\jayab\Duke_DLDS\Series_Classification\Series_Classification'
csv_path = r'C:\Users\jayab\Duke_DLDS\SeriesClassificationKey.csv'
output_folder = r'C:\Users\jayab\Duke_DLDS\Ensemble_output'

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
def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"{model_name} - Epoch {epoch + 1}/{num_epochs} - Training"):
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


# Evaluate the model
def evaluate_model(model, model_name, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
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

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Save metrics to file
    metrics_file_path = os.path.join(output_folder, f'{model_name}_metrics.txt')
    with open(metrics_file_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
    
    # Generate bar chart for metrics
    generate_metrics_barchart(model_name, accuracy, precision, recall, f1)

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
        'f1': f1
    }
    
    return report, cm, metrics


# Function to generate a bar chart of metrics
def generate_metrics_barchart(model_name, accuracy, precision, recall, f1):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy, precision, recall, f1]
    colors = ['blue', 'green', 'orange', 'red']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=colors)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, 
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

        model, history, best_val_acc = train_model(
            model, model_name, train_loader, val_loader, criterion, optimizer,
            num_epochs=10, device=device
        )

        models[model_name] = model
        best_val_accs[model_name] = best_val_acc

        # Evaluate individual model
        print(f"Evaluating {model_name}...")
        report, cm, metrics = evaluate_model(model, model_name, test_loader, device=device)
        metrics_results[model_name] = metrics
    
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

    # Train only the ensemble classifier layer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ensemble_model.classifier.parameters(), lr=0.001)

    ensemble_model, ensemble_history, ensemble_best_val_acc = train_ensemble(
        ensemble_model, train_loader, val_loader, criterion, optimizer,
        num_epochs=5, device=device
    )

    # Evaluate ensemble model
    print("Evaluating ensemble model...")
    report, cm, ensemble_metrics = evaluate_model(ensemble_model, "ensemble", test_loader, device=device)
    metrics_results["ensemble"] = ensemble_metrics
    
    # Compare performance
    print("\n=== Performance Comparison ===")
    print(f"DenseNet121: {best_val_accs['densenet121']:.4f}, Test Accuracy: {metrics_results['densenet121']['accuracy']:.4f}")
    print(f"ResNet50: {best_val_accs['resnet50']:.4f}, Test Accuracy: {metrics_results['resnet50']['accuracy']:.4f}")
    print(f"EfficientNet-B0: {best_val_accs['efficientnet_b0']:.4f}, Test Accuracy: {metrics_results['efficientnet_b0']['accuracy']:.4f}")
    print(f"Ensemble: {ensemble_best_val_acc:.4f}, Test Accuracy: {metrics_results['ensemble']['accuracy']:.4f}")
    
    # Generate combined metrics bar chart for all models
    generate_combined_metrics_barchart(metrics_results)
    
    print("\nTraining and evaluation complete. All outputs saved to output folder.")


# Function to generate a combined bar chart comparing all models
def generate_combined_metrics_barchart(metrics_results):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
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
            metrics_results[model_name]['f1']
        ]
        
        # Create bars with offset
        bars = ax.bar(x + (i - len(model_names)/2 + 0.5) * width, 
                      model_metrics, 
                      width, 
                      label=model_name)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, model_metrics):
            ax.text(bar.get_x() + bar.get_width()/2, 
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


if __name__ == "__main__":
    main()

