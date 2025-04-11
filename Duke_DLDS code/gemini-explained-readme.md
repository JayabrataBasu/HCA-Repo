# Liver Fibrosis and Cirrhosis Detection with Attention Fusion of Deep Neural Networks

## Overall Purpose

The code aims to classify medical image series (likely DICOM format, based on `pydicom` usage) into different categories specified in a CSV file (`SeriesClassificationKey.csv`).  
It trains several individual Convolutional Neural Network (CNN) models and combines their predictions using two ensemble methods:

- A **simple ensemble**.
- An **attention-based ensemble**.

The objective is to compare these approaches and identify the best-performing model for the classification task.

---

## Methods Employed

### 1. Data Handling & Preprocessing

- **Input**:  
  Reads DICOM image series from a nested directory structure (`root_dir`) and corresponding labels from a CSV file (`csv_path`).

- **Dataset Representation**:  
  Uses a custom PyTorch `Dataset` class (`DICOMDataset`) to manage the data.  
  It selects the **middle slice** of each DICOM series as the representative image.

- **Image Loading**:  
  - Loads images using `pydicom`.
  - Converts single-channel images to 3 channels (required by standard CNNs).
  - Resizes images to **224x224** using `cv2.resize`.
  - Normalizes pixel values to the **[0, 1]** range.

- **Data Splitting**:  
  Splits the data into **training**, **validation**, and **test** sets using `train_test_split` with **stratification** to maintain label distribution.

- **Data Loading**:  
  Uses PyTorch `DataLoader` for efficient batching during training and evaluation.

### 2. Data Augmentation

Applies random transformations to the training images to improve model robustness:

- Random Rotation (±15 degrees)
- Random Horizontal Flip
- Random Vertical Flip
- Random Brightness Adjustment
- Random Contrast Adjustment

Augmentations are applied using `torchvision.transforms.functional` after converting tensor images to PIL format and back.

---

## Modeling

### Base Models

Utilizes three well-known pretrained CNN architectures from `torchvision.models`:

- **DenseNet121**
- **ResNet50**
- **EfficientNet-B0**

Each model:

- Has its final classification layer replaced to match the number of output classes.
- Is fine-tuned on the DICOM dataset using **transfer learning** (`pretrained=True`).

---

### Simple Ensemble (`EnsembleModel`)

- Takes the outputs (logits/class scores) from the trained base models.
- Concatenates these outputs.
- Feeds the concatenated vector into a final `nn.Linear` classifier layer to produce ensemble predictions.  
  (This is a form of **feature concatenation fusion**.)

---

### Attention Ensemble (`AttentionEnsembleModel`)

- Takes outputs from the trained base models.
- **Attention Mechanism**:  
  - A small network computes attention weights based on concatenated outputs.
  - Learns how much to weight each model's prediction dynamically.
- **Weighted Voting**:  
  - Calculates a weighted sum of the softmax probabilities from each model.
- **Feature Fusion**:  
  - Independently concatenates base model outputs and passes them through a separate multi-layer classifier.
- **Combined Loss**:  
  - Loss is computed from both the attention-weighted voting output and the feature fusion output.
  - Controlled by an **alpha** parameter.

---

## Training

- **Individual Models**:  
  - Trained separately using a standard training loop (`train_model`).
  - Tracks training and validation loss/accuracy per epoch.
  - Saves the best-performing model based on validation accuracy.

- **Ensemble Models**:  
  - Trained **after** individual models are trained.
  - **Base model weights are frozen** (`param.requires_grad = False`).
  - Only new classifier layers (Simple Ensemble) or attention mechanism and classifiers (Attention Ensemble) are trained.

- **Optimization**:  
  - Adam optimizer (`optim.Adam`).

- **Loss Function**:  
  - Cross-Entropy Loss (`nn.CrossEntropyLoss`), suitable for multi-class classification.

---

## Evaluation

- **Metrics**:  
  - Accuracy
  - Precision (weighted)
  - Recall (weighted)
  - F1-Score (weighted)

- **Confusion Matrix**:  
  - Generates and saves a confusion matrix to visualize model performance.

---

## Notes

- Gradient accumulation is supported but not necessary for the selected models.
- Model performance comparisons between individual models, simple ensemble, and attention ensemble are documented.

---

Would you also want a **table of contents** at the top (`[TOC]`) and maybe a **Usage** or **Environment Setup** section if you're planning to upload this to GitHub? 🚀  
I can help you add those if you like!
