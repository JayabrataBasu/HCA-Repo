# Fusion-Reborn: Multi-Model Ensemble for Medical Image Classification

## Overview

Fusion-Reborn is an advanced deep learning framework designed for medical image classification, specifically focused on classifying liver fibrosis from DICOM medical images. It implements several ensemble methods to improve classification performance beyond what individual models can achieve.

The framework features:

- Multiple pre-trained CNN architectures (DenseNet121, ResNet50, EfficientNet-B0)
- Two ensemble methods:
  - Standard feature fusion ensemble
  - Attention-based ensemble with learnable model weights
- Multi-slice processing for 3D context awareness
- Hyperparameter optimization with Optuna
- Comprehensive cross-validation and evaluation metrics

## Table of Contents

1. [Requirements](#requirements)
2. [Directory Structure](#directory-structure)
3. [Dataset Preparation](#dataset-preparation)
4. [Models Architecture](#models-architecture)
5. [Training Process](#training-process)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Hyperparameter Optimization](#hyperparameter-optimization)
8. [Usage Guide](#usage-guide)
9. [Output Files](#output-files)
10. [Visualizations](#visualizations)

## Requirements

The code requires the following libraries:

- PyTorch
- torchvision
- NumPy
- pandas
- pydicom
- scikit-learn
- matplotlib
- seaborn
- albumentations
- tqdm
- optuna

## Directory Structure

```
project/
├── Duke_DLDS code/
│   └── Fusion-Reborn(ensemble3.0).py  # Main code file
├── Series_Classification/              # DICOM dataset directory
│   └── patient_id/
│       └── series_id/
│           └── slice_files.dcm
└── SeriesClassificationKey.csv         # Labels for series
```

The output is saved to `C:\Softwares\All Programs\HCA\Duke_DLDS\Ensemble_new_output`.

## Dataset Preparation

The framework expects DICOM images organized in a specific hierarchy:

- Root directory contains patient folders
- Each patient folder contains series folders
- Each series folder contains DICOM slice files

Series labels must be provided in a CSV file with columns:

- `DLDS`: Patient ID
- `Series`: Series ID
- `Label`: Classification label

The `DICOMDataset` class handles dataset loading, preprocessing, and augmentation:

- Multiple slices per sample (configurable)
- Comprehensive augmentations for training
- Normalization using ImageNet statistics

## Models Architecture

### Base Models

Three pre-trained CNN architectures are used as base models:

1. **DenseNet121**: Good at capturing fine-grained features
2. **ResNet50**: Excellent at extracting hierarchical features
3. **EfficientNet-B0**: Optimized for efficiency and accuracy

Each base model has its classification layer replaced to match the number of target classes.

### Standard Ensemble Model

The `EnsembleModel` class combines multiple base models:

- Processes multiple slices per model
- Averages predictions across slices
- Concatenates features from all models
- Final classifier processes combined features

### Attention-Based Ensemble Model

The `AttentionEnsembleModel` extends the standard ensemble with an attention mechanism:

- Learns to weight the importance of each model
- Uses both feature fusion and weighted voting approaches
- Compares and combines results from both methods

## Training Process

### Individual Models

Base models are trained using:

- Cross-entropy loss
- Gradient accumulation for large models
- Learning rate scheduling with ReduceLROnPlateau
- Best model checkpointing

### Ensemble Training

Ensemble models provide two training options:

1. **Fixed base models**: Only train the fusion classifier
2. **Fine-tuning**: Partially unfreeze base models while training the fusion classifier

The attention-based ensemble additionally:

- Dynamically learns model weights
- Balances between fusion and voting loss using alpha parameter
- Visualizes attention weights during training

## Evaluation Metrics

The framework provides comprehensive evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC (overall and per-class)
- Confusion matrices
- Classification reports

All metrics are saved as text files and visualized as plots.

## Hyperparameter Optimization

The `objective` function works with Optuna to optimize hyperparameters:

- Base learning rates
- Optimizer selection (Adam, AdamW, SGD)
- Weight decay
- Batch size
- Fine-tuning strategies
- Ensemble-specific parameters

The best parameters are saved and used for the full cross-validation.

## Usage Guide

### Basic Usage

Run the script with default settings:

```python
python "Fusion-Reborn(ensemble3.0).py"
```

To skip optimization and run with default parameters, set:

```python
run_optimization = False  # In the main() function
```

### Customization

To adapt to a different dataset:

1. Update the paths at the beginning of the script:

   ```python
   root_dir = "path/to/your/dicom/data"
   csv_path = "path/to/your/labels.csv"
   output_folder = "path/to/output/directory"
   ```

2. Adjust the label loading function to match your CSV format:

   ```python
   def load_series_labels(csv_path):
       # Modify to match your CSV structure
   ```

3. Modify the number of output classes by updating the label mapping.

## Output Files

The framework generates numerous output files:

### For Each Model

- Model checkpoints (`.pth` files)
- Training/validation curves
- ROC curves
- Confusion matrices
- Classification metrics

### For Ensembles

- Ensemble model checkpoints
- Attention weights (CSV and visualizations)
- Performance comparisons

### For Cross-Validation

- Fold-specific results
- Average performance metrics
- Standard deviations
- Comparative visualizations

### For Optimization

- Optimization history
- Parameter importance plots
- Best parameter configuration

## Visualizations

The framework generates multiple visualizations:

- Training/validation loss and accuracy curves
- Learning rate schedules
- ROC curves (individual and comparison)
- Confusion matrices
- Performance metric bar charts
- Attention weight distributions
- Cross-validation comparisons
- Optimization history

## Code Structure

The code is organized into several key components:

1. **Dataset Handling**
   - `load_series_labels()`: Loads labels from CSV
   - `get_train_transform()`, `get_val_transform()`: Define augmentation pipelines
   - `DICOMDataset` class: Handles loading and preprocessing of DICOM images

2. **Model Definitions**
   - `EnsembleModel` class: Standard ensemble architecture
   - `AttentionEnsembleModel` class: Attention-based ensemble
   - `create_model()`: Creates and configures base models

3. **Training Functions**
   - `train_model()`: Trains individual models
   - `train_ensemble()`: Trains standard ensemble
   - `train_attention_ensemble()`: Trains attention-based ensemble
   - `unfreeze_last_layers()`: Helper for fine-tuning

4. **Evaluation Functions**
   - `evaluate_model()`: Evaluates model performance
   - `evaluate_attention_ensemble()`: Evaluates attention ensemble
   - `calculate_metrics()`: Computes performance metrics

5. **Visualization Functions**
   - `plot_roc_curves()`: Plots ROC curves
   - `plot_attention_weights()`: Visualizes attention distributions
   - `generate_metrics_barchart()`: Creates metric visualizations
   - `plot_combined_roc_curves()`: Compares ROC curves

6. **Optimization and Cross-Validation**
   - `objective()`: Optuna objective function for optimization
   - `run_full_cv()`: Runs full cross-validation
   - `calculate_cross_val_stats()`: Computes CV statistics
   - `create_combined_cv_comparison()`: Visualizes CV results

7. **Main Execution**
   - `main()`: Controls overall execution flow
