# Project Documentation: Liver Fibrosis Classification System

## Overview
This project implements an ensemble deep learning system for classifying liver fibrosis stages from DICOM medical images. It combines three powerful CNN architectures (DenseNet121, ResNet50, and EfficientNet-B0) to create a robust classification system.

## Table of Contents
1. [Dataset Handling](#dataset-handling)
2. [Model Architecture](#model-architecture)
3. [Training Process](#training-process)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Visualization](#visualization)
6. [Implementation Details](#implementation-details)
7. [Performance Optimization](#performance-optimization)
8. [Common Issues and Solutions](#common-issues-and-solutions)

## Dataset Handling

### Data Organization
- **Input Structure**: DICOM images organized in patient/series directories
- **Label Source**: Series labels loaded from CSV file (SeriesClassificationKey.csv)
- **Image Selection**: Middle slice selected as representative for each series
- **Data Preprocessing**:
  - Resize to 224x224 (standard for CNNs)
  - Normalize pixel values to [0,1]
  - Convert single-channel images to 3-channel by stacking

### Data Splitting
- **Train/Validation/Test Split**: 60%/20%/20%
- **Stratification**: Maintains class distribution across splits
- **Random Seed**: Fixed at `42` for reproducibility
- **Implementation**: Uses `train_test_split` with two-stage splitting

### Error Handling
- Graceful handling of corrupted DICOM files
- Returns blank images instead of crashing when errors occur
- Detailed error logging for debugging

## Model Architecture

### Individual Models
1. **DenseNet121**
   - Pretrained on ImageNet
   - Final classifier replaced with custom layer for fibrosis stages
   - Input features: 1024

2. **ResNet50**
   - Pretrained on ImageNet
   - FC layer adapted to fibrosis classification
   - Input features: 2048

3. **EfficientNet-B0**
   - Pretrained on ImageNet
   - Classifier's final layer modified for fibrosis stages
   - Maintains efficiency with fewer parameters

### Ensemble Design
- **Concatenation Strategy**: Features from all models concatenated
- **Final Classification**: Linear layer maps combined features to class probabilities
- **Parameter Freezing**: Individual models frozen during ensemble training
- **Forward Pass**: All models process input independently, outputs combined

## Training Process

### Individual Model Training
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (learning rate: 0.001)
- **Epochs**: 10 per individual model
- **Batch Size**: 8
- **Validation Strategy**: Best model saved based on validation accuracy
- **Progress Tracking**: Loss and accuracy monitored for both training and validation

### Ensemble Training
- **Strategy**: Only train ensemble classifier layer (base models frozen)
- **Epochs**: 5 for ensemble training
- **Parameter Efficiency**: Only trains a small fraction of parameters
- **Model Selection**: Uses best version of individual models

## Evaluation Metrics

### Core Metrics
- **Accuracy**: Overall correct classifications / total samples
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under Receiver Operating Characteristic curve

### Detailed Analysis
- **Per-Class Metrics**: Performance breakdown by fibrosis stage
- **Confusion Matrix**: Visual representation of classification errors
- **ROC Curves**: For each class and micro-average
- **Comparative Visualization**: Bar charts and plots comparing all models

## Visualization

### Training Visualizations
- **Learning Curves**: Loss and accuracy plots over epochs
- **Performance Charts**: Bar charts of key metrics
- **ROC Curves**: Class-specific and micro-average ROC curves

### Comparison Visualizations
- **Combined Metrics Chart**: Side-by-side comparison of all models
- **Combined ROC Plot**: All models' ROC curves on one graph for easy comparison
- **Confusion Matrices**: Heat maps showing classification patterns

## Implementation Details

### File Structure
- **Dataset Class**: `DICOMDataset` handles DICOM loading and processing
- **Model Creation**: `create_model` function for model initialization
- **Training Functions**: Separate functions for individual and ensemble training
- **Evaluation**: Comprehensive evaluation with visualization
- **Output Management**: All results saved to specified output directory

### Key Parameters
- **Random Seed**: 42 for reproducible results
- **Device**: Automatically selects CUDA if available, falls back to CPU
- **Learning Rate**: 0.001 for all models
- **Class Mapping**: Automatically created from unique labels

## Performance Optimization

### Memory Management
- Images loaded on demand rather than all at once
- Middle slice selection reduces memory requirements
- Batch processing limits peak memory usage

### Training Efficiency
- Pre-trained models accelerate convergence
- Parameter freezing during ensemble training
- Best model checkpointing reduces overfitting

### Error Resilience
- Exception handling for corrupted images
- Automatic device selection based on availability
- Progress bars track training process

## Common Issues and Solutions

### DICOM Loading Issues
- Use `force=True` with pydicom to handle non-standard files
- Middle slice selection avoids memory errors with large series
- Check for empty directories in preprocessing

### Class Imbalance
- Stratified sampling maintains class distribution
- Weighted metrics account for class imbalance
- Plot confusion matrix to identify problematic classes

### Model Selection
- Monitor both accuracy and AUC-ROC for balanced assessment
- Compare individual models against ensemble to verify improvement
- Check for cases where ensemble underperforms individual models

### Hardware Requirements
- CUDA-compatible GPU recommended for reasonable training time
- Batch size can be adjusted based on available memory
- CPU fallback available but significantly slower
