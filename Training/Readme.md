# Real-Time Deepfake Detection via Frame-Level EfficientNet Ensemble and Client-Server Deployment

# Training Details for DeepFake Detection Model

## Dataset Information
The model was trained on a comprehensive dataset combining three major deepfake datasets:
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [DeeperForensics](https://github.com/EndlessSora/DeeperForensics-1.0)

## Training Environment
- Platform: Kaggle Notebooks
- Hardware: Tesla T4 GPU
- Runtime: Python with Deep Learning frameworks

## Model Architecture
The model implements an ensemble approach combining:
- EfficientNet-B4
- EfficientNet-B5

This ensemble architecture leverages the strengths of both models to improve detection accuracy and robustness.

## Training Parameters
- Batch Size: 64
- Learning Rate: 1e-5
- Model Base: EfficientNet-B4 and B5 ensemble
- Framework: PyTorch
- Optimizer: Adam
- Loss Function: Binary CrossEntropy With Logits Loss


## Data Preprocessing Pipeline
1. Video Processing:
   - Frame extraction from video sequences
2. Face Processing:
   - Face detection and extraction
   - Face zooming using Dlib 68 facial landmark detection

3. Input Preparation:
   - Image Resizing: All images are resized to 224×224 pixels
   - Normalization: Images are normalized using ImageNet statistics
     - Mean: [0.485, 0.456, 0.406]
     - Standard Deviation: [0.229, 0.224, 0.225]
   
   Data Augmentation (Training Only):
   - Random Horizontal Flip
   - Random Rotation (±10 degrees)
   - Random Resized Crop (scale: 0.8 to 1.0)
   - Color Jitter:
     - Brightness: ±0.4
     - Contrast: ±0.4
     - Saturation: ±0.2
     - Hue: ±0.1
   - Random Grayscale Conversion (20% probability)

## Dataset Organization
The dataset is organized into three splits:
- Train
- Validation
- Test

Each split contains two classes:
- Real
- Fake

```python
# Dataset Structure
dataset_root/
    ├── Train/
    │   ├── Real/
    │   └── Fake/
    ├── Val/
    │   ├── Real/
    │   └── Fake/
    └── Test/
        ├── Real/
        └── Fake/
```

## Data Loading Pipeline
The dataset is loaded using a custom PyTorch Dataset class that:
- Maintains separate lists for images and their labels
- Handles image loading and transformation
- Provides error handling for corrupt images
- Tracks class distribution across splits

## Model Performance
Performance metrics for individual datasets were not recorded. For best practices in future iterations, it's recommended to track:
- Accuracy
- Confusion matrix
- ROC curves
- Precision-Recall metrics

## Hardware Requirements
For Training:
- GPU: Tesla T4 or equivalent
- 16 GB 
- 100 GB


## 🔄 Training Workflow
1. Load video dataset (Celeb-DF, FF++, DeeperForensics)
2. Extract and preprocess frames
3. Face alignment and cropping
4. Apply data augmentation
5. Train EfficientNet-B4 and B5 individually
6. Ensemble model predictions
7. Evaluate on validation and test splits
