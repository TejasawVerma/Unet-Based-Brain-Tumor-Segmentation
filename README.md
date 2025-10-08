# Brain Tumor Segmentation using U-Net

Deep learning model for automatic segmentation of brain tumors from MRI scans using U-Net architecture with batch normalization.

## Overview

This project implements a U-Net convolutional neural network to automatically identify and segment brain tumors in MRI images. 

## Results

- **Test Dice Coefficient:** 0.8283
- **Target Achievement:** 97.4%
- **Dataset Size:** 3,929 brain MRI scans
- **Model Architecture:** U-Net with Batch Normalization

## Dataset

The model was trained on the [LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) from Kaggle, which contains brain MRI images with corresponding tumor masks.

- **Total Images:** 3,929 image-mask pairs
- **Image Size:** 128×128 pixels (resized from original)
- **Format:** Grayscale MRI scans with binary segmentation masks
- **Split:** 70% training, 15% validation, 15% test

## Model Architecture

### U-Net Structure

The model follows the classic U-Net architecture with modern improvements:

**Encoder (Downsampling Path):**
- 3 convolutional blocks with increasing filters (32 → 64 → 128)
- Each block: 2× Conv2D + BatchNorm + ReLU
- MaxPooling for downsampling

**Bottleneck:**
- 256 filters
- 2× Conv2D + BatchNorm + ReLU

**Decoder (Upsampling Path):**
- 3 transposed convolution blocks
- Skip connections from encoder
- Decreasing filters (128 → 64 → 32)

**Output Layer:**
- Single 1×1 convolution with sigmoid activation
- Produces binary segmentation mask

### Key Features

- **Batch Normalization:** Stabilizes training and improves convergence
- **He Normal Initialization:** Better weight initialization for ReLU activations
- **Skip Connections:** Preserves spatial information from encoder

## Loss Function

Combined loss for optimal performance:

```python
Loss = Binary Cross-Entropy + Dice Loss
```

- **Binary Cross-Entropy:** Pixel-wise classification accuracy
- **Dice Loss:** Optimizes overlap between prediction and ground truth
- **Dice Coefficient:** Primary evaluation metric (range: 0-1, higher is better)

## Training Details

### Hyperparameters

- **Optimizer:** Adam (learning rate: 0.001)
- **Batch Size:** 32
- **Epochs:** 50 (with early stopping)
- **Image Size:** 128×128×1

### Callbacks

- **ModelCheckpoint:** Saves best model based on validation Dice score
- **EarlyStopping:** Stops training after 15 epochs without improvement
- **ReduceLROnPlateau:** Reduces learning rate by 50% when validation loss plateaus

### Data Preprocessing

1. Load grayscale MRI images and masks
2. Resize to 128×128 pixels
3. Normalize images to [0, 1] range
4. Binarize masks (threshold at 127)
5. Add channel dimension for model compatibility

## Installation

### Requirements

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

### Dependencies

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

### Training

```python
# Load and preprocess data
python train.py

# Model will be saved as 'best_model.h5'
```
### Inference

```python
import tensorflow as tf
import cv2
import numpy as np

# Load trained model
model = tf.keras.models.load_model('best_model.h5', 
    custom_objects={'combined_loss': combined_loss, 
                    'dice_coefficient': dice_coefficient})

# Load and preprocess new MRI scan
img = cv2.imread('brain_scan.tif', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 128))
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=[0, -1])

# Predict tumor segmentation
prediction = model.predict(img)
mask = (prediction[0].squeeze() > 0.5).astype(np.uint8) * 255
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Test Dice Coefficient | 0.8283 |
| Test Loss | 0.3456 |
| Training Samples | 2,750 |
| Validation Samples | 590 |
| Test Samples | 589 |

## Results

1. **Predictions** (`predictions.png`): Side-by-side comparison of MRI scans, ground truth masks, predictions, and overlays

## Project Structure

```
brain-tumor-segmentation/
│
├── train.py                 # Main training script
├── best_model.h5           # Trained model weights
├── training_history.png    # Training curves
├── predictions.png         # Prediction visualizations
└── README.md              # This file
```

## Future Improvements

- Increase image resolution to 256×256 for better detail
- Implement data augmentation (rotation, flip, zoom)
- Add attention mechanisms to U-Net
- Experiment with deeper architectures (ResNet backbone)
- Multi-class segmentation for different tumor types
- 3D U-Net for volumetric MRI analysis

## Acknowledgments

- Dataset: [LGG MRI Segmentation on Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- Architecture inspired by: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

---

**Note:** This model is for research purposes only and should not be used for clinical diagnosis without proper validation and regulatory approval.
