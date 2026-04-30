# Flood Prediction ML

A machine learning project for flood detection using deep learning. This project utilizes a Convolutional Neural Network (CNN) to classify satellite or aerial images to predict whether an area is experiencing flooding or not.

## Project Description

This project implements a state-of-the-art deep learning model for automatic flood detection from satellite imagery. The model uses a balanced CNN architecture to classify images into two categories:
- **Flood**: Areas experiencing flooding
- **Non-Flood**: Areas without flooding

### Key Features
- Binary image classification using CNN
- Balanced dataset handling with class weights
- Comprehensive metrics (Accuracy, Precision, Recall, AUC)
- Data augmentation and preprocessing
- Training visualization and validation history logging
- Model persistence and prediction capabilities

## Setup

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sajibul-islam-robin/Flood_Prediction_ML.git
   cd Flood_Prediction_ML
   ```

2. **Create a virtual environment**
   ```bash
   # Using venv
   python -m venv myenv
   
   # Activate the environment
   # On Windows:
   myenv\Scripts\activate
   # On macOS/Linux:
   source myenv/bin/activate
   ```

3. **Install required dependencies**
   ```bash
   pip install tensorflow keras scikit-learn opencv-python matplotlib numpy scipy
   ```

   Or if you have a requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The dataset used in this project is sourced from **Zenodo**:

**Dataset Link**: [Flood Detection Dataset on Zenodo](https://zenodo.org/records/13366122)

### Dataset Structure
The dataset should be organized in the following directory structure:

```
data/
├── train/
│   ├── images/
│   │   ├── train_images_flood/
│   │   └── train_images_non_flood/
│   └── masks/
│       ├── train_masks_flood/
│       └── train_masks_non_flood/
├── val/
│   ├── images/
│   │   ├── val_images_flood/
│   │   └── val_images_non_flood/
│   └── masks/
│       ├── val_masks_flood/
│       └── val_masks_non_flood/
└── test/
    ├── images/
    │   ├── test_images_flood/
    │   └── test_images_non_flood/
    └── masks/
        ├── test_masks_flood/
        └── test_masks_non_flood/
```

### Downloading the Dataset
1. Visit [https://zenodo.org/records/13366122](https://zenodo.org/records/13366122)
2. Download the dataset files
3. Extract and organize them in the `data/` directory as shown above

## Project Structure

```
.
├── model.py                 # CNN model architecture
├── train.py                # Model training script
├── predict.py              # Single image prediction
├── validation.py           # Model validation utilities
├── data_processing.py      # Data loading and preprocessing
├── dataset.py             # Dataset utility functions
├── utils.py               # Utility functions
├── visualize.py           # Visualization utilities
├── best_model.h5          # Pre-trained model weights
├── training_log.csv       # Training history log
├── README.md              # This file
└── data/                  # Dataset directory (not included in repo)
    ├── train/
    ├── val/
    └── test/
```

## Usage

### Training the Model

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Build the CNN model
- Train the model with balanced class weights
- Generate training curves and metrics
- Save the best model as `best_model.h5`

### Making Predictions

To make flood predictions on a single image:

```bash
python predict.py --image path/to/image.jpg
```

### Validation

To validate the model performance:

```bash
python validation.py
```

## Model Architecture

The CNN model consists of:
- **Input Layer**: 128×128×3 (RGB images)
- **Entry Block**: Conv2D(32), BatchNorm, MaxPool, Dropout(0.2)
- **Middle Block**: Conv2D(64), BatchNorm, MaxPool, Dropout(0.3)
- **Exit Block**: Conv2D(128), BatchNorm, GlobalAvgPool, Dropout(0.5)
- **Classifier**: Dense(1) with sigmoid activation for binary classification

**Regularization**: L2 regularization (0.001) applied to all convolutional layers

## Training Configuration

- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy, Precision, Recall, AUC
- **Class Weights**: Balanced to handle class imbalance
- **Image Size**: 128×128 pixels
- **Batch Size**: Configurable (see training script)

## Results

The model achieves strong performance on flood detection:
- Training accuracy and validation metrics logged in `training_log.csv`
- Training curves saved as `model_curve.png`
- Sample predictions visualized during validation

## Requirements

Core dependencies:
- TensorFlow/Keras (Deep learning)
- scikit-learn (Class weight calculation)
- OpenCV (Image processing)
- Matplotlib (Data visualization)
- NumPy (Numerical operations)
- SciPy (Scientific computing)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project or dataset, please cite:
```
Dataset: Flood Detection Dataset
Source: https://zenodo.org/records/13366122
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.

---

**Last Updated**: April 2026
**Repository**: [Flood_Prediction_ML](https://github.com/sajibul-islam-robin/Flood_Prediction_ML)
