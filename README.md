# TissueMNIST: Comparative Study of CNN and Transformer Models

A comprehensive comparative study framework for evaluating CNN and Transformer-based models on the TissueMNIST medical imaging dataset. This project implements and compares multiple state-of-the-art architectures including ResNet, DenseNet, EfficientNet, Vision Transformer (ViT), and Swin Transformer.

## 📋 Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Model Analysis](#model-analysis)
- [Monitoring Training](#monitoring-training)
- [Evaluation Metrics](#evaluation-metrics)
- [Notebooks](#notebooks)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)

## ✨ Features

- **Multiple Model Architectures**: 
  - CNN Models: ResNet18, ResNet50, DenseNet121, EfficientNet-B0
  - Transformer Models: ViT-B/16, Swin-Tiny, Swin-Base
  
- **Comprehensive Evaluation**:
  - Training, validation, and test metrics
  - MedMNIST evaluator integration
  - Confusion matrices and classification reports
  - Per-class precision, recall, and F1-scores
  - Comparative analysis and visualization
  
- **Modern PyTorch Implementation**:
  - Uses latest torchvision weights API
  - Transfer learning with ImageNet pretrained weights
  - Mixed precision training (FP16) for transformer models
  - GPU/CPU automatic detection
  - Reproducible results with seed setting
  
- **Training Features**:
  - Early stopping based on validation loss
  - Train/validation/test split (80/20/test)
  - TensorBoard logging for real-time monitoring
  - Model complexity analysis (FLOPs and parameters)
  
- **Results and Visualization**:
  - Training curves and accuracy plots
  - JSON results export
  - Model checkpointing
  - Research paper deliverables (tables, figures, sample images)

## 📦 Requirements

- Python 3.8+
- PyTorch 2.9.0+
- CUDA-capable GPU (optional, but recommended)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd TissueMNIST
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

All required dependencies including `medmnist`, `thop`, and `tensorboard` are listed in `requirements.txt`.

### 4. Download Dataset

The dataset will be automatically downloaded on first run, or you can manually download it. The script looks for the dataset in:
- `./mnist_dataset/` (relative to project root)
- `./dataset/` (fallback)

## 🏃 Quick Start

### Run Comparative Study

Train and evaluate all models:

```bash
python train_comparative_study.py
```

This will:
1. Set random seed (42) for reproducibility
2. Load the TissueMNIST dataset with train/val/test split (80/20/test)
3. Analyze model complexity (FLOPs and parameters)
4. Train all configured models with early stopping
5. Log metrics to TensorBoard
6. Evaluate using MedMNIST evaluator and generate confusion matrices
7. Generate visualizations and save results
8. Create research paper deliverables (tables, figures, sample images)

### Run Individual Model

You can also use the models defined in `tissue_main.py` directly:

```python
from tissue_main import ResNet50Classifier
import torch

# Initialize model
model = ResNet50Classifier(num_classes=8, pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

## 📖 Usage

### Training Configuration

Edit the `Config` class in `train_comparative_study.py` to customize:

```python
class Config:
    # Dataset (TissueMNIST-specific)
    DATASET_PATH = 'mnist_dataset'
    IMAGE_SIZE = 224
    BATCH_SIZE = 128
    
    # Training
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    USE_PRETRAINED = True
    
    # Model saving
    SAVE_MODELS = True
    MODEL_SAVE_DIR = 'saved_models_comparative'
    RESULTS_DIR = 'results_comparative'
    
    # Research paper deliverables
    SAVE_PAPER_DELIVERABLES = True
    PAPER_DIR = 'paper_deliverables'
    
    # Select which models to train
    MODELS_TO_TRAIN = [
        'ResNet18',
        'ResNet50',
        'DenseNet121',
        'EfficientNet-B0',
        'ViT-B/16',
        'Swin-Tiny',
        'Swin-Base'
    ]
```

**Note**: The workflow is optimized specifically for TissueMNIST (8 classes). The dataset is automatically split into 80% training, 20% validation, and the original test set.

### Training a Subset of Models

To train only specific models, modify `MODELS_TO_TRAIN`:

```python
MODELS_TO_TRAIN = [
    'ResNet50',
    'ViT-B/16'
]
```

### Using Jupyter Notebooks

The project includes Jupyter notebooks for interactive exploration:

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

Available notebooks:
- `notebooks/TissueMNIST.ipynb`: Interactive training and evaluation
- `notebooks/BloodMNIST.ipynb`: BloodMNIST dataset experiments
- `notebooks/visualize_results.ipynb`: Results visualization

## 📁 Project Structure

```
TissueMNIST/
├── tissue_main.py              # Model definitions (CNN & Transformers)
├── train_comparative_study.py  # Main training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── METHODS.md                  # Detailed methodology documentation
├── .gitignore                  # Git ignore rules
│
├── notebooks/                  # Jupyter notebooks
│   ├── TissueMNIST.ipynb
│   ├── BloodMNIST.ipynb
│   └── visualize_results.ipynb
│
├── mnist_dataset/              # Dataset directory (auto-created)
│   └── tissuemnist_224.npz
│
├── saved_models_comparative/   # Saved model checkpoints (auto-created)
│   └── *.pt files
│
├── results_comparative/        # Results and visualizations (auto-created)
│   ├── results_*.json
│   ├── report_*.txt
│   ├── training_curves_*.png
│   ├── accuracy_curves_*.png
│   └── paper_deliverables/     # Research paper outputs
│       ├── performance_table_*.csv/.tex
│       ├── sample_images/
│       └── figures/
│
└── runs/                       # TensorBoard logs (auto-created)
    └── {model_name}/
```

## ⚙️ Configuration

### Model Selection

Models are defined in `tissue_main.py`. Available models:

**CNN Models:**
- `ResNet18Classifier`: ResNet-18 with ImageNet pretrained weights
- `ResNet50Classifier`: ResNet-50 with ImageNet pretrained weights
- `DenseNet121Classifier`: DenseNet-121 with ImageNet pretrained weights
- `EfficientNetClassifier`: EfficientNet-B0 with ImageNet pretrained weights

**Transformer Models:**
- `ViTClassifier`: Vision Transformer (ViT-B/16)
- `SwinTransformerClassifier`: Swin Transformer (Tiny/Base variants)

**Note**: Transformer models (ViT, Swin) automatically use mixed precision training (FP16) for faster training and reduced memory usage.

### Hyperparameters

Key hyperparameters can be adjusted in the `Config` class:

- `NUM_EPOCHS`: Number of training epochs (default: 10, with early stopping)
- `BATCH_SIZE`: Batch size for training (default: 128)
- `LEARNING_RATE`: Learning rate for SGD optimizer (default: 0.001)
- `MOMENTUM`: SGD momentum (default: 0.9)
- `WEIGHT_DECAY`: L2 regularization (default: 1e-4)
- `USE_PRETRAINED`: Use ImageNet pretrained weights (default: True)

### Early Stopping

Early stopping is automatically enabled with:
- **Patience**: 3 epochs
- **Min Delta**: 0.001
- **Metric**: Validation loss

Training stops if validation loss doesn't improve for 3 consecutive epochs.

### TensorBoard Monitoring

Training metrics are automatically logged to TensorBoard. View them with:

```bash
tensorboard --logdir=runs
```

Logs include:
- Training/validation/test loss
- Training/validation/test accuracy
- Organized by model name

## 📊 Output Files

After training, the following files are generated:

### Results Directory (`results_comparative/`)

1. **JSON Results** (`results_*.json`):
   - Complete training history for all models
   - Test metrics (AUC, Accuracy)
   - Configuration parameters

2. **Text Report** (`report_*.txt`):
   - Human-readable summary
   - Model rankings by performance
   - Parameter counts

3. **Visualizations**:
   - `training_curves_*.png`: Training and test loss/accuracy comparison
   - `accuracy_curves_*.png`: Individual model accuracy curves

4. **Model Checkpoints** (in `saved_models_comparative/`):
   - `*_best.pt`: Best model during training
   - `*_latest.pt`: Latest checkpoint
   - `*_final.pt`: Final model with full history

### Research Paper Deliverables (`results_comparative/paper_deliverables/`)

1. **Performance Tables**:
   - `performance_table_*.csv`: CSV format for analysis
   - `performance_table_*.tex`: LaTeX format ready for paper inclusion

2. **Sample Images** (`sample_images/`):
   - Individual class samples (3 per class)
   - Montage of all classes

3. **Publication Figures** (`figures/`):
   - Loss comparison (PNG + PDF, 300 DPI)
   - Accuracy comparison (PNG + PDF, 300 DPI)
   - Accuracy bar charts (PNG + PDF, 300 DPI)

### TensorBoard Logs (`runs/`)

- Real-time training metrics
- Organized by model name
- View with: `tensorboard --logdir=runs`

## 📓 Notebooks

### TissueMNIST.ipynb

Interactive notebook for training CNN (ResNet18) and Swin Transformer models with:
- Real-time training progress
- Model checkpointing
- Evaluation with MedMNIST evaluator
- Visualization of training curves

### BloodMNIST.ipynb

Similar structure for BloodMNIST dataset experiments.

### visualize_results.ipynb

For visualizing and analyzing saved results.

## 🔧 Troubleshooting

### CUDA Out of Memory

If you encounter GPU memory issues:

1. Reduce batch size:
   ```python
   BATCH_SIZE = 64  # or 32
   ```

2. Train models sequentially instead of all at once

3. Use CPU (slower but no memory issues):
   ```python
   DEVICE = torch.device('cpu')
   ```

### Dataset Not Found

If the dataset path is not detected:

1. Ensure the dataset is in `./mnist_dataset/` directory
2. The script will automatically detect the path relative to project root
3. If needed, modify `DATASET_PATH` in the `Config` class

### Import Errors

If you encounter import errors:

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
pip install medmnist

# Verify installation
python -c "import torch; import torchvision; import transformers; import medmnist; print('All imports successful')"
```

### Model Loading Issues

If pretrained weights fail to load:

- Check internet connection (weights are downloaded from HuggingFace/torchvision)
- Verify torchvision version supports the weights API:
  ```bash
  pip install --upgrade torchvision
  ```

## 📝 Notes

- **Optimized for TissueMNIST**: The workflow is specifically optimized for the TissueMNIST dataset (8 classes)
- **Modern API**: Uses the latest torchvision weights API (not the deprecated `pretrained=True` parameter)
- **Transfer Learning**: All models use ImageNet pretrained weights by default
- **Data Preprocessing**: TissueMNIST grayscale images are converted to RGB using PIL's `convert("RGB")` method
- **Mixed Precision**: Transformer models (ViT, Swin) automatically use FP16 mixed precision for faster training
- **Reproducibility**: Random seed is set to 42 for reproducible results
- **Early Stopping**: Automatically stops training when validation loss stops improving
- **Model Analysis**: FLOPs and parameter counts are calculated and displayed for each model
- **Training Time**: CNNs are faster, Transformers take longer but benefit from mixed precision

## 📚 Documentation

- **README.md**: This file - setup and usage instructions
- **METHODS.md**: Detailed methodology documentation for research papers

## 🔬 Model Analysis

Each model's computational complexity is analyzed during initialization:

- **FLOPs**: Floating Point Operations (in billions)
- **Parameters**: Total trainable parameters (in millions)

This information is printed during model creation and helps understand the computational cost of each architecture.

## 📈 Monitoring Training

### TensorBoard

Real-time training monitoring is available via TensorBoard:

```bash
# Start TensorBoard
tensorboard --logdir=runs

# Access at http://localhost:6006
```

### Console Output

Each epoch displays:
- Training loss and accuracy
- Validation loss and accuracy
- Test loss and accuracy
- Best validation accuracy and epoch
- Early stopping status (if triggered)

## 🔬 Evaluation Metrics

After training, each model is evaluated with:

1. **MedMNIST Evaluator**: Standard AUC and accuracy metrics
2. **Confusion Matrix**: Detailed classification errors
3. **Classification Report**: Per-class precision, recall, F1-score, and support

All metrics are saved and can be found in the results directory.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- MedMNIST dataset and evaluator
- PyTorch and torchvision teams
- HuggingFace transformers library
- Original model architectures (ResNet, DenseNet, EfficientNet, ViT, Swin)
- THOP library for FLOPs calculation

---

**Happy Training! 🚀**

