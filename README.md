# TissueMNIST: Comparative Study of CNN and Transformer Models

A comprehensive comparative study framework for evaluating CNN and Transformer-based models on the TissueMNIST medical imaging dataset. This project implements and compares multiple state-of-the-art architectures including ResNet, DenseNet, EfficientNet, Vision Transformer (ViT), DeiT (Data-efficient Image Transformer), and Swin Transformer.

## 📋 Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Scripts](#scripts)
- [Output Files](#output-files)
- [Model Analysis](#model-analysis)
- [Monitoring Training](#monitoring-training)
- [Evaluation Metrics](#evaluation-metrics)
- [Notebooks](#notebooks)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)

## ✨ Features

- **Multiple Model Architectures**: 
  - **CNN Models**: ResNet18, ResNet50, DenseNet121, EfficientNet-B0
  - **Transformer Models**: ViT-B/16, DeiT-Tiny, DeiT-Base, Swin-Tiny, Swin-Base
  
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
  - Individual model training with `run.py`
  - Checkpoint saving per model
  
- **Results and Visualization**:
  - Training curves and accuracy plots
  - JSON results export
  - Model checkpointing
  - Research paper deliverables (tables, figures, sample images)
  - Confusion matrix plotting from checkpoints

## 📦 Requirements

- Python 3.8+
- PyTorch 2.0.0+
- CUDA-capable GPU (optional, but recommended for transformer models)

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

All required dependencies including `medmnist`, `thop`, `tensorboard`, and `transformers` are listed in `requirements.txt`.

### 4. Configure Environment (Optional)

Create a `.env` file in the project root to customize settings:

```bash
# Example .env file
DATASET_PATH=mnist_dataset
BATCH_SIZE=128
NUM_EPOCHS=10
LEARNING_RATE=0.001
MODELS_TO_TRAIN=ResNet18,ResNet50,DenseNet121,EfficientNet-B0,ViT-B/16,DeiT-Tiny,DeiT-Base,Swin-Tiny,Swin-Base
```

### 5. Download Dataset

The dataset will be automatically downloaded on first run, or you can manually download it. The script looks for the dataset in:
- `./mnist_dataset/` (relative to project root)
- `./dataset/` (fallback)

## 🏃 Quick Start

### Run Individual Model Training

Train a single model with all deliverables saved in model-specific directories:

```bash
# Train ResNet18
python run.py ResNet18

# Train ViT-B/16
python run.py "ViT-B/16"

# Train DeiT-Tiny
python run.py DeiT-Tiny

# Train DeiT-Base
python run.py DeiT-Base

# Train all models sequentially
python run.py
```

This will:
1. Set random seed (42) for reproducibility
2. Load the TissueMNIST dataset with train/val/test split (80/20/test)
3. Train the specified model(s) with early stopping
4. Save checkpoints in `checkpoints/{model_name}/`
5. Save results and visualizations in `checkpoints/{model_name}/results/`
6. Generate research paper deliverables

### Run Comparative Study

Train and evaluate all models using the main utility script:

```bash
python utils.py
```

This will:
1. Train all configured models sequentially
2. Evaluate using MedMNIST evaluator
3. Generate comparative visualizations
4. Save results in `results_comparative/`

### Plot Confusion Matrix from Checkpoint

Generate confusion matrix and classification report from a saved checkpoint:

```bash
# Plot confusion matrix for test set
python plot_confusion_matrix.py checkpoints/resnet18/best_model.pt

# Plot confusion matrix for validation set
python plot_confusion_matrix.py checkpoints/vit_b_16/best_model.pt --split val

# Plot confusion matrix for training set
python plot_confusion_matrix.py checkpoints/deit_tiny/best_model.pt --split train
```

## 📖 Usage

### Model Selection

Available models:

**CNN Models:**
- `ResNet18`: ResNet-18 with ImageNet pretrained weights
- `ResNet50`: ResNet-50 with ImageNet pretrained weights
- `DenseNet121`: DenseNet-121 with ImageNet pretrained weights
- `EfficientNet-B0`: EfficientNet-B0 with ImageNet pretrained weights

**Transformer Models:**
- `ViT-B/16`: Vision Transformer (ViT-Base/16)
- `DeiT-Tiny`: Data-efficient Image Transformer (Tiny variant)
- `DeiT-Base`: Data-efficient Image Transformer (Base variant, distilled)
- `Swin-Tiny`: Swin Transformer (Tiny variant)
- `Swin-Base`: Swin Transformer (Base variant)

**Note**: Transformer models (ViT, DeiT, Swin) automatically use mixed precision training (FP16) for faster training and reduced memory usage.

### Configuration

Configuration is managed through environment variables (via `.env` file) or defaults in the `Config` class in `utils.py`:

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
    
    # Model selection
    MODELS_TO_TRAIN = [
        'ResNet18',
        'ResNet50',
        'DenseNet121',
        'EfficientNet-B0',
        'ViT-B/16',
        'DeiT-Tiny',
        'DeiT-Base',
        'Swin-Tiny',
        'Swin-Base'
    ]
```

### Training a Subset of Models

To train only specific models, set the `MODELS_TO_TRAIN` environment variable in your `.env` file:

```bash
MODELS_TO_TRAIN=ResNet50,ViT-B/16,DeiT-Base
```

Or modify the default list in `utils.py`.

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

## 📁 Project Structure

```
TissueMNIST/
├── models.py                    # Model definitions (CNN & Transformers)
├── utils.py                     # Main training utilities and functions
├── run.py                       # Individual model training script
├── plot_confusion_matrix.py     # Confusion matrix plotting from checkpoints
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── METHODS.md                   # Detailed methodology documentation
├── .env                         # Environment variables (create from .env.example)
├── .gitignore                   # Git ignore rules
│
├── notebooks/                   # Jupyter notebooks
│   ├── TissueMNIST.ipynb
│   ├── BloodMNIST.ipynb
│   └── visualize_results.ipynb
│
├── mnist_dataset/              # Dataset directory (auto-created)
│   └── tissuemnist_224.npz
│
├── checkpoints/                 # Model checkpoints (auto-created)
│   ├── resnet18/
│   │   ├── best_model.pt
│   │   ├── resnet18_checkpoint_*.pt
│   │   └── results/
│   │       ├── resnet18_results_*.json
│   │       ├── training_curves_*.png
│   │       ├── accuracy_curves_*.png
│   │       └── paper_deliverables/
│   │           ├── performance_table_*.csv/.tex
│   │           ├── sample_images/
│   │           └── figures/
│   ├── vit_b_16/
│   ├── deit_tiny/
│   ├── deit_base/
│   └── ...
│
├── results_comparative/        # Comparative study results (auto-created)
│   ├── results_*.json
│   ├── report_*.txt
│   ├── training_curves_*.png
│   ├── accuracy_curves_*.png
│   └── paper_deliverables/
│
└── runs/                       # TensorBoard logs (auto-created)
    └── {model_name}/
```

## 🔧 Scripts

### `run.py` - Individual Model Training

Train models individually with all outputs saved in model-specific directories.

**Usage:**
```bash
# Train a specific model
python run.py ResNet18
python run.py "ViT-B/16"
python run.py DeiT-Tiny
python run.py DeiT-Base

# Train all models sequentially
python run.py
```

**Outputs:**
- Checkpoints saved in `checkpoints/{model_name}/best_model.pt`
- Results JSON in `checkpoints/{model_name}/results/`
- Visualizations in `checkpoints/{model_name}/results/`
- Paper deliverables in `checkpoints/{model_name}/results/paper_deliverables/`

### `plot_confusion_matrix.py` - Confusion Matrix Plotting

Load a saved checkpoint and generate confusion matrix and classification report.

**Usage:**
```bash
python plot_confusion_matrix.py <checkpoint_path> [--split test|val|train] [--save-path <path>] [--display]
```

**Examples:**
```bash
# Plot confusion matrix for test set (default)
python plot_confusion_matrix.py checkpoints/resnet18/best_model.pt

# Plot for validation set
python plot_confusion_matrix.py checkpoints/vit_b_16/best_model.pt --split val

# Save to custom path
python plot_confusion_matrix.py checkpoints/deit_base/best_model.pt --save-path confusion_matrix.png

# Display plot interactively
python plot_confusion_matrix.py checkpoints/swin_tiny/best_model.pt --display
```

**Supported Models:**
- All CNN models: ResNet18, ResNet50, DenseNet121, EfficientNet-B0
- All Transformer models: ViT-B/16, DeiT-Tiny, DeiT-Base, Swin-Tiny, Swin-Base

### `utils.py` - Main Training Utilities

Contains the main training functions, dataset loading, model creation, and evaluation utilities. Can be run directly for comparative study:

```bash
python utils.py
```

## 📊 Output Files

### Individual Model Outputs (`checkpoints/{model_name}/`)

1. **Checkpoints**:
   - `best_model.pt`: Best model checkpoint (includes model state, training history, config)
   - `{model_name}_checkpoint_{timestamp}.pt`: Timestamped checkpoint

2. **Results** (`checkpoints/{model_name}/results/`):
   - `{model_name}_results_{timestamp}.json`: Complete training history and metrics
   - `training_curves_{timestamp}.png`: Training and validation loss/accuracy curves
   - `accuracy_curves_{timestamp}.png`: Accuracy progression over epochs

3. **Paper Deliverables** (`checkpoints/{model_name}/results/paper_deliverables/`):
   - `performance_table_{timestamp}.csv/.tex`: Performance metrics table
   - `sample_images/`: Sample images from each class
   - `figures/`: Publication-ready figures (PNG + PDF, 300 DPI)

### Comparative Study Outputs (`results_comparative/`)

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

## 📓 Notebooks

### TissueMNIST.ipynb

Interactive notebook for training and evaluation with:
- Real-time training progress
- Model checkpointing
- Evaluation with MedMNIST evaluator
- Visualization of training curves

### BloodMNIST.ipynb

Similar structure for BloodMNIST dataset experiments using HuggingFace Transformers.

### visualize_results.ipynb

For visualizing and analyzing saved results.

## 🔬 Model Analysis

Each model's computational complexity is analyzed during initialization:

- **FLOPs**: Floating Point Operations (in billions) - skipped for transformer models (too slow)
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
2. **Confusion Matrix**: Detailed classification errors (can be plotted with `plot_confusion_matrix.py`)
3. **Classification Report**: Per-class precision, recall, F1-score, and support

All metrics are saved and can be found in the results directory.

## 🔧 Troubleshooting

### CUDA Out of Memory

If you encounter GPU memory issues:

1. Reduce batch size in `.env` file:
   ```bash
   BATCH_SIZE=64  # or 32
   ```

2. Train models one at a time using `run.py` instead of all at once

3. Use CPU (slower but no memory issues):
   ```bash
   DEVICE=cpu
   ```

### Model Not Found Error

If you get an error like "Model 'DeiT-Base' not in MODELS_TO_TRAIN":

1. Check your `.env` file and ensure all models are listed:
   ```bash
   MODELS_TO_TRAIN=ResNet18,ResNet50,DenseNet121,EfficientNet-B0,ViT-B/16,DeiT-Tiny,DeiT-Base,Swin-Tiny,Swin-Base
   ```

2. Or remove the `MODELS_TO_TRAIN` line from `.env` to use defaults

### Dataset Not Found

If the dataset path is not detected:

1. Ensure the dataset is in `./mnist_dataset/` directory
2. The script will automatically detect the path relative to project root
3. If needed, modify `DATASET_PATH` in `.env` file

### Import Errors

If you encounter import errors:

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

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

### Checkpoint Loading Errors

If you encounter "Unexpected key(s) in state_dict" when loading checkpoints:

- This is normal for checkpoints saved with `thop` analysis
- The `plot_confusion_matrix.py` script automatically filters out these keys
- If loading manually, filter out keys ending with `_ops` or `_params`

## 📝 Notes

- **Optimized for TissueMNIST**: The workflow is specifically optimized for the TissueMNIST dataset (8 classes)
- **Modern API**: Uses the latest torchvision weights API (not the deprecated `pretrained=True` parameter)
- **Transfer Learning**: All models use ImageNet pretrained weights by default
- **Data Preprocessing**: TissueMNIST grayscale images are converted to RGB using PIL's `convert("RGB")` method
- **Mixed Precision**: Transformer models (ViT, DeiT, Swin) automatically use FP16 mixed precision for faster training
- **Reproducibility**: Random seed is set to 42 for reproducible results
- **Early Stopping**: Automatically stops training when validation loss stops improving
- **Model Analysis**: FLOPs and parameter counts are calculated and displayed for each model (FLOPs skipped for transformers)
- **Training Time**: CNNs are faster, Transformers take longer but benefit from mixed precision
- **Memory Management**: Models are moved to CPU and GPU cache is cleared after each model to prevent memory accumulation

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- MedMNIST dataset and evaluator
- PyTorch and torchvision teams
- HuggingFace transformers library
- Original model architectures (ResNet, DenseNet, EfficientNet, ViT, DeiT, Swin)
- THOP library for FLOPs calculation

---

**Happy Training! 🚀**
