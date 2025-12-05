# Methods

## Dataset

### TissueMNIST
- **Source**: MedMNIST v2 dataset collection
- **Task Type**: Multi-class classification
- **Number of Classes**: 8
- **Class Labels**:
  0. Collecting Duct, Connecting Tubule
  1. Distal Convoluted Tubule
  2. Glomerular endothelial cells
  3. Interstitial endothelial cells
  4. Leukocytes
  5. Podocytes
  6. Proximal Tubule Segments
  7. Thick Ascending Limb

### Data Split
- **Training Set**: 80% of original training data (used for model training)
- **Validation Set**: 20% of original training data (used for early stopping and hyperparameter tuning)
- **Test Set**: Original test split (held-out for final evaluation)
- **Image Size**: 224×224 pixels
- **Channels**: Grayscale (converted to RGB for pretrained models)

## Data Preprocessing

### Image Transformations
All images undergo the following preprocessing pipeline:

1. **PIL Image Conversion**: Convert numpy arrays to PIL Image format
2. **RGB Conversion**: Convert grayscale images to RGB format using PIL's `convert("RGB")` method
3. **Tensor Conversion**: Convert to PyTorch tensor format
4. **Normalization**: Apply ImageNet normalization statistics:
   - Mean: [0.485, 0.456, 0.406]
   - Standard Deviation: [0.229, 0.224, 0.225]

This preprocessing ensures compatibility with ImageNet-pretrained models while properly handling the grayscale nature of TissueMNIST images.

## Model Architectures

### CNN Models

#### ResNet18
- **Architecture**: Residual Network with 18 layers
- **Pretrained Weights**: ImageNet-1K V1 (`ResNet18_Weights.IMAGENET1K_V1`)
- **Modification**: Final fully connected layer replaced for 8-class classification
- **Implementation**: `torchvision.models.resnet18`

#### ResNet50
- **Architecture**: Residual Network with 50 layers
- **Pretrained Weights**: ImageNet-1K V1 (`ResNet50_Weights.IMAGENET1K_V1`)
- **Modification**: Final fully connected layer replaced for 8-class classification
- **Implementation**: `torchvision.models.resnet50`

#### DenseNet121
- **Architecture**: Densely Connected Convolutional Network with 121 layers
- **Pretrained Weights**: ImageNet-1K V1 (`DenseNet121_Weights.IMAGENET1K_V1`)
- **Modification**: Classifier layer replaced for 8-class classification
- **Implementation**: `torchvision.models.densenet121`

#### EfficientNet-B0
- **Architecture**: EfficientNet-B0 (compound scaling method)
- **Pretrained Weights**: ImageNet-1K V1 (`EfficientNet_B0_Weights.IMAGENET1K_V1`)
- **Modification**: Final classifier layer replaced for 8-class classification
- **Implementation**: `torchvision.models.efficientnet_b0`

### Transformer Models

#### Vision Transformer (ViT-B/16)
- **Architecture**: Vision Transformer Base with patch size 16
- **Model**: `google/vit-base-patch16-224`
- **Pretrained**: ImageNet-21k pretrained weights from HuggingFace
- **Modification**: Custom classifier head added:
  - Dropout (0.1)
  - Linear layer (hidden_size → 8 classes)
- **Pooling**: Uses [CLS] token or pooler output
- **Implementation**: `transformers.ViTModel`

#### Swin Transformer Tiny
- **Architecture**: Swin Transformer Tiny variant
- **Model**: `microsoft/swin-tiny-patch4-window7-224`
- **Pretrained**: ImageNet-22k pretrained weights from HuggingFace
- **Modification**: Custom classifier head added:
  - Dropout (0.1)
  - Linear layer (hidden_size → 8 classes)
- **Pooling**: Mean pooling of last hidden state
- **Implementation**: `transformers.SwinModel`

#### Swin Transformer Base
- **Architecture**: Swin Transformer Base variant
- **Model**: `microsoft/swin-base-patch4-window7-224`
- **Pretrained**: ImageNet-22k pretrained weights from HuggingFace
- **Modification**: Custom classifier head added:
  - Dropout (0.1)
  - Linear layer (hidden_size → 8 classes)
- **Pooling**: Mean pooling of last hidden state
- **Implementation**: `transformers.SwinModel`

## Training Methodology

### Transfer Learning
All models use **ImageNet pretrained weights** for transfer learning:
- CNN models: ImageNet-1K V1 weights from torchvision
- Transformer models: ImageNet-21k/22k pretrained weights from HuggingFace

### Training Configuration

#### Hyperparameters
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Learning Rate**: 0.001
- **Momentum**: 0.9
- **Weight Decay**: 1e-4 (L2 regularization)
- **Batch Size**: 128
- **Number of Epochs**: 10 (with early stopping)
- **Loss Function**: Cross-Entropy Loss

#### Mixed Precision Training
- **Models**: ViT-B/16, Swin-Tiny, Swin-Base
- **Method**: FP16 mixed precision using PyTorch's Automatic Mixed Precision (AMP)
- **Benefits**: Reduced memory usage and faster training for transformer models
- **Gradient Clipping**: Applied with max_norm=1.0 to prevent gradient explosion

#### Early Stopping
- **Patience**: 3 epochs
- **Min Delta**: 0.001
- **Metric**: Validation loss
- **Behavior**: Training stops if validation loss doesn't improve by at least `min_delta` for `patience` consecutive epochs

### Data Loading
- **Workers**: 4 (GPU) or 2 (CPU)
- **Pin Memory**: Enabled for GPU training
- **Persistent Workers**: Enabled for faster data loading
- **Shuffling**: Enabled for training set, disabled for validation and test sets

## Evaluation Metrics

### Primary Metrics
1. **Accuracy**: Percentage of correctly classified samples
2. **AUC**: Area Under the ROC Curve (using MedMNIST evaluator)
3. **Per-Class Metrics**: Precision, Recall, F1-Score for each class

### Evaluation Tools
- **MedMNIST Evaluator**: Standard evaluation using MedMNIST's built-in evaluator
- **Scikit-learn Metrics**: Confusion matrix and classification report
- **TensorBoard**: Real-time training and validation metrics logging

### Evaluation Strategy
- **During Training**: Validation set used for early stopping and model selection
- **Final Evaluation**: Test set used for final performance reporting
- **Metrics Calculated**: Training loss/accuracy, validation loss/accuracy, test loss/accuracy, AUC, confusion matrix, per-class classification report

## Reproducibility

### Random Seed
- **Seed Value**: 42
- **Seeds Set**:
  - PyTorch CPU: `torch.manual_seed(42)`
  - PyTorch CUDA: `torch.cuda.manual_seed_all(42)`
  - NumPy: `np.random.seed(42)`
  - Python Random: `random.seed(42)`

### Deterministic Behavior
- **CuDNN Deterministic**: Enabled (`torch.backends.cudnn.deterministic = True`)
- **CuDNN Benchmark**: Disabled (`torch.backends.cudnn.benchmark = False`)

These settings ensure reproducible results across multiple runs.

## Model Analysis

### Computational Complexity
- **FLOPs**: Floating Point Operations (calculated using `thop` library)
- **Parameters**: Total number of trainable parameters
- **Analysis**: Performed during model initialization for all models

### Model Comparison
Models are compared based on:
1. **Performance Metrics**: Test accuracy, AUC, per-class F1-scores
2. **Computational Efficiency**: FLOPs, parameter count
3. **Training Efficiency**: Training time, convergence speed
4. **Generalization**: Train/validation/test accuracy gap

## Implementation Details

### Software and Libraries
- **PyTorch**: 2.9.0+
- **Torchvision**: 0.24.0+ (with new weights API)
- **Transformers**: 4.57.1+ (HuggingFace)
- **MedMNIST**: 2.2.0+
- **Scikit-learn**: 1.7.0+ (for metrics)
- **TensorBoard**: 2.15.0+ (for logging)
- **THOP**: 0.1.1+ (for FLOPs calculation)

### Hardware
- **Device**: CUDA-capable GPU (if available), otherwise CPU
- **Mixed Precision**: Automatically enabled for transformer models on GPU

### Training Monitoring
- **TensorBoard**: Real-time logging of:
  - Training/validation/test loss
  - Training/validation/test accuracy
- **Console Output**: Per-epoch progress with all metrics
- **Model Checkpoints**: Best models saved based on validation accuracy

## Statistical Analysis

### Performance Reporting
- **Best Model Selection**: Based on validation accuracy
- **Final Metrics**: Reported on held-out test set
- **Confusion Matrix**: Generated for detailed error analysis
- **Classification Report**: Per-class precision, recall, F1-score, and support

### Visualization
- **Training Curves**: Loss and accuracy plots for all models
- **Comparison Charts**: Bar charts comparing final test accuracies
- **Confusion Matrices**: Visual representation of classification errors
- **Sample Images**: Representative samples from each class

## Deliverables

### Research Paper Outputs
All results are saved in `results_comparative/paper_deliverables/`:

1. **Performance Tables**:
   - CSV format: `performance_table_{timestamp}.csv`
   - LaTeX format: `performance_table_{timestamp}.tex` (ready for paper inclusion)

2. **Visualizations**:
   - Training curves (PNG + PDF, 300 DPI)
   - Accuracy comparisons (PNG + PDF, 300 DPI)
   - Bar charts (PNG + PDF, 300 DPI)

3. **Sample Images**:
   - Individual class samples (PNG, 300 DPI)
   - Montage of all classes (PNG, 300 DPI)

4. **Model Analysis**:
   - FLOPs and parameter counts
   - Confusion matrices
   - Classification reports

## Code Organization

### Main Scripts
- **`train_comparative_study.py`**: Main training and evaluation pipeline
- **`tissue_main.py`**: Model architecture definitions

### Key Functions
- `set_seed()`: Reproducibility setup
- `load_tissuemnist()`: Dataset loading with train/val/test split
- `get_data_transform()`: Preprocessing pipeline
- `create_models()`: Model initialization with analysis
- `train_model()`: Training loop with early stopping and TensorBoard logging
- `evaluate_with_metrics()`: Confusion matrix and classification report
- `evaluate_with_medmnist()`: MedMNIST standard evaluation
- `save_paper_deliverables()`: Research paper outputs generation

## Experimental Design

### Fair Comparison
To ensure fair comparison across models:
- **Same Training Configuration**: All models use identical hyperparameters
- **Same Data Split**: All models trained on identical train/val/test splits
- **Same Preprocessing**: Identical data augmentation and normalization
- **Same Evaluation**: All models evaluated using same metrics and test set
- **Same Random Seed**: Reproducible initialization and data shuffling

### Model Selection
Models were selected to represent:
- **CNN Architectures**: Classic (ResNet), Dense (DenseNet), Efficient (EfficientNet)
- **Transformer Architectures**: Standard (ViT), Hierarchical (Swin)
- **Model Sizes**: Small (ResNet18, Swin-Tiny) to Large (ResNet50, Swin-Base)

This selection provides comprehensive coverage of modern deep learning architectures for medical image classification.

