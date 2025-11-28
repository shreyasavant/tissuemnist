"""
Comparative Study Training and Testing Framework for TissueMNIST
================================================================
This script trains and evaluates multiple CNN and Transformer models
on TissueMNIST dataset (8 classes) for comparative analysis.

Models included:
- CNN: ResNet18, ResNet50, DenseNet121, EfficientNet-B0
- Transformers: ViT-B/16, Swin-Tiny, Swin-Base

Optimized for TissueMNIST dataset.
"""

import os
import sys
import warnings
import json
import csv
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.tensorboard import SummaryWriter
from thop import profile

import medmnist
from medmnist import Evaluator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import models from tissue_main.py
from tissue_main import (
    ResNet18Classifier,
    ResNet50Classifier,
    DenseNet121Classifier,
    EfficientNetClassifier,
    ViTClassifier,
    SwinTransformerClassifier
)

# Reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration - Optimized for TissueMNIST
# TissueMNIST constants
TISSUEMNIST_NUM_CLASSES = 8
TISSUEMNIST_TASK = 'multi-class'
TISSUEMNIST_LABELS = [
    'Collecting Duct, Connecting Tubule',
    'Distal Convoluted Tubule',
    'Glomerular endothelial cells',
    'Interstitial endothelial cells',
    'Leukocytes',
    'Podocytes',
    'Proximal Tubule Segments',
    'Thick Ascending Limb'
]

class Config:
    """Configuration class that reads from environment variables with fallback to defaults"""
    
    # Helper method to get env var with type conversion
    def _get_env(key, default, type_func=str):
        value = os.getenv(key)
        if value is None:
            return default
        if type_func == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        if type_func == list:
            return [item.strip() for item in value.split(',') if item.strip()]
        return type_func(value)
    
    # Dataset (TissueMNIST-specific)
    DATASET_PATH = _get_env('DATASET_PATH', 'mnist_dataset', str)
    IMAGE_SIZE = _get_env('IMAGE_SIZE', 224, int)
    BATCH_SIZE = _get_env('BATCH_SIZE', 128, int)
    
    # Training
    NUM_EPOCHS = _get_env('NUM_EPOCHS', 10, int)
    LEARNING_RATE = _get_env('LEARNING_RATE', 0.001, float)
    MOMENTUM = _get_env('MOMENTUM', 0.9, float)
    WEIGHT_DECAY = _get_env('WEIGHT_DECAY', 1e-4, float)
    USE_PRETRAINED = _get_env('USE_PRETRAINED', True, bool)
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = _get_env('EARLY_STOPPING_PATIENCE', 3, int)
    EARLY_STOPPING_MIN_DELTA = _get_env('EARLY_STOPPING_MIN_DELTA', 0.001, float)
    
    # Model saving
    SAVE_MODELS = _get_env('SAVE_MODELS', True, bool)
    MODEL_SAVE_DIR = _get_env('MODEL_SAVE_DIR', 'saved_models_comparative', str)
    RESULTS_DIR = _get_env('RESULTS_DIR', 'results_comparative', str)
    
    # Research paper deliverables
    SAVE_PAPER_DELIVERABLES = _get_env('SAVE_PAPER_DELIVERABLES', True, bool)
    PAPER_DIR = _get_env('PAPER_DIR', 'paper_deliverables', str)
    
    # TensorBoard
    TENSORBOARD_LOG_DIR = _get_env('TENSORBOARD_LOG_DIR', 'runs', str)
    ENABLE_TENSORBOARD = _get_env('ENABLE_TENSORBOARD', True, bool)
    
    # Device
    _device_str = _get_env('DEVICE', 'auto', str)
    if _device_str == 'auto':
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device(_device_str)
    
    # Models to train (can be modified to train subset)
    _models_str = _get_env('MODELS_TO_TRAIN', None, list)
    if _models_str:
        MODELS_TO_TRAIN = _models_str
    else:
        MODELS_TO_TRAIN = [
            'ResNet18',
            'ResNet50',
            'DenseNet121',
            'EfficientNet-B0',
            'ViT-B/16',
            'Swin-Tiny',
            'Swin-Base'
        ]
    
    # DataLoader settings
    NUM_WORKERS = _get_env('NUM_WORKERS', None, int)  # Will be set based on CUDA availability if None
    PIN_MEMORY = _get_env('PIN_MEMORY', None, bool)  # Will be set based on CUDA availability if None
    
    # Advanced settings
    TRAIN_VAL_SPLIT = _get_env('TRAIN_VAL_SPLIT', 0.8, float)
    USE_MIXED_PRECISION = _get_env('USE_MIXED_PRECISION', True, bool)
    GRADIENT_CLIP_NORM = _get_env('GRADIENT_CLIP_NORM', 1.0, float)
    
    # Reproducibility
    RANDOM_SEED = _get_env('RANDOM_SEED', 42, int)
    
    # Verbose output
    VERBOSE = _get_env('VERBOSE', True, bool)

# Data Loading
def to_pil_image_safe(img):
    """Convert to PIL Image if not already PIL (picklable function for multiprocessing)"""
    if isinstance(img, Image.Image):
        return img
    return transforms.ToPILImage()(img)

def convert_to_rgb(img):
    """Convert PIL image to RGB format if not already RGB (picklable function for multiprocessing)"""
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def get_data_transform():
    """Get data transformation for pretrained models
    """
    return transforms.Compose([
        transforms.Lambda(to_pil_image_safe),  # Convert to PIL Image (handles both PIL and numpy arrays)
        transforms.Lambda(convert_to_rgb),  # Convert to RGB format
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

def load_tissuemnist(config):
    """Load TissueMNIST dataset - optimized for this specific dataset"""
    from medmnist import TissueMNIST
    
    # Determine dataset path (simplified)
    dataset_path = config.DATASET_PATH
    if not os.path.exists(dataset_path):
        # Try project root if running from subdirectory
        project_root = os.path.dirname(os.getcwd()) if 'notebooks' in os.getcwd() else os.getcwd()
        dataset_path = os.path.join(project_root, config.DATASET_PATH)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"TissueMNIST dataset not found at {dataset_path}. "
                                  f"Please ensure the dataset is downloaded.")
    
    data_transform = get_data_transform()
    
    # Load TissueMNIST datasets
    train_dataset = TissueMNIST(
        split='train',
        transform=data_transform,
        download=False,
        root=dataset_path,
        size=config.IMAGE_SIZE,
        mmap_mode='r'
    )
    
    test_dataset = TissueMNIST(
        split='test',
        transform=data_transform,
        download=False,
        root=dataset_path,
        size=config.IMAGE_SIZE,
        mmap_mode='r'
    )
    
    # Split training data into train/val (configurable split)
    train_size = int(config.TRAIN_VAL_SPLIT * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_split, val_split = random_split(train_dataset, [train_size, val_size])
    
    # Optimized DataLoaders
    # Use config values if set, otherwise use defaults based on CUDA availability
    if config.NUM_WORKERS is not None:
        num_workers = config.NUM_WORKERS
    else:
        num_workers = 4 if torch.cuda.is_available() else 0
    
    if config.PIN_MEMORY is not None:
        pin_memory = config.PIN_MEMORY
    else:
        pin_memory = torch.cuda.is_available()
    
    train_loader = data.DataLoader(
        dataset=train_split,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    val_loader = data.DataLoader(
        dataset=val_split,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    return train_loader, val_loader, test_loader, dataset_path

# Model Analysis
def analyze_model(model, device, input_shape=(1, 3, 224, 224), skip_flops=False):
    """Analyze model FLOPs and parameters using thop
    
    Args:
        model: Model to analyze
        device: Device to run analysis on
        input_shape: Input shape for FLOPs calculation
        skip_flops: If True, skip FLOPs calculation (faster, only count parameters)
    """
    # Count parameters (always fast)
    params = sum(p.numel() for p in model.parameters())

    # Skip FLOPs if requested (useful for transformer models which are slow)
    if skip_flops:
        return None, params / 1e6
    
    # Try to calculate FLOPs (can be slow for large models)
    try:
        dummy_input = torch.randn(input_shape).to(device)
        flops, params_calc = profile(model, inputs=(dummy_input,), verbose=False)
        return flops / 1e9, params / 1e6  # Convert to billions and millions
    except Exception as e:
        # Fallback to simple parameter count if FLOPs calculation fails
        return None, params / 1e6

# Model Initialization
def create_models(config):
    """Create all models for TissueMNIST (8 classes) - optimized"""
    models = {}
    num_classes = TISSUEMNIST_NUM_CLASSES
    
    total_models = len(config.MODELS_TO_TRAIN)
    model_count = 0
    print(f"Creating {total_models} models...")
    
    if 'ResNet18' in config.MODELS_TO_TRAIN:
        model_count += 1
        print(f"  [{model_count}/{total_models}] Creating ResNet18...", end=' ', flush=True)
        model = ResNet18Classifier(
            num_classes=num_classes,
            pretrained=config.USE_PRETRAINED
        ).to(config.DEVICE)
        flops, params = analyze_model(model, config.DEVICE)
        if flops is not None:
            print(f"✓ FLOPs: {flops:.2f}B, Parameters: {params:.2f}M")
        else:
            print(f"✓ Parameters: {params:.2f}M")
        models['ResNet18'] = model
    
    if 'ResNet50' in config.MODELS_TO_TRAIN:
        model_count += 1
        print(f"  [{model_count}/{total_models}] Creating ResNet50...", end=' ', flush=True)
        model = ResNet50Classifier(
            num_classes=num_classes,
            pretrained=config.USE_PRETRAINED
        ).to(config.DEVICE)
        flops, params = analyze_model(model, config.DEVICE)
        if flops is not None:
            print(f"✓ FLOPs: {flops:.2f}B, Parameters: {params:.2f}M")
        else:
            print(f"✓ Parameters: {params:.2f}M")
        models['ResNet50'] = model
    
    if 'DenseNet121' in config.MODELS_TO_TRAIN:
        model_count += 1
        print(f"  [{model_count}/{total_models}] Creating DenseNet121...", end=' ', flush=True)
        model = DenseNet121Classifier(
            num_classes=num_classes,
            pretrained=config.USE_PRETRAINED
        ).to(config.DEVICE)
        flops, params = analyze_model(model, config.DEVICE)
        if flops is not None:
            print(f"✓ FLOPs: {flops:.2f}B, Parameters: {params:.2f}M")
        else:
            print(f"✓ Parameters: {params:.2f}M")
        models['DenseNet121'] = model
    
    if 'EfficientNet-B0' in config.MODELS_TO_TRAIN:
        model_count += 1
        print(f"  [{model_count}/{total_models}] Creating EfficientNet-B0...", end=' ', flush=True)
        model = EfficientNetClassifier(
            num_classes=num_classes,
            pretrained=config.USE_PRETRAINED
        ).to(config.DEVICE)
        flops, params = analyze_model(model, config.DEVICE)
        if flops is not None:
            print(f"✓ FLOPs: {flops:.2f}B, Parameters: {params:.2f}M")
        else:
            print(f"✓ Parameters: {params:.2f}M")
        models['EfficientNet-B0'] = model
    
    if 'ViT-B/16' in config.MODELS_TO_TRAIN:
        model_count += 1
        print(f"  [{model_count}/{total_models}] Creating ViT-B/16...", end=' ', flush=True)
        model = ViTClassifier(
            num_classes=num_classes,
            model_name="google/vit-base-patch16-224",
            pretrained=config.USE_PRETRAINED
        ).to(config.DEVICE)
        # Skip FLOPs for transformer models (too slow)
        flops, params = analyze_model(model, config.DEVICE, skip_flops=True)
        print(f"✓ Parameters: {params:.2f}M")
        models['ViT-B/16'] = model
    
    if 'Swin-Tiny' in config.MODELS_TO_TRAIN:
        model_count += 1
        print(f"  [{model_count}/{total_models}] Creating Swin-Tiny...", end=' ', flush=True)
        model = SwinTransformerClassifier(
            num_classes=num_classes,
            model_name="microsoft/swin-tiny-patch4-window7-224",
            pretrained=config.USE_PRETRAINED
        ).to(config.DEVICE)
        # Skip FLOPs for transformer models (too slow)
        flops, params = analyze_model(model, config.DEVICE, skip_flops=True)
        print(f"✓ Parameters: {params:.2f}M")
        models['Swin-Tiny'] = model
    
    if 'Swin-Base' in config.MODELS_TO_TRAIN:
        model_count += 1
        print(f"  [{model_count}/{total_models}] Creating Swin-Base...", end=' ', flush=True)
        model = SwinTransformerClassifier(
            num_classes=num_classes,
            model_name="microsoft/swin-base-patch4-window7-224",
            pretrained=config.USE_PRETRAINED
        ).to(config.DEVICE)
        # Skip FLOPs for transformer models (too slow)
        flops, params = analyze_model(model, config.DEVICE, skip_flops=True)
        print(f"✓ Parameters: {params:.2f}M")
        models['Swin-Base'] = model
    
    print()  # Empty line after all models created
    return models

# Early Stopping
class EarlyStopper:
    """Early stopping utility to stop training when validation loss stops improving"""
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
    
    def early_stop(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

# Training Functions
def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, gradient_clip_norm=1.0):
    """Train for one epoch - optimized for TissueMNIST (multi-class)
    
    Args:
        scaler: GradScaler instance for mixed precision training (None for FP32)
        gradient_clip_norm: Maximum norm for gradient clipping
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    use_mixed_precision = scaler is not None
    
    for inputs, targets in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.squeeze().long().to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_mixed_precision:
            # Mixed precision training for ViT/Swin models
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard FP32 training for CNN models
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        _, pred = torch.max(outputs, 1)
        correct += (pred == targets).sum().item()
        running_loss += loss.item()
        total += targets.size(0)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc

def evaluate(model, data_loader, criterion, device, use_mixed_precision=False):
    """Evaluate model on dataset - optimized for TissueMNIST (multi-class)
    
    Args:
        use_mixed_precision: If True, use FP16 mixed precision evaluation (for ViT/Swin models)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.squeeze().long().to(device, non_blocking=True)
            
            if use_mixed_precision:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            _, pred = torch.max(outputs, 1)
            correct += (pred == targets).sum().item()
            running_loss += loss.item()
            total += targets.size(0)
    
    epoch_loss = running_loss / len(data_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc

def train_model(model, model_name, train_loader, val_loader, test_loader, config, criterion):
    """Train a single model"""
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    
    # Determine if model should use mixed precision (ViT and Swin models)
    use_mixed_precision = model_name in ['ViT-B/16', 'Swin-Tiny', 'Swin-Base']
    if use_mixed_precision:
        print("Using mixed precision training (FP16) for faster training")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Initialize scaler for mixed precision training (only for ViT/Swin models)
    scaler = GradScaler() if use_mixed_precision else None
    
    # Initialize early stopper
    early_stopper = EarlyStopper(patience=3, min_delta=0.001)
    
    # Initialize TensorBoard writer
    print("Initializing TensorBoard writer...", end=' ', flush=True)
    writer = SummaryWriter(f'runs/{model_name}')
    print("✓")
    
    # Training history
    history = {
        'train_losses': [],
        'train_accuracies': [],
        'test_losses': [],
        'test_accuracies': []
    }
    
    best_test_acc = 0.0
    best_epoch = 0
    
    # Test DataLoader before training (helps identify hangs)
    print("Testing DataLoader (fetching first batch)...", end=' ', flush=True)
    try:
        first_batch = next(iter(train_loader))
        print(f"✓ (batch size: {first_batch[0].shape[0]})")
    except Exception as e:
        print(f"✗ Error: {e}")
        raise
    
    # Training loop
    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, scaler, config.GRADIENT_CLIP_NORM
        )
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, config.DEVICE, use_mixed_precision
        )
        
        # Evaluate on test set (for monitoring, but use val for early stopping)
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, config.DEVICE, use_mixed_precision
        )
        
        # Update history
        history['train_losses'].append(train_loss)
        history['train_accuracies'].append(train_acc)
        history['test_losses'].append(test_loss)
        history['test_accuracies'].append(test_acc)
        
        # Track best model based on validation accuracy
        if val_acc > best_test_acc:
            best_test_acc = val_acc
            best_epoch = epoch + 1
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar(f'{model_name}/Loss/train', train_loss, epoch)
            writer.add_scalar(f'{model_name}/Loss/val', val_loss, epoch)
            writer.add_scalar(f'{model_name}/Loss/test', test_loss, epoch)
            writer.add_scalar(f'{model_name}/Accuracy/train', train_acc, epoch)
            writer.add_scalar(f'{model_name}/Accuracy/val', val_acc, epoch)
            writer.add_scalar(f'{model_name}/Accuracy/test', test_acc, epoch)
        
        # Print progress
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"  Best Val Acc: {best_test_acc:.2f}% (Epoch {best_epoch})")
        print("-" * 70)
        
        # Early stopping check (use validation loss)
        if early_stopper.early_stop(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best validation loss: {early_stopper.best_loss:.4f}")
            print(f"Stopped after {early_stopper.counter} epochs without improvement")
            break
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    history['best_test_accuracy'] = best_test_acc
    history['best_epoch'] = best_epoch
    history['num_parameters'] = num_params
    
    return history

# Evaluation with Metrics
def evaluate_with_metrics(model, data_loader, device, num_classes):
    """Evaluate model and return confusion matrix and classification report"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.squeeze().cpu().numpy())
    
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, 
                                   target_names=TISSUEMNIST_LABELS)
    
    return cm, report

# Evaluation with MedMNIST Evaluator
def evaluate_with_medmnist(model, model_name, data_loader, split, device, dataset_path=None):
    """Evaluate using MedMNIST evaluator - optimized for TissueMNIST
    
    Args:
        model: Model to evaluate
        model_name: Name of the model
        data_loader: DataLoader for evaluation
        split: Dataset split ('train', 'val', 'test')
        device: Device to run evaluation on
        dataset_path: Path to the dataset directory (same as used for loading)
    """
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(inputs)
            outputs = outputs.softmax(dim=-1)
            
            y_score = torch.cat((y_score, outputs.cpu()), 0)
            if targets.dim() > 1:
                targets = targets.squeeze()
            y_true = torch.cat((y_true, targets.cpu()), 0)
    
    y_score = y_score.detach().numpy()
    y_true = y_true.detach().numpy()
    
    # Use the same dataset path as used for loading the dataset
    # This ensures the Evaluator looks in the correct location
    if dataset_path:
        # Try to pass root parameter first (most direct approach)
        try:
            evaluator = Evaluator('tissuemnist', split, size=224, root=dataset_path)
        except (TypeError, AttributeError):
            # If Evaluator doesn't accept root parameter, try setting root attribute after creation
            try:
                evaluator = Evaluator('tissuemnist', split, size=224)
                # Try to set root attribute if it exists
                if hasattr(evaluator, 'root'):
                    evaluator.root = dataset_path
                elif hasattr(evaluator, '_root'):
                    evaluator._root = dataset_path
            except FileNotFoundError as e:
                # Evaluator is looking in wrong location, try to find dataset file
                dataset_file = os.path.join(dataset_path, 'tissuemnist_224.npz')
                if os.path.exists(dataset_file):
                    # File exists but Evaluator can't find it - provide helpful error
                    error_msg = (
                        f"Evaluator cannot find dataset file. "
                        f"Dataset exists at: {dataset_file}, "
                        f"but Evaluator is looking elsewhere. "
                        f"Original error: {str(e)}"
                    )
                    raise FileNotFoundError(error_msg) from e
                else:
                    # Dataset file not found
                    raise FileNotFoundError(
                        f"Dataset file not found at {dataset_path}. "
                        f"Expected file: tissuemnist_224.npz. "
                        f"Original error: {str(e)}"
                    )
    else:
        evaluator = Evaluator('tissuemnist', split, size=224)
    
    try:
        metrics = evaluator.evaluate(y_score, y_true)
    except TypeError:
        metrics = evaluator.evaluate(y_score)
    
    print(f"{model_name} - {split.upper()} Results:")
    print(f"  AUC: {metrics[0]:.3f}, Accuracy: {metrics[1]:.3f}")
    
    return metrics

# Visualization
def plot_training_curves(all_histories, config, save_path):
    """Plot training curves for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = range(1, config.NUM_EPOCHS + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_histories)))
    
    # Loss comparison
    ax = axes[0, 0]
    for (model_name, history), color in zip(all_histories.items(), colors):
        ax.plot(epochs, history['train_losses'], '--', color=color, alpha=0.7, label=f'{model_name} Train')
        ax.plot(epochs, history['test_losses'], '-', color=color, linewidth=2, label=f'{model_name} Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Test Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Accuracy comparison
    ax = axes[0, 1]
    for (model_name, history), color in zip(all_histories.items(), colors):
        ax.plot(epochs, history['train_accuracies'], '--', color=color, alpha=0.7, label=f'{model_name} Train')
        ax.plot(epochs, history['test_accuracies'], '-', color=color, linewidth=2, label=f'{model_name} Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training and Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Final test accuracy comparison (bar chart)
    ax = axes[1, 0]
    model_names = list(all_histories.keys())
    final_test_accs = [all_histories[name]['test_accuracies'][-1] for name in model_names]
    bars = ax.bar(model_names, final_test_accs, color=colors[:len(model_names)])
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, final_test_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom')
    
    # Best test accuracy comparison
    ax = axes[1, 1]
    best_test_accs = [all_histories[name]['best_test_accuracy'] for name in model_names]
    bars = ax.bar(model_names, best_test_accs, color=colors[:len(model_names)])
    ax.set_ylabel('Best Test Accuracy (%)')
    ax.set_title('Best Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, best_test_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training curves to {save_path}")

def plot_accuracy_curves(all_histories, config, save_path):
    """Plot focused accuracy curves"""
    num_models = len(all_histories)
    fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5))
    
    if num_models == 1:
        axes = [axes]
    
    epochs = range(1, config.NUM_EPOCHS + 1)
    
    for idx, (model_name, history) in enumerate(all_histories.items()):
        ax = axes[idx]
        ax.plot(epochs, history['train_accuracies'], 'b-o', 
                label='Training Accuracy', linewidth=2, markersize=6)
        ax.plot(epochs, history['test_accuracies'], 'r-s', 
                label='Testing Accuracy', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'{model_name}\nTraining and Testing Accuracy', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(epochs)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved accuracy curves to {save_path}")

# Single Model Results Saving
def save_single_model_results(model_name, history, metrics, cm, report, config, timestamp):
    """Save results for a single model immediately after training"""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Save individual model JSON
    model_results = {
        'timestamp': timestamp,
        'model_name': model_name,
        'dataset': 'TissueMNIST',
        'config': {
            'num_epochs': config.NUM_EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'use_pretrained': config.USE_PRETRAINED,
            'num_classes': TISSUEMNIST_NUM_CLASSES,
            'task': TISSUEMNIST_TASK,
            'labels': TISSUEMNIST_LABELS
        },
        'training_history': history,
        'test_metrics': {
            'auc': float(metrics[0]),
            'accuracy': float(metrics[1])
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    model_name_clean = model_name.replace(' ', '_').replace('/', '_').lower()
    json_path = os.path.join(config.RESULTS_DIR, f'{model_name_clean}_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(model_results, f, indent=2)
    
    if config.VERBOSE:
        print(f"  ✓ Saved individual results to {json_path}")

# Results Saving
def save_results(all_histories, all_metrics, config):
    """Save all results to JSON and generate report - optimized for TissueMNIST"""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Prepare results dictionary
    results = {
        'timestamp': timestamp,
        'dataset': 'TissueMNIST',
        'config': {
            'num_epochs': config.NUM_EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'use_pretrained': config.USE_PRETRAINED,
            'num_classes': TISSUEMNIST_NUM_CLASSES,
            'task': TISSUEMNIST_TASK,
            'labels': TISSUEMNIST_LABELS
        },
        'models': {}
    }
    
    for model_name in all_histories.keys():
        results['models'][model_name] = {
            'training_history': all_histories[model_name],
            'test_metrics': {
                'auc': float(all_metrics[model_name][0]),
                'accuracy': float(all_metrics[model_name][1])
            }
        }
    
    # Save JSON
    json_path = os.path.join(config.RESULTS_DIR, f'results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {json_path}")
    
    # Generate text report
    report_path = os.path.join(config.RESULTS_DIR, f'report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TISSUEMNIST COMPARATIVE STUDY RESULTS REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: TissueMNIST\n")
        f.write(f"Number of Classes: {TISSUEMNIST_NUM_CLASSES}\n")
        f.write(f"Task Type: {TISSUEMNIST_TASK}\n")
        f.write(f"Classes: {', '.join(TISSUEMNIST_LABELS)}\n")
        f.write(f"Number of Epochs: {config.NUM_EPOCHS}\n")
        f.write(f"Batch Size: {config.BATCH_SIZE}\n")
        f.write(f"Learning Rate: {config.LEARNING_RATE}\n")
        f.write(f"Use Pretrained: {config.USE_PRETRAINED}\n\n")
        
        f.write("="*70 + "\n")
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        # Sort by test accuracy
        sorted_models = sorted(
            all_histories.items(),
            key=lambda x: x[1]['best_test_accuracy'],
            reverse=True
        )
        
        for rank, (model_name, history) in enumerate(sorted_models, 1):
            metrics = all_metrics[model_name]
            f.write(f"\n{rank}. {model_name}\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Parameters: {history['num_parameters']/1e6:.2f}M\n")
            f.write(f"  Best Test Accuracy: {history['best_test_accuracy']:.2f}% (Epoch {history['best_epoch']})\n")
            f.write(f"  Final Test Accuracy: {history['test_accuracies'][-1]:.2f}%\n")
            f.write(f"  Final Train Accuracy: {history['train_accuracies'][-1]:.2f}%\n")
            f.write(f"  MedMNIST Test AUC: {metrics[0]:.3f}\n")
            f.write(f"  MedMNIST Test Accuracy: {metrics[1]:.3f}\n")
    
    print(f"✓ Saved report to {report_path}")
    
    return json_path, report_path

# Research Paper Deliverables
def save_performance_table(all_histories, all_metrics, config, timestamp):
    """Save performance comparison table in CSV and LaTeX formats"""
    paper_dir = os.path.join(config.RESULTS_DIR, config.PAPER_DIR)
    os.makedirs(paper_dir, exist_ok=True)
    
    # Prepare table data
    table_data = []
    sorted_models = sorted(
        all_histories.items(),
        key=lambda x: x[1]['best_test_accuracy'],
        reverse=True
    )
    
    for rank, (model_name, history) in enumerate(sorted_models, 1):
        metrics = all_metrics[model_name]
        table_data.append({
            'Rank': rank,
            'Model': model_name,
            'Parameters (M)': f"{history['num_parameters']/1e6:.2f}",
            'Best Test Acc (%)': f"{history['best_test_accuracy']:.2f}",
            'Final Test Acc (%)': f"{history['test_accuracies'][-1]:.2f}",
            'Final Train Acc (%)': f"{history['train_accuracies'][-1]:.2f}",
            'Test AUC': f"{metrics[0]:.3f}",
            'Test Accuracy': f"{metrics[1]:.3f}",
            'Best Epoch': history['best_epoch']
        })
    
    # Save CSV
    csv_path = os.path.join(paper_dir, f'performance_table_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=table_data[0].keys())
        writer.writeheader()
        writer.writerows(table_data)
    print(f"✓ Saved performance table (CSV) to {csv_path}")
    
    # Save LaTeX table
    latex_path = os.path.join(paper_dir, f'performance_table_{timestamp}.tex')
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance Comparison of Different Models on TissueMNIST}\n")
        f.write("\\label{tab:performance_comparison}\n")
        f.write("\\begin{tabular}{lccccccc}\n")
        f.write("\\toprule\n")
        f.write("Model & Params (M) & Best Test Acc & Final Test Acc & Train Acc & AUC & Accuracy & Best Epoch \\\\\n")
        f.write("\\midrule\n")
        
        for row in table_data:
            f.write(f"{row['Model']} & {row['Parameters (M)']} & {row['Best Test Acc (%)']} & "
                   f"{row['Final Test Acc (%)']} & {row['Final Train Acc (%)']} & "
                   f"{row['Test AUC']} & {row['Test Accuracy']} & {row['Best Epoch']} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"✓ Saved performance table (LaTeX) to {latex_path}")
    
    return csv_path, latex_path

def save_sample_images(train_loader, config, timestamp, num_samples_per_class=3):
    """Save sample images from each class for paper visualization - TissueMNIST"""
    paper_dir = os.path.join(config.RESULTS_DIR, config.PAPER_DIR, 'sample_images')
    os.makedirs(paper_dir, exist_ok=True)
    
    # Collect samples from each class
    class_samples = {i: [] for i in range(TISSUEMNIST_NUM_CLASSES)}
    
    for inputs, targets in train_loader:
        targets = targets.squeeze().long()
        for img, label in zip(inputs, targets):
            label_idx = label.item()
            if len(class_samples[label_idx]) < num_samples_per_class:
                class_samples[label_idx].append(img)
                if all(len(samples) >= num_samples_per_class for samples in class_samples.values()):
                    break
        if all(len(samples) >= num_samples_per_class for samples in class_samples.values()):
            break
    
    # Save individual sample images
    for class_idx, samples in class_samples.items():
        class_name = TISSUEMNIST_LABELS[class_idx]
        for sample_idx, img_tensor in enumerate(samples):
            # Denormalize image
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_denorm = img_tensor * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1)
            
            # Convert to PIL and save
            img_np = img_denorm.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            filename = f'class_{class_idx}_{class_name}_sample_{sample_idx+1}.png'
            filepath = os.path.join(paper_dir, filename)
            img_pil.save(filepath, dpi=300)
    
    # Create a montage of all samples
    fig, axes = plt.subplots(TISSUEMNIST_NUM_CLASSES, num_samples_per_class, 
                             figsize=(num_samples_per_class * 2, TISSUEMNIST_NUM_CLASSES * 2))
    
    for class_idx, samples in class_samples.items():
        class_name = TISSUEMNIST_LABELS[class_idx]
        for sample_idx, img_tensor in enumerate(samples):
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_denorm = img_tensor * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1)
            
            img_np = img_denorm.permute(1, 2, 0).numpy()
            axes[class_idx, sample_idx].imshow(img_np)
            axes[class_idx, sample_idx].set_title(f'{class_name}' if sample_idx == 0 else '', 
                                                   fontsize=8)
            axes[class_idx, sample_idx].axis('off')
    
    plt.suptitle('Sample Images from Each Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    montage_path = os.path.join(paper_dir, f'sample_images_montage_{timestamp}.png')
    plt.savefig(montage_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved sample images montage to {montage_path}")
    
    return paper_dir

def save_publication_figures(all_histories, config, timestamp):
    """Save high-quality figures for publication"""
    paper_dir = os.path.join(config.RESULTS_DIR, config.PAPER_DIR, 'figures')
    os.makedirs(paper_dir, exist_ok=True)
    
    epochs = range(1, config.NUM_EPOCHS + 1)
    
    # 1. Training and Test Loss Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_histories)))
    
    for (model_name, history), color in zip(all_histories.items(), colors):
        ax.plot(epochs, history['train_losses'], '--', color=color, alpha=0.7, 
               linewidth=2, label=f'{model_name} Train')
        ax.plot(epochs, history['test_losses'], '-', color=color, linewidth=2.5, 
               label=f'{model_name} Test')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Test Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(paper_dir, f'loss_comparison_{timestamp}.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight', format='png')
    plt.savefig(loss_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"✓ Saved loss comparison figure to {loss_path}")
    
    # 2. Training and Test Accuracy Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for (model_name, history), color in zip(all_histories.items(), colors):
        ax.plot(epochs, history['train_accuracies'], '--', color=color, alpha=0.7, 
               linewidth=2, label=f'{model_name} Train')
        ax.plot(epochs, history['test_accuracies'], '-', color=color, linewidth=2.5, 
               label=f'{model_name} Test')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Training and Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(paper_dir, f'accuracy_comparison_{timestamp}.png')
    plt.savefig(acc_path, dpi=300, bbox_inches='tight', format='png')
    plt.savefig(acc_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"✓ Saved accuracy comparison figure to {acc_path}")
    
    # 3. Final Test Accuracy Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(all_histories.keys())
    final_test_accs = [all_histories[name]['test_accuracies'][-1] for name in model_names]
    bars = ax.bar(model_names, final_test_accs, color=colors[:len(model_names)])
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, final_test_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    bar_path = os.path.join(paper_dir, f'accuracy_bar_chart_{timestamp}.png')
    plt.savefig(bar_path, dpi=300, bbox_inches='tight', format='png')
    plt.savefig(bar_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"✓ Saved accuracy bar chart to {bar_path}")
    
    return paper_dir

def save_paper_deliverables(all_histories, all_metrics, config, train_loader, timestamp):
    """Save all research paper deliverables - optimized for TissueMNIST"""
    if not config.SAVE_PAPER_DELIVERABLES:
        return
    
    print("\n" + "="*70)
    print("SAVING RESEARCH PAPER DELIVERABLES")
    print("="*70)
    
    # Save performance tables
    save_performance_table(all_histories, all_metrics, config, timestamp)
    
    # Save sample images
    save_sample_images(train_loader, config, timestamp)
    
    # Save publication figures
    save_publication_figures(all_histories, config, timestamp)
    
    paper_dir = os.path.join(config.RESULTS_DIR, config.PAPER_DIR)
    print(f"\n✓ All paper deliverables saved to: {paper_dir}")
    print(f"  - Performance tables (CSV & LaTeX)")
    print(f"  - Sample images (individual & montage)")
    print(f"  - Publication figures (PNG & PDF)")

# Model Saving
def save_models(models, all_histories, config):
    """Save all trained models"""
    if not config.SAVE_MODELS:
        return
    
    print(f"Attempting to save models to: {os.path.abspath(config.MODEL_SAVE_DIR)}")
    
    # Check if directory can be created/accessed
    try:
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        # Test write permissions
        test_file = os.path.join(config.MODEL_SAVE_DIR, '.write_test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"✓ Directory is writable: {os.path.abspath(config.MODEL_SAVE_DIR)}")
        except (IOError, OSError) as e:
            print(f"✗ ERROR: Cannot write to directory {config.MODEL_SAVE_DIR}: {e}")
            print(f"  Please check permissions or disk quota")
            return
    except (OSError, PermissionError) as e:
        print(f"✗ ERROR: Cannot create directory {config.MODEL_SAVE_DIR}: {e}")
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    saved_count = 0
    failed_count = 0
    
    for model_name, model in models.items():
        model_name_clean = model_name.replace(' ', '_').replace('/', '_').lower()
        save_path = os.path.join(
            config.MODEL_SAVE_DIR,
            f'{model_name_clean}_{timestamp}.pt'
        )
        
        try:
            # Check available disk space (if possible)
            try:
                import shutil
                stat = shutil.disk_usage(os.path.dirname(os.path.abspath(save_path)))
                free_gb = stat.free / (1024**3)
                if free_gb < 0.1:  # Less than 100MB free
                    print(f"⚠ WARNING: Low disk space ({free_gb:.2f} GB free) for {model_name}")
            except:
                pass  # Skip disk space check if not available
            
            # Get state_dict and move to CPU (important for CUDA models)
            # This ensures models can be loaded on any device later
            state_dict = model.state_dict()
            # Move all tensors in state_dict to CPU
            cpu_state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                            for k, v in state_dict.items()}
            
            # Prepare checkpoint data
            checkpoint = {
                'model_state_dict': cpu_state_dict,  # Use CPU state_dict
                'model_name': model_name, 
                'training_history': all_histories.get(model_name, {}),
                'config': {
                    'num_epochs': config.NUM_EPOCHS,
                    'batch_size': config.BATCH_SIZE,
                    'learning_rate': config.LEARNING_RATE,
                    'use_pretrained': config.USE_PRETRAINED
                }
            }
            
            # Save the model (using CPU tensors)
            torch.save(checkpoint, save_path)
            
            # Verify the file was created and has content
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                file_size_mb = os.path.getsize(save_path) / (1024**2)
                print(f"✓ Saved {model_name} to {save_path} ({file_size_mb:.2f} MB)")
                saved_count += 1
            else:
                print(f"✗ ERROR: File was not created properly for {model_name}")
                failed_count += 1
                
        except OSError as e:
            if e.errno == 122:  # Disk quota exceeded
                print(f"✗ ERROR: Disk quota exceeded while saving {model_name}")
                print(f"  Path: {save_path}")
                print(f"  Error: {e}")
            else:
                print(f"✗ ERROR: OS error while saving {model_name}: {e}")
            failed_count += 1
        except RuntimeError as e:
            # Catch CUDA-related errors
            if 'cuda' in str(e).lower() or 'CUDA' in str(e):
                print(f"✗ ERROR: CUDA error while saving {model_name}: {e}")
                print(f"  Attempting to save with model moved to CPU...")
                try:
                    # Try saving with model temporarily on CPU
                    model_cpu = model.cpu()
                    cpu_state_dict = model_cpu.state_dict()
                    model_cpu = model_cpu.to(config.DEVICE)  # Move back to original device
                    
                    checkpoint = {
                        'model_state_dict': cpu_state_dict,
                        'model_name': model_name, 
                        'training_history': all_histories.get(model_name, {}),
                        'config': {
                            'num_epochs': config.NUM_EPOCHS,
                            'batch_size': config.BATCH_SIZE,
                            'learning_rate': config.LEARNING_RATE,
                            'use_pretrained': config.USE_PRETRAINED
                        }
                    }
                    torch.save(checkpoint, save_path)
                    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                        file_size_mb = os.path.getsize(save_path) / (1024**2)
                        print(f"✓ Saved {model_name} to {save_path} ({file_size_mb:.2f} MB) [CPU fallback]")
                        saved_count += 1
                    else:
                        failed_count += 1
                except Exception as e2:
                    print(f"✗ ERROR: Failed to save {model_name} even with CPU fallback: {e2}")
                    failed_count += 1
            else:
                print(f"✗ ERROR: Runtime error while saving {model_name}: {e}")
                failed_count += 1
        except Exception as e:
            print(f"✗ ERROR: Unexpected error while saving {model_name}: {e}")
            print(f"  Error type: {type(e).__name__}")
            failed_count += 1
    
    # Summary
    print(f"\nModel saving summary: {saved_count} saved, {failed_count} failed")
    if saved_count > 0:
        print(f"✓ Models saved to: {os.path.abspath(config.MODEL_SAVE_DIR)}")
    if failed_count > 0:
        print(f"⚠ WARNING: {failed_count} model(s) failed to save. Check disk quota and permissions.")
# Main Training Pipeline
def main():
    """Main training and evaluation pipeline - optimized for TissueMNIST"""
    # Create config first to load environment variables
    config = Config()
    
    # Set seed for reproducibility (use config value, or skip if -1)
    if config.RANDOM_SEED >= 0:
        set_seed(config.RANDOM_SEED)
    
    print("="*70)
    print("TISSUEMNIST COMPARATIVE STUDY: CNN vs Transformer Models")
    print("="*70)
    print(f"Dataset: TissueMNIST ({TISSUEMNIST_NUM_CLASSES} classes)")
    print(f"Device: {config.DEVICE}")
    print(f"Models to train: {', '.join(config.MODELS_TO_TRAIN)}")
    print(f"Save models: {config.SAVE_MODELS} (directory: {config.MODEL_SAVE_DIR})")
    if config.RANDOM_SEED >= 0:
        print(f"Random seed: {config.RANDOM_SEED} (for reproducibility)")
    else:
        print("Random seed: Disabled")
    print("="*70)
    
    # Load TissueMNIST dataset
    print("\nLoading TissueMNIST dataset...")
    train_loader, val_loader, test_loader, dataset_path = load_tissuemnist(config)
    print(f"✓ TissueMNIST dataset loaded")
    print(f"  Dataset path: {dataset_path}")
    print(f"  Number of classes: {TISSUEMNIST_NUM_CLASSES}")
    print(f"  Task type: {TISSUEMNIST_TASK}")
    print(f"  Classes: {', '.join(TISSUEMNIST_LABELS)}")
    print(f"  Train/Val/Test split: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)} samples")
    
    # Create models
    print("\nInitializing models...")
    models = create_models(config)
    print(f"✓ Initialized {len(models)} models")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize accumulators for final summary
    all_histories = {}
    all_metrics = {}
    
    # Shared timestamp for this run
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Process each model completely before moving to next
    print("\n" + "="*70)
    print("PROCESSING MODELS (Train → Evaluate → Save → Cleanup)")
    print("="*70)
    
    for model_idx, (model_name, model) in enumerate(models.items(), 1):
        print("\n" + "="*70)
        print(f"MODEL {model_idx}/{len(models)}: {model_name}")
        print("="*70)
        
        try:
            # 1. Train model
            print(f"\n[1/5] Training {model_name}...")
            history = train_model(
                model, model_name, train_loader, val_loader, test_loader,
                config, criterion
            )
            all_histories[model_name] = history
            print(f"✓ Training complete for {model_name}")
            
            # 2. Evaluate with MedMNIST evaluator
            print(f"\n[2/5] Evaluating {model_name} with MedMNIST evaluator...")
            metrics = evaluate_with_medmnist(
                model, model_name, test_loader, 'test',
                config.DEVICE, dataset_path
            )
            all_metrics[model_name] = metrics
            
            # 3. Get confusion matrix and classification report
            print(f"\n[3/5] Computing detailed metrics for {model_name}...")
            cm, report = evaluate_with_metrics(
                model, test_loader, config.DEVICE, TISSUEMNIST_NUM_CLASSES
            )
            
            print(f"\n{model_name} - Classification Report:")
            print(report)
            
            # 4. Save model checkpoint immediately
            print(f"\n[4/5] Saving {model_name} checkpoint...")
            if config.SAVE_MODELS:
                temp_models = {model_name: model}
                temp_histories = {model_name: history}
                save_models(temp_models, temp_histories, config)
                print(f"✓ Checkpoint saved for {model_name}")
            else:
                print(f"⚠ Model saving disabled (SAVE_MODELS=False)")
            
            # 5. Save per-model deliverables
            print(f"\n[5/5] Saving {model_name} deliverables...")
            save_single_model_results(model_name, history, metrics, cm, report, config, run_timestamp)
            print(f"✓ Deliverables saved for {model_name}")
            
            # Cleanup: Move model to CPU to free GPU memory
            print(f"\nCleaning up memory for {model_name}...", end=' ', flush=True)
            if next(model.parameters()).is_cuda:
                model = model.cpu()
                models[model_name] = model  # Update in dict
                if torch.cuda.is_available() and config.DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
                    if config.VERBOSE:
                        allocated = torch.cuda.memory_allocated() / (1024**3)
                        reserved = torch.cuda.memory_reserved() / (1024**3)
                        print(f"✓ (GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved)")
                    else:
                        print("✓")
                else:
                    print("✓")
            else:
                print("✓ (already on CPU)")
            
            print(f"\n{'='*70}")
            print(f"✓ {model_name} processing complete!")
            print(f"{'='*70}")
            
        except KeyboardInterrupt:
            print(f"\n\n⚠ Training interrupted by user (Ctrl+C) during {model_name}")
            print(f"⚠ {model_name} may not have been saved. Check if training completed.")
            raise  # Re-raise to exit gracefully
        except Exception as e:
            print(f"\n✗ ERROR processing {model_name}: {e}")
            print(f"  Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print(f"\n⚠ Continuing with next model...")
            # Mark this model as failed but continue
            all_histories[model_name] = {'error': str(e), 'failed': True}
            continue
    
    # Final summary and combined deliverables
    print("\n" + "="*70)
    print("GENERATING FINAL SUMMARY AND COMBINED DELIVERABLES")
    print("="*70)
    
    # Filter out failed models for final deliverables
    successful_models = {k: v for k, v in all_histories.items() if not v.get('failed', False)}
    
    if len(successful_models) == 0:
        print("\n⚠ WARNING: No models were successfully trained!")
        print("⚠ Check error messages above for details.")
        return
    
    # Save combined results
    print("\n[1/4] Saving combined results...")
    successful_metrics = {k: v for k, v in all_metrics.items() if k in successful_models}
    json_path, report_path = save_results(successful_models, successful_metrics, config)
    
    # Generate combined visualizations
    print("\n[2/4] Generating combined visualizations...")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    curves_path = os.path.join(config.RESULTS_DIR, f'training_curves_{timestamp}.png')
    accuracy_path = os.path.join(config.RESULTS_DIR, f'accuracy_curves_{timestamp}.png')
    
    plot_training_curves(successful_models, config, curves_path)
    plot_accuracy_curves(successful_models, config, accuracy_path)
    print("✓ Visualizations generated")
    
    # Save combined paper deliverables
    print("\n[3/4] Saving combined paper deliverables...")
    if config.SAVE_PAPER_DELIVERABLES:
        save_paper_deliverables(successful_models, successful_metrics, config, train_loader, timestamp)
        print("✓ Paper deliverables saved")
    else:
        print("⚠ Paper deliverables disabled (SAVE_PAPER_DELIVERABLES=False)")
    
    # Final summary
    print("\n[4/4] Generating final summary...")
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    # Filter successful models for ranking
    successful_histories = {k: v for k, v in all_histories.items() if not v.get('failed', False)}
    
    if len(successful_histories) > 0:
        sorted_models = sorted(
            successful_histories.items(),
            key=lambda x: x[1].get('best_test_accuracy', 0),
            reverse=True
        )
        
        print("\nRanking by Best Validation Accuracy:")
        for rank, (model_name, history) in enumerate(sorted_models, 1):
            best_acc = history.get('best_test_accuracy', 0)
            num_params = history.get('num_parameters', 0) / 1e6
            print(f"{rank}. {model_name}: {best_acc:.2f}% "
                  f"(Params: {num_params:.2f}M)")
    
    # Report failed models if any
    failed_models = [k for k, v in all_histories.items() if v.get('failed', False)]
    if failed_models:
        print(f"\n⚠ Failed Models ({len(failed_models)}):")
        for model_name in failed_models:
            print(f"  - {model_name}")
    
    print("\n" + "="*70)
    print("COMPARATIVE STUDY COMPLETE!")
    print("="*70)
    print(f"Successfully processed: {len(successful_histories)}/{len(models)} models")
    print(f"Results saved to: {config.RESULTS_DIR}")
    if config.SAVE_MODELS:
        print(f"Models saved to: {config.MODEL_SAVE_DIR}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user (Ctrl+C)")
        print("⚠ Models may not have been saved. Check if training completed.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ FATAL ERROR: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("\n⚠ Models may not have been saved due to this error.")
        sys.exit(1)