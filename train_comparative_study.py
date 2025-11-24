"""
Comparative Study Training and Testing Framework
================================================
This script trains and evaluates multiple CNN and Transformer models
on TissueMNIST dataset for comparative analysis.

Models included:
- CNN: ResNet18, ResNet50, DenseNet121, EfficientNet-B0
- Transformers: ViT-B/16, Swin-Tiny, Swin-Base
"""

import os
import sys
import warnings
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import medmnist
from medmnist import INFO, Evaluator

# Import models from tissue_main.py
from tissue_main import (
    ResNet50Classifier,
    DenseNet121Classifier,
    EfficientNetClassifier,
    ViTClassifier,
    SwinTransformerClassifier
)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tqdm')

# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Dataset
    DATA_FLAG = 'tissuemnist'
    DATASET_PATH = None  # Will be auto-detected
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
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Models to train (can be modified to train subset)
    MODELS_TO_TRAIN = [
        'ResNet50',
        'DenseNet121',
        'EfficientNet-B0',
        'ViT-B/16',
        'Swin-Tiny',
        'Swin-Base'
    ]

# ============================================================================
# Data Loading
# ============================================================================
def get_data_transform():
    """Get data transformation for pretrained models"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Grayscale to RGB
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

def load_dataset(config):
    """Load TissueMNIST dataset"""
    info = INFO[config.DATA_FLAG]
    DataClass = getattr(medmnist, info['python_class'])
    
    # Determine dataset path
    current_dir = os.getcwd()
    if 'notebooks' in current_dir:
        project_root = os.path.dirname(current_dir)
    else:
        project_root = current_dir
    
    dataset_path = os.path.join(project_root, 'mnist_dataset')
    if not os.path.exists(dataset_path):
        # Try alternative path
        dataset_path = os.path.join(project_root, 'dataset')
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset path not found. Using default: {dataset_path}")
    
    data_transform = get_data_transform()
    
    train_dataset = DataClass(
        split='train',
        transform=data_transform,
        download=False,
        root=dataset_path,
        size=config.IMAGE_SIZE,
        mmap_mode='r'
    )
    
    test_dataset = DataClass(
        split='test',
        transform=data_transform,
        download=False,
        root=dataset_path,
        size=config.IMAGE_SIZE,
        mmap_mode='r'
    )
    
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0
    )
    
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0
    )
    
    return train_loader, test_loader, info

# ============================================================================
# Model Initialization
# ============================================================================
def create_models(config, num_classes):
    """Create all models for comparative study"""
    models = {}
    
    if 'ResNet50' in config.MODELS_TO_TRAIN:
        models['ResNet50'] = ResNet50Classifier(
            num_classes=num_classes,
            pretrained=config.USE_PRETRAINED
        ).to(config.DEVICE)
    
    if 'DenseNet121' in config.MODELS_TO_TRAIN:
        models['DenseNet121'] = DenseNet121Classifier(
            num_classes=num_classes,
            pretrained=config.USE_PRETRAINED
        ).to(config.DEVICE)
    
    if 'EfficientNet-B0' in config.MODELS_TO_TRAIN:
        models['EfficientNet-B0'] = EfficientNetClassifier(
            num_classes=num_classes,
            pretrained=config.USE_PRETRAINED
        ).to(config.DEVICE)
    
    if 'ViT-B/16' in config.MODELS_TO_TRAIN:
        models['ViT-B/16'] = ViTClassifier(
            num_classes=num_classes,
            model_name="google/vit-base-patch16-224",
            pretrained=config.USE_PRETRAINED
        ).to(config.DEVICE)
    
    if 'Swin-Tiny' in config.MODELS_TO_TRAIN:
        models['Swin-Tiny'] = SwinTransformerClassifier(
            num_classes=num_classes,
            model_name="microsoft/swin-tiny-patch4-window7-224",
            pretrained=config.USE_PRETRAINED
        ).to(config.DEVICE)
    
    if 'Swin-Base' in config.MODELS_TO_TRAIN:
        models['Swin-Base'] = SwinTransformerClassifier(
            num_classes=num_classes,
            model_name="microsoft/swin-base-patch4-window7-224",
            pretrained=config.USE_PRETRAINED
        ).to(config.DEVICE)
    
    return models

# ============================================================================
# Training Functions
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device, task):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
            pred = (torch.sigmoid(outputs) > 0.5).int()
            correct += (pred == targets.int()).all(dim=1).sum().item()
        else:
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
            _, pred = torch.max(outputs, 1)
            correct += (pred == targets).sum().item()
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        total += targets.size(0)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc

def evaluate(model, data_loader, criterion, device, task):
    """Evaluate model on dataset"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
                pred = (torch.sigmoid(outputs) > 0.5).int()
                correct += (pred == targets.int()).all(dim=1).sum().item()
            else:
                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)
                _, pred = torch.max(outputs, 1)
                correct += (pred == targets).sum().item()
            
            running_loss += loss.item()
            total += targets.size(0)
    
    epoch_loss = running_loss / len(data_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc

def train_model(model, model_name, train_loader, test_loader, config, task, criterion):
    """Train a single model"""
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    
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
    
    # Training history
    history = {
        'train_losses': [],
        'train_accuracies': [],
        'test_losses': [],
        'test_accuracies': []
    }
    
    best_test_acc = 0.0
    best_epoch = 0
    
    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, task
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, config.DEVICE, task
        )
        
        # Update history
        history['train_losses'].append(train_loss)
        history['train_accuracies'].append(train_acc)
        history['test_losses'].append(test_loss)
        history['test_accuracies'].append(test_acc)
        
        # Track best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
        
        # Print progress
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"  Best Test Acc: {best_test_acc:.2f}% (Epoch {best_epoch})")
        print("-" * 70)
    
    history['best_test_accuracy'] = best_test_acc
    history['best_epoch'] = best_epoch
    history['num_parameters'] = num_params
    
    return history

# ============================================================================
# Evaluation with MedMNIST Evaluator
# ============================================================================
def evaluate_with_medmnist(model, model_name, data_loader, split, device, task, data_flag):
    """Evaluate using MedMNIST evaluator"""
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            outputs = outputs.softmax(dim=-1)
            
            y_score = torch.cat((y_score, outputs.cpu()), 0)
            if targets.dim() > 1:
                targets = targets.squeeze()
            y_true = torch.cat((y_true, targets.cpu()), 0)
    
    y_score = y_score.detach().numpy()
    y_true = y_true.detach().numpy()
    
    evaluator = Evaluator(data_flag, split, size=224)
    try:
        metrics = evaluator.evaluate(y_score, y_true)
    except TypeError:
        metrics = evaluator.evaluate(y_score)
    
    print(f"{model_name} - {split.upper()} Results:")
    print(f"  AUC: {metrics[0]:.3f}, Accuracy: {metrics[1]:.3f}")
    
    return metrics

# ============================================================================
# Visualization
# ============================================================================
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

# ============================================================================
# Results Saving
# ============================================================================
def save_results(all_histories, all_metrics, config, info):
    """Save all results to JSON and generate report"""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Prepare results dictionary
    results = {
        'timestamp': timestamp,
        'config': {
            'data_flag': config.DATA_FLAG,
            'num_epochs': config.NUM_EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'use_pretrained': config.USE_PRETRAINED,
            'num_classes': len(info['label']),
            'task': info['task']
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
        f.write("COMPARATIVE STUDY RESULTS REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {config.DATA_FLAG}\n")
        f.write(f"Number of Classes: {len(info['label'])}\n")
        f.write(f"Task Type: {info['task']}\n")
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

# ============================================================================
# Model Saving
# ============================================================================
def save_models(models, all_histories, config):
    """Save all trained models"""
    if not config.SAVE_MODELS:
        return
    
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    for model_name, model in models.items():
        model_name_clean = model_name.replace(' ', '_').replace('/', '_').lower()
        save_path = os.path.join(
            config.MODEL_SAVE_DIR,
            f'{model_name_clean}_{timestamp}.pt'
        )
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'training_history': all_histories[model_name],
            'config': {
                'num_epochs': config.NUM_EPOCHS,
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE,
                'use_pretrained': config.USE_PRETRAINED
            }
        }, save_path)
        print(f"✓ Saved {model_name} to {save_path}")

# ============================================================================
# Main Training Pipeline
# ============================================================================
def main():
    """Main training and evaluation pipeline"""
    config = Config()
    
    print("="*70)
    print("COMPARATIVE STUDY: CNN vs Transformer Models")
    print("="*70)
    print(f"Device: {config.DEVICE}")
    print(f"Models to train: {', '.join(config.MODELS_TO_TRAIN)}")
    print("="*70)
    
    # Load dataset
    print("\nLoading dataset...")
    train_loader, test_loader, info = load_dataset(config)
    num_classes = len(info['label'])
    print(f"✓ Dataset loaded: {config.DATA_FLAG}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Task type: {info['task']}")
    
    # Create models
    print("\nInitializing models...")
    models = create_models(config, num_classes)
    print(f"✓ Initialized {len(models)} models")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Train all models
    all_histories = {}
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    for model_name, model in models.items():
        history = train_model(
            model, model_name, train_loader, test_loader,
            config, info['task'], criterion
        )
        all_histories[model_name] = history
    
    # Evaluate with MedMNIST evaluator
    print("\n" + "="*70)
    print("FINAL EVALUATION WITH MEDMNIST EVALUATOR")
    print("="*70)
    all_metrics = {}
    for model_name, model in models.items():
        metrics = evaluate_with_medmnist(
            model, model_name, test_loader, 'test',
            config.DEVICE, info['task'], config.DATA_FLAG
        )
        all_metrics[model_name] = metrics
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    json_path, report_path = save_results(all_histories, all_metrics, config, info)
    
    # Generate visualizations
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    curves_path = os.path.join(config.RESULTS_DIR, f'training_curves_{timestamp}.png')
    accuracy_path = os.path.join(config.RESULTS_DIR, f'accuracy_curves_{timestamp}.png')
    
    plot_training_curves(all_histories, config, curves_path)
    plot_accuracy_curves(all_histories, config, accuracy_path)
    
    # Save models
    if config.SAVE_MODELS:
        print("\n" + "="*70)
        print("SAVING MODELS")
        print("="*70)
        save_models(models, all_histories, config)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    sorted_models = sorted(
        all_histories.items(),
        key=lambda x: x[1]['best_test_accuracy'],
        reverse=True
    )
    
    print("\nRanking by Best Test Accuracy:")
    for rank, (model_name, history) in enumerate(sorted_models, 1):
        print(f"{rank}. {model_name}: {history['best_test_accuracy']:.2f}% "
              f"(Params: {history['num_parameters']/1e6:.2f}M)")
    
    print("\n" + "="*70)
    print("COMPARATIVE STUDY COMPLETE!")
    print("="*70)
    print(f"Results saved to: {config.RESULTS_DIR}")
    if config.SAVE_MODELS:
        print(f"Models saved to: {config.MODEL_SAVE_DIR}")

if __name__ == '__main__':
    main()

