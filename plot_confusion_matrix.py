"""
Load Model Checkpoint and Plot Confusion Matrix
==============================================
This script loads a saved .pt checkpoint file, recreates the model,
runs inference on the test set, and plots a confusion matrix.

Supported Models:
    - CNN: ResNet18, ResNet50, DenseNet121, EfficientNet-B0
    - Transformers: ViT-B/16, DeiT-Base (facebook/deit-base-distilled-patch16-224), 
                    Swin-Tiny, Swin-Base

Usage:
    python plot_confusion_matrix.py <checkpoint_path> [--split test|val|train]
    
Example:
    python plot_confusion_matrix.py checkpoints/resnet18/best_model.pt
    python plot_confusion_matrix.py checkpoints/vit_b_16/best_model.pt --split test
    python plot_confusion_matrix.py checkpoints/deit_base/best_model.pt
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import from train_comparative_study
from train_comparative_study import (
    Config, load_tissuemnist, create_models,
    evaluate_with_metrics, TISSUEMNIST_NUM_CLASSES, TISSUEMNIST_LABELS
)
from tissue_main import (
    ResNet18Classifier, ResNet50Classifier, DenseNet121Classifier,
    EfficientNetClassifier, ViTClassifier, SwinTransformerClassifier
)


def load_checkpoint(checkpoint_path, device):
    """Load checkpoint from .pt file"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"✓ Checkpoint loaded")
    print(f"  Model name: {checkpoint.get('model_name', 'Unknown')}")
    print(f"  Timestamp: {checkpoint.get('timestamp', 'Unknown')}")
    
    return checkpoint


def recreate_model(model_name, checkpoint, device):
    """Recreate the model from checkpoint"""
    num_classes = TISSUEMNIST_NUM_CLASSES
    
    print(f"\nRecreating model: {model_name}")
    
    # Determine model class based on model_name
    if model_name == 'ResNet18':
        model = ResNet18Classifier(
            num_classes=num_classes,
            pretrained=checkpoint.get('config', {}).get('use_pretrained', True)
        )
    elif model_name == 'ResNet50':
        model = ResNet50Classifier(
            num_classes=num_classes,
            pretrained=checkpoint.get('config', {}).get('use_pretrained', True)
        )
    elif model_name == 'DenseNet121':
        model = DenseNet121Classifier(
            num_classes=num_classes,
            pretrained=checkpoint.get('config', {}).get('use_pretrained', True)
        )
    elif model_name == 'EfficientNet-B0':
        model = EfficientNetClassifier(
            num_classes=num_classes,
            pretrained=checkpoint.get('config', {}).get('use_pretrained', True)
        )
    elif model_name == 'ViT-B/16':
        model = ViTClassifier(
            num_classes=num_classes,
            model_name="google/vit-base-patch16-224",
            pretrained=checkpoint.get('config', {}).get('use_pretrained', True)
        )
    elif model_name in ['DeiT-Base', 'DeiT', 'facebook/deit-base-distilled-patch16-224']:
        # DeiT (Data-efficient Image Transformer) is compatible with ViT architecture
        model = ViTClassifier(
            num_classes=num_classes,
            model_name="facebook/deit-base-distilled-patch16-224",
            pretrained=checkpoint.get('config', {}).get('use_pretrained', True)
        )
    elif model_name == 'Swin-Tiny':
        model = SwinTransformerClassifier(
            num_classes=num_classes,
            model_name="microsoft/swin-tiny-patch4-window7-224",
            pretrained=checkpoint.get('config', {}).get('use_pretrained', True)
        )
    elif model_name == 'Swin-Base':
        model = SwinTransformerClassifier(
            num_classes=num_classes,
            model_name="microsoft/swin-base-patch4-window7-224",
            pretrained=checkpoint.get('config', {}).get('use_pretrained', True)
        )
    else:
        # Try to detect if it's a DeiT model from the model name or checkpoint
        if 'deit' in model_name.lower() or 'distilled' in model_name.lower():
            print(f"  Detected DeiT model from name: {model_name}")
            model = ViTClassifier(
                num_classes=num_classes,
                model_name="facebook/deit-base-distilled-patch16-224",
                pretrained=checkpoint.get('config', {}).get('use_pretrained', True)
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}. Supported models: ResNet18, ResNet50, DenseNet121, EfficientNet-B0, ViT-B/16, DeiT-Base, Swin-Tiny, Swin-Base")
    
    # Load state dict (filter out any thop-related keys if present)
    state_dict = checkpoint['model_state_dict']
    # Filter out thop-related keys that might have been accidentally saved
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                          if 'total_ops' not in k and 'total_params' not in k}
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model recreated and loaded to {device}")
    
    return model


def plot_confusion_matrix(cm, model_name, labels, save_path=None, show=True):
    """Plot confusion matrix with proper formatting"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )
    
    # Plot with custom formatting
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def print_classification_summary(cm, labels):
    """Print summary statistics from confusion matrix"""
    print("\n" + "="*70)
    print("CLASSIFICATION SUMMARY")
    print("="*70)
    
    # Calculate metrics
    total = cm.sum()
    correct = np.trace(cm)
    accuracy = correct / total if total > 0 else 0
    
    print(f"Overall Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    print(f"\nPer-Class Statistics:")
    print(f"{'Class':<40} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*70)
    
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = cm[i, :].sum()
        
        print(f"{label[:38]:<40} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f} {support:<10}")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Load checkpoint and plot confusion matrix',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_confusion_matrix.py checkpoints/resnet18/best_model.pt
  python plot_confusion_matrix.py checkpoints/vit_b_16/best_model.pt --split test
  python plot_confusion_matrix.py saved_models/resnet50_2024-01-15_10-30-45.pt --no-show
        """
    )
    parser.add_argument('checkpoint_path', type=str, help='Path to .pt checkpoint file')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], default='test',
                       help='Dataset split to evaluate on (default: test)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save confusion matrix plot (default: auto-generate)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display the plot (useful for saving only)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to run inference on (default: auto)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("="*70)
    print("LOAD CHECKPOINT AND PLOT CONFUSION MATRIX")
    print("="*70)
    print(f"Device: {device}")
    print(f"Dataset split: {args.split}")
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint_path, device)
    model_name = checkpoint.get('model_name', 'Unknown')
    
    # Recreate model
    model = recreate_model(model_name, checkpoint, device)
    
    # Load dataset
    print(f"\nLoading TissueMNIST dataset...")
    config = Config()
    train_loader, val_loader, test_loader, dataset_path = load_tissuemnist(config)
    
    # Select appropriate data loader
    if args.split == 'train':
        data_loader = train_loader
    elif args.split == 'val':
        data_loader = val_loader
    else:
        data_loader = test_loader
    
    print(f"✓ Dataset loaded ({args.split} split: {len(data_loader.dataset)} samples)")
    
    # Run inference and get confusion matrix
    print(f"\nRunning inference on {args.split} set...")
    cm, report = evaluate_with_metrics(model, data_loader, device, TISSUEMNIST_NUM_CLASSES)
    print(f"✓ Inference complete")
    
    # Print classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(report)
    
    # Print summary statistics
    print_classification_summary(cm, TISSUEMNIST_LABELS)
    
    # Determine save path
    if args.save is None:
        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        model_name_clean = model_name.replace(' ', '_').replace('/', '_').lower()
        save_path = os.path.join(checkpoint_dir, f'confusion_matrix_{args.split}.png')
    else:
        save_path = args.save
    
    # Plot confusion matrix
    print(f"\nPlotting confusion matrix...")
    plot_confusion_matrix(
        cm, model_name, TISSUEMNIST_LABELS,
        save_path=save_path, show=not args.no_show
    )
    
    print("\n" + "="*70)
    print("✓ Complete!")
    print("="*70)


if __name__ == '__main__':
    main()

