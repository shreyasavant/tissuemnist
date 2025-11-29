"""
Individual Model Runner for TissueMNIST
======================================
This script runs each model individually, saving all deliverables and checkpoints
in model-specific folders under checkpoints/ directory.

Recipes/Usage Examples:
    1. Run all models sequentially:
       python run.py
    
    2. Run a specific model from command line:
       python run.py ResNet18
       python run.py ResNet50
       python run.py DenseNet121
       python run.py EfficientNet-B0
       python run.py "ViT-B/16"
       python run.py Swin-Tiny
       python run.py Swin-Base
"""

import os
import sys
import shutil
from datetime import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Import from train_comparative_study
from train_comparative_study import (
    Config, set_seed, load_tissuemnist, create_models,
    train_model, evaluate_with_medmnist, evaluate_with_metrics,
    save_single_model_results, plot_training_curves, plot_accuracy_curves,
    save_performance_table, save_sample_images, save_publication_figures,
    TISSUEMNIST_NUM_CLASSES, TISSUEMNIST_LABELS
)


def save_model_checkpoint(model, model_name, history, config, checkpoint_dir, timestamp):
    """Save model checkpoint in model-specific directory"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get state_dict and move to CPU
    state_dict = model.state_dict()
    cpu_state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                     for k, v in state_dict.items()}
    
    checkpoint = {
        'model_state_dict': cpu_state_dict,
        'model_name': model_name,
        'training_history': history,
        'config': {
            'num_epochs': config.NUM_EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'use_pretrained': config.USE_PRETRAINED,
            'image_size': config.IMAGE_SIZE
        },
        'timestamp': timestamp
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name.replace(" ", "_").replace("/", "_").lower()}_checkpoint_{timestamp}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as 'best_model.pt' for easy access
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    torch.save(checkpoint, best_model_path)
    
    print(f"✓ Checkpoint saved to {checkpoint_path}")
    print(f"✓ Best model saved to {best_model_path}")
    
    return checkpoint_path


def save_model_visualizations(model_name, history, output_dir, timestamp, config):
    """Save training curves and visualizations for a single model"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a single-model history dict for plotting functions
    single_history = {model_name: history}
    
    # Create a minimal config object for plotting functions
    # Use actual num_epochs from history length
    num_epochs = len(history['train_losses'])
    plot_config = type('Config', (), {'NUM_EPOCHS': num_epochs})()
    
    # Training curves
    curves_path = os.path.join(output_dir, f'training_curves_{timestamp}.png')
    plot_training_curves(single_history, plot_config, curves_path)
    
    # Accuracy curves
    accuracy_path = os.path.join(output_dir, f'accuracy_curves_{timestamp}.png')
    plot_accuracy_curves(single_history, plot_config, accuracy_path)
    
    print(f"✓ Visualizations saved to {output_dir}")


def run_single_model(model_name, config, train_loader, val_loader, test_loader, dataset_path):
    """Run training and evaluation for a single model"""
    print("\n" + "="*70)
    print(f"RUNNING MODEL: {model_name}")
    print("="*70)
    
    # Create model-specific directories
    model_name_clean = model_name.replace(' ', '_').replace('/', '_').lower()
    checkpoint_dir = os.path.join('checkpoints', model_name_clean)
    results_dir = os.path.join('checkpoints', model_name_clean, 'results')
    paper_dir = os.path.join('checkpoints', model_name_clean, 'results', 'paper_deliverables')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(paper_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    try:
        # Create the model
        print(f"\n[1/7] Creating {model_name}...")
        models = create_models(config)
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found in available models")
        
        model = models[model_name]
        model = model.to(config.DEVICE)
        print(f"✓ {model_name} created and moved to {config.DEVICE}")
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        print(f"\n[2/7] Training {model_name}...")
        history = train_model(
            model, model_name, train_loader, val_loader, test_loader,
            config, criterion
        )
        print(f"✓ Training complete for {model_name}")
        
        # Evaluate with MedMNIST evaluator
        print(f"\n[3/7] Evaluating {model_name} with MedMNIST evaluator...")
        metrics = evaluate_with_medmnist(
            model, model_name, test_loader, 'test',
            config.DEVICE, dataset_path
        )
        print(f"✓ MedMNIST evaluation complete")
        
        # Get confusion matrix and classification report
        print(f"\n[4/7] Computing detailed metrics for {model_name}...")
        cm, report = evaluate_with_metrics(
            model, test_loader, config.DEVICE, TISSUEMNIST_NUM_CLASSES
        )
        print(f"✓ Detailed metrics computed")
        
        # Save results
        print(f"\n[5/7] Saving results and deliverables for {model_name}...")
        # Create a config-like object with RESULTS_DIR and other needed attributes
        results_config = type('Config', (), {
            'RESULTS_DIR': results_dir,
            'NUM_EPOCHS': config.NUM_EPOCHS,
            'BATCH_SIZE': config.BATCH_SIZE,
            'LEARNING_RATE': config.LEARNING_RATE,
            'USE_PRETRAINED': config.USE_PRETRAINED,
            'VERBOSE': config.VERBOSE
        })()
        save_single_model_results(model_name, history, metrics, cm, report, 
                                 results_config, timestamp)
        
        # Save visualizations
        save_model_visualizations(model_name, history, results_dir, timestamp, config)
        
        # Save paper deliverables
        print(f"\n[6/7] Saving paper deliverables for {model_name}...")
        # Check if paper deliverables should be saved
        save_deliverables = getattr(config, 'SAVE_PAPER_DELIVERABLES', True)
        
        if save_deliverables:
            # Create config for paper deliverables with correct paths
            paper_config = type('Config', (), {
                'RESULTS_DIR': results_dir,
                'PAPER_DIR': 'paper_deliverables',
                'NUM_EPOCHS': config.NUM_EPOCHS
            })()
            
            # Create single-model dicts for deliverables functions
            single_history = {model_name: history}
            single_metrics = {model_name: metrics}
            
            # Save performance table (single model version)
            try:
                save_performance_table(single_history, single_metrics, paper_config, timestamp)
                print(f"  ✓ Performance table saved")
            except Exception as e:
                print(f"  ⚠ Warning: Could not save performance table: {e}")
            
            # Save sample images
            try:
                save_sample_images(train_loader, paper_config, timestamp)
                print(f"  ✓ Sample images saved")
            except Exception as e:
                print(f"  ⚠ Warning: Could not save sample images: {e}")
            
            # Save publication figures (single model version)
            try:
                save_publication_figures(single_history, paper_config, timestamp)
                print(f"  ✓ Publication figures saved")
            except Exception as e:
                print(f"  ⚠ Warning: Could not save publication figures: {e}")
            
            print(f"✓ Paper deliverables saved to {paper_dir}")
        else:
            print(f"⚠ Paper deliverables disabled (SAVE_PAPER_DELIVERABLES=False)")
        
        # Save checkpoint
        print(f"\n[7/7] Saving checkpoint for {model_name}...")
        save_model_checkpoint(model, model_name, history, config, checkpoint_dir, timestamp)
        
        # Cleanup: Move model to CPU
        print(f"\nCleaning up memory for {model_name}...", end=' ', flush=True)
        if next(model.parameters()).is_cuda:
            model = model.cpu()
            if torch.cuda.is_available() and config.DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
                print("✓")
            else:
                print("✓")
        else:
            print("✓ (already on CPU)")
        
        print(f"\n{'='*70}")
        print(f"✓ {model_name} processing complete!")
        print(f"  Checkpoint directory: {os.path.abspath(checkpoint_dir)}")
        print(f"  Results directory: {os.path.abspath(results_dir)}")
        print(f"{'='*70}\n")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n\n⚠ Training interrupted by user (Ctrl+C) during {model_name}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR processing {model_name}: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def main(model_name=None):
    """Main function to run models individually
    
    Args:
        model_name: Name of the model to run. If None, runs all models in Config.MODELS_TO_TRAIN.
                   Can also be set via command line argument.
    """
    # Create config
    config = Config()
    
    # Set seed for reproducibility
    if config.RANDOM_SEED >= 0:
        set_seed(config.RANDOM_SEED)
        print(f"Random seed set to: {config.RANDOM_SEED}")
    else:
        print("Random seed: Disabled")
    
    print("="*70)
    print("TISSUEMNIST INDIVIDUAL MODEL RUNNER")
    print("="*70)
    
    # Determine which model(s) to run
    # Priority: 1) function parameter, 2) command line argument, 3) all models
    if model_name is None and len(sys.argv) > 1:
        model_name = sys.argv[1]
    
    if model_name is not None:
        if model_name not in config.MODELS_TO_TRAIN:
            print(f"✗ ERROR: Model '{model_name}' not in MODELS_TO_TRAIN")
            print(f"  Available models: {', '.join(config.MODELS_TO_TRAIN)}")
            sys.exit(1)
        models_to_run = [model_name]
        print(f"Running single model: {model_name}")
    else:
        models_to_run = config.MODELS_TO_TRAIN
        print(f"Running all models: {', '.join(models_to_run)}")
    
    # Load dataset once (shared across all models)
    print("\nLoading TissueMNIST dataset...")
    train_loader, val_loader, test_loader, dataset_path = load_tissuemnist(config)
    print(f"✓ TissueMNIST dataset loaded")
    print(f"  Dataset path: {dataset_path}")
    print(f"  Number of classes: {TISSUEMNIST_NUM_CLASSES}")
    print(f"  Train/Val/Test split: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)} samples")
    
    # Create checkpoints base directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Run each model
    successful_models = []
    failed_models = []
    
    for model_idx, model_name in enumerate(models_to_run, 1):
        print(f"\n{'#'*70}")
        print(f"MODEL {model_idx}/{len(models_to_run)}: {model_name}")
        print(f"{'#'*70}")
        
        success = run_single_model(
            model_name, config, train_loader, val_loader, test_loader, dataset_path
        )
        
        if success:
            successful_models.append(model_name)
        else:
            failed_models.append(model_name)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total models processed: {len(models_to_run)}")
    print(f"Successful: {len(successful_models)}")
    print(f"Failed: {len(failed_models)}")
    
    if successful_models:
        print(f"\n✓ Successfully processed models:")
        for model in successful_models:
            model_clean = model.replace(' ', '_').replace('/', '_').lower()
            checkpoint_dir = os.path.join('checkpoints', model_clean)
            print(f"  - {model}: {os.path.abspath(checkpoint_dir)}")
    
    if failed_models:
        print(f"\n✗ Failed models:")
        for model in failed_models:
            print(f"  - {model}")
    
    print(f"\nAll checkpoints saved under: {os.path.abspath('checkpoints')}")
    print("="*70)


if __name__ == '__main__':
    main()

