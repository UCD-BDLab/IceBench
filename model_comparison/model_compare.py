import os
import sys
import argparse
import importlib.util
import torch
import configparser
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import time
from datetime import datetime
import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Import utility functions
from src.utils.utils import *

# Conditional imports for different tasks
try:
    from src.classification.data_loader.data_loader_classification import BenchmarkDataset_directory, BenchmarkTestDataset_directory
    from src.segmentation.data_loader.data_loader_segmentation import BenchmarkDataset, BenchmarkTestDataset
except ImportError as e:
    print(f"Warning: Could not import data loaders: {e}")


def load_user_model(model_file_path: str, model_class_name: str) -> torch.nn.Module:
    """
    Dynamically load a user-defined model from a Python file.
    
    Args:
        model_file_path: Path to the Python file containing the model definition
        model_class_name: Name of the model class to instantiate
        
    Returns:
        An instance of the model class
    """
    try:
        # Load the module spec
        spec = importlib.util.spec_from_file_location("user_model", model_file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec from {model_file_path}")
        
        # Create the module
        user_model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_model_module)
        
        # Get the model class
        model_class = getattr(user_model_module, model_class_name)
        return model_class
    except Exception as e:
        print(f"Error loading user model: {e}")
        raise


def load_configs() -> Tuple[Dict, Dict]:
    """
    Load data and model configurations from config files.
    
    Returns:
        Tuple of (data_config, model_config)
    """
    config_data_path = "configs/config_data.ini"
    config_model_path = "configs/config_model.ini"
    
    if not os.path.exists(config_data_path):
        raise FileNotFoundError(f"Data config file not found at {config_data_path}")
        
    if not os.path.exists(config_model_path):
        raise FileNotFoundError(f"Model config file not found at {config_model_path}")
    
    # Load data config
    config_data = read_config_file(config_data_path)
    hparams_data = process_settings_data(config_data)
    
    # Load model config
    hparams_model = read_config_model(config_model_path)
    
    return hparams_data, hparams_model


def setup_dataloaders(task_type: str, data_config: Dict, model_config: Dict) -> Tuple:
    """
    Set up the appropriate dataloaders based on the task type.
    
    Args:
        task_type: Either 'classification' or 'segmentation'
        data_config: Data configuration dictionary
        model_config: Model configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if task_type == "classification":
        # Setup for classification
        print("Setting up classification dataloaders...")
        # 
        train_dataset = BenchmarkDataset_directory(data_config, model_config, data_config['dir_train_with_icecharts'])
        val_dataset = BenchmarkDataset_directory(data_config, model_config, data_config['dir_val_with_icecharts'])
        test_dataset = BenchmarkTestDataset_directory(data_config, model_config, data_config['dir_test_with_icecharts'])
    
    elif task_type == "segmentation":
        # Setup for segmentation
        print("Setting up segmentation dataloaders...")
        train_files = [f for f in os.listdir(data_config['dir_train_with_icecharts']) 
                      if f.endswith('.nc') and f in data_config['train_list']]
        
        # Get all training files and randomly select validation files
        all_train_files = [f for f in os.listdir(data_config['dir_train_with_icecharts']) 
                      if f.endswith('.nc')]
        
        # Fix the random seed for reproducibility
        random.seed(int(model_config['datamodule']['seed']))
        
        # Randomly select validation files
        num_val_scenes = int(model_config['datamodule']['num_val_scenes'])
        val_files = random.sample(all_train_files, num_val_scenes)
        
        # Remove validation files from training files
        train_files = [f for f in all_train_files if f not in val_files]
        
        test_files = [f for f in os.listdir(data_config['dir_test_with_icecharts']) 
                     if f.endswith('.nc') and f in data_config['test_list']]
        
        train_dataset = BenchmarkDataset(data_config, model_config, train_files)
        val_dataset = BenchmarkDataset(data_config, model_config, val_files)
        test_dataset = BenchmarkTestDataset(data_config, model_config, test_files, test=True)
    
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Create DataLoaders
    batch_size = int(model_config['train']['batch_size'])
    num_workers = int(model_config['train']['num_workers'])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


class LightningModelWrapper(pl.LightningModule):
    """
    A PyTorch Lightning wrapper for user-defined models.
    """
    
    def __init__(self, model: torch.nn.Module, task_type: str, hparams_data: Dict, hparams_model: Dict):
        """
        Initialize the Lightning wrapper around a user-defined model.
        
        Args:
            model: The user-defined model
            task_type: Either 'classification' or 'segmentation'
            hparams_data: Data configuration
            hparams_model: Model configuration
        """
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.hparams_data = hparams_data
        self.hparams_model = hparams_model
        
        # Store additional configuration as hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
        # Define loss function based on task type
        if self.task_type == "classification":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:  # segmentation
            ignore_index = self.hparams_data['class_fill_values'].get(
                self.hparams_model['model']['label'].strip("'"), 255)
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Batch of training data
            batch_idx: Index of the batch
            
        Returns:
            Loss value
        """
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate accuracy for classification
        if self.task_type == "classification":
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y).sum().item()
            accuracy = correct / y.size(0)
            self.log('train_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Batch of validation data
            batch_idx: Index of the batch
            
        Returns:
            Dictionary of validation metrics
        """
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        # Calculate additional metrics for classification
        if self.task_type == "classification":
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y).sum().item()
            accuracy = correct / y.size(0)
            self.log('val_acc', accuracy, on_epoch=True, prog_bar=True)
            
            return {'val_loss': loss, 'val_acc': accuracy}
        
        return {'val_loss': loss}
    
    def test_step(self, batch, batch_idx):
        """
        Test step.
        
        Args:
            batch: Batch of test data
            batch_idx: Index of the batch
            
        Returns:
            Dictionary of test metrics
        """
        if self.task_type == "classification":
            x, y = batch
            outputs = self(x)
            loss = self.criterion(outputs, y)
            
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y).sum().item()
            accuracy = correct / y.size(0)
            
            self.log('test_loss', loss, on_epoch=True)
            self.log('test_acc', accuracy, on_epoch=True)
            
            return {
                'test_loss': loss,
                'test_acc': accuracy,
                'y_true': y,
                'y_pred': predicted
            }
        else:  # segmentation
            x, y, masks, name = batch
            outputs = self(x)
            
            # Loss calculation (excluding masked areas)
            loss = self.criterion(outputs, y)
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=1)
            
            self.log('test_loss', loss, on_epoch=True)
            
            return {
                'test_loss': loss,
                'y_true': y,
                'y_pred': predictions,
                'masks': masks,
                'name': name
            }
    
    def configure_optimizers(self):
        """
        Configure optimizers for training.
        
        Returns:
            Optimizer and learning rate scheduler
        """
        optimizer_name = self.hparams_model['optimizer']['optimizer'].lower()
        lr = float(self.hparams_model['optimizer']['lr'])
        weight_decay = float(self.hparams_model['optimizer']['weight_decay'])
        
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        scheduler_name = self.hparams_model['optimizer']['scheduler_name']
        
        if scheduler_name == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'monitor': 'val_loss'
                }
            }
        elif scheduler_name == 'ReduceLROnPlateau':
            patience = int(self.hparams_model['optimizer']['reduce_lr_patience'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=patience, verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'monitor': 'val_loss'
                }
            }
        else:
            return optimizer
    
    def test_epoch_end(self, outputs):
        """
        Process outputs at the end of test epoch.
        
        Args:
            outputs: List of outputs from test_step
            
        Returns:
            Dictionary of aggregated test metrics
        """
        if self.task_type == "classification":
            # Concatenate all predictions and targets
            all_preds = torch.cat([x['y_pred'] for x in outputs])
            all_targets = torch.cat([x['y_true'] for x in outputs])
            
            # Move to CPU for sklearn metrics
            all_preds = all_preds.cpu().numpy()
            all_targets = all_targets.cpu().numpy()
            
            # Import metrics from scikit-learn
            from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
            
            # Calculate metrics
            accuracy = accuracy_score(all_targets, all_preds)
            precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
            conf_matrix = confusion_matrix(all_targets, all_preds)
            
            # Log metrics
            self.log('test_accuracy', accuracy)
            self.log('test_precision', precision)
            self.log('test_recall', recall)
            self.log('test_f1', f1)
            
            # Return results dictionary
            return {
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'confusion_matrix': conf_matrix
            }
        else:  # segmentation
            # Concatenate predictions and targets (excluding masked areas)
            all_preds = []
            all_targets = []
            
            for output in outputs:
                predictions = output['y_pred'].cpu()
                targets = output['y_true'].cpu()
                masks = output['masks']
                
                # Exclude masked areas
                for i in range(len(predictions)):
                    pred = predictions[i].numpy()
                    target = targets[i].numpy()
                    mask = masks[i].numpy()
                    
                    pred_flat = pred[~mask]
                    target_flat = target[~mask]
                    
                    all_preds.extend(pred_flat)
                    all_targets.extend(target_flat)
            
            # Import metrics from scikit-learn
            from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, jaccard_score
            
            # Calculate metrics
            f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=1)
            accuracy = accuracy_score(all_targets, all_preds)
            precision = precision_score(all_targets, all_preds, average='weighted', zero_division=1)
            recall = recall_score(all_targets, all_preds, average='weighted', zero_division=1)
            jaccard = jaccard_score(all_targets, all_preds, average='weighted', zero_division=1)
            
            # Log metrics
            self.log('test_f1', f1)
            self.log('test_accuracy', accuracy)
            self.log('test_precision', precision)
            self.log('test_recall', recall)
            self.log('test_jaccard', jaccard)
            
            # Return results dictionary
            return {
                'test_f1': f1,
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_jaccard': jaccard
            }


def train_and_evaluate(
    model_class, 
    task_type: str, 
    hparams_data: Dict, 
    hparams_model: Dict,
    model_params: Dict = None,
    output_dir: str = "model_results",
    mode: str = "both"
) -> Dict:
    """
    Train and evaluate a model using PyTorch Lightning.
    
    Args:
        model_class: The class of the model to train
        task_type: Either 'classification' or 'segmentation'
        hparams_data: Data configuration
        hparams_model: Model configuration
        model_params: Parameters to pass to model constructor
        output_dir: Directory to save output files
        mode: Mode of operation ('train', 'evaluate', or 'both')
        
    Returns:
        Dictionary containing evaluation results
    """
    # Initialize model
    if model_params is None:
        model_params = {}
    
    model = model_class(**model_params)
    print(f"Model architecture:\n{model}")
    
    # Create Lightning model
    lightning_model = LightningModelWrapper(model, task_type, hparams_data, hparams_model)
    
    # Set up data loaders
    train_loader, val_loader, test_loader = setup_dataloaders(task_type, hparams_data, hparams_model)
    
    # Set up trainer
    epochs = int(hparams_model['train']['epochs'])
    patience = int(hparams_model['train']['patience'])
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=f"{model.__class__.__name__}_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        verbose=True,
        mode="min"
    )
    
    # Set up logger
    logger = TensorBoardLogger(save_dir=output_dir, name="lightning_logs")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
        deterministic=True
    )
    
    # Training mode
    if mode in ["train", "both"]:
        print("Starting training...")
        trainer.fit(lightning_model, train_loader, val_loader)
        
        # Plot training curves from TensorBoard logs
        try:
            from tensorboard.backend.event_processing import event_accumulator
            
            # Get the latest event file
            event_file = sorted(Path(logger.log_dir).glob('events.out.tfevents.*'))[-1]
            ea = event_accumulator.EventAccumulator(str(event_file))
            ea.Reload()
            
            # Extract scalars
            train_loss = [(s.step, s.value) for s in ea.Scalars('train_loss_epoch')]
            val_loss = [(s.step, s.value) for s in ea.Scalars('val_loss')]
            
            if task_type == "classification":
                train_acc = [(s.step, s.value) for s in ea.Scalars('train_acc_epoch')]
                val_acc = [(s.step, s.value) for s in ea.Scalars('val_acc')]
            
            # Plot curves
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot([x[0] for x in train_loss], [x[1] for x in train_loss], label='Train Loss')
            plt.plot([x[0] for x in val_loss], [x[1] for x in val_loss], label='Validation Loss')
            plt.title('Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            if task_type == "classification":
                plt.subplot(1, 2, 2)
                plt.plot([x[0] for x in train_acc], [x[1] for x in train_acc], label='Train Accuracy')
                plt.plot([x[0] for x in val_acc], [x[1] for x in val_acc], label='Validation Accuracy')
                plt.title('Accuracy Curves')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'training_curves.png'))
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate training curves: {e}")
    
    # Evaluation mode
    results = {}
    if mode in ["evaluate", "both"]:
        print("Starting evaluation...")
        test_results = trainer.test(lightning_model, test_loader, ckpt_path=checkpoint_callback.best_model_path)
        
        # Extract and print results
        results = test_results[0]
        print("\nEvaluation Results:")
        for metric, value in results.items():
            if not isinstance(value, np.ndarray):  # Skip confusion matrix for printing
                print(f"{metric}: {value:.4f}")
        
        # Save results
        np.save(os.path.join(output_dir, "evaluation_results.npy"), results)
    
    return results


def evaluate_existing_model(
    model_class, 
    model_checkpoint: str,
    task_type: str, 
    hparams_data: Dict, 
    hparams_model: Dict,
    model_params: Dict = None,
    test_loader = None
) -> Dict:
    """
    Evaluate an existing model from a checkpoint.
    
    Args:
        model_class: The class of the model to evaluate
        model_checkpoint: Path to the model checkpoint
        task_type: Either 'classification' or 'segmentation'
        hparams_data: Data configuration
        hparams_model: Model configuration
        model_params: Parameters to pass to model constructor
        test_loader: Test data loader (optional)
        
    Returns:
        Dictionary containing evaluation results
    """
    # Initialize model
    if model_params is None:
        model_params = {}
    
    model = model_class(**model_params)
    
    # Load weights from checkpoint
    checkpoint = torch.load(model_checkpoint)
    # If it's a Lightning checkpoint, extract the model state dict
    if 'state_dict' in checkpoint:
        # Get the model state dict by removing the 'model.' prefix
        model_state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()
                            if k.startswith('model.')}
        model.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create Lightning model
    lightning_model = LightningModelWrapper(model, task_type, hparams_data, hparams_model)
    
    # Set up test loader if not provided
    if test_loader is None:
        _, _, test_loader = setup_dataloaders(task_type, hparams_data, hparams_model)
    
    # Create trainer for evaluation only
    trainer = pl.Trainer(logger=False)
    
    # Evaluate model
    test_results = trainer.test(lightning_model, test_loader)
    
    return test_results[0]


def compare_models(
    user_model_results: Dict,
    compare_models: List[str],
    model_class,
    task_type: str,
    hparams_data: Dict,
    hparams_model: Dict,
    model_params: Dict = None,
    output_dir: str = "model_results"
) -> None:
    """
    Compare the user's model with other models.
    
    Args:
        user_model_results: Dictionary of user model evaluation results
        compare_models: List of paths to model checkpoints to compare with
        model_class: The class of the model
        task_type: Either 'classification' or 'segmentation'
        hparams_data: Data configuration
        hparams_model: Model configuration
        model_params: Parameters to pass to model constructor
        output_dir: Directory to save output files
    """
    if not compare_models:
        return
    
    # Setup test loader
    _, _, test_loader = setup_dataloaders(task_type, hparams_data, hparams_model)
    
    # Collect results for all models
    all_results = {'user_model': user_model_results}
    
    for model_path in compare_models:
        model_name = os.path.basename(model_path).split('_best')[0]
        print(f"Evaluating comparison model: {model_name}")
        
        results = evaluate_existing_model(
            model_class,
            model_path,
            task_type,
            hparams_data,
            hparams_model,
            model_params,
            test_loader
        )
        
        all_results[model_name] = results
    
    # Plot comparison
    plot_comparison(all_results, output_dir)


def plot_comparison(all_results: Dict[str, Dict], output_dir: str) -> None:
    """
    Plot comparison metrics for different models.
    
    Args:
        all_results: Dictionary mapping model names to their evaluation results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model names and common metrics
    model_names = list(all_results.keys())
    
    # Get the first model to determine available metrics
    first_model = next(iter(all_results.values()))
    metrics = [metric for metric in first_model.keys() 
              if not isinstance(first_model[metric], np.ndarray) and metric.startswith('test_')]
    
    # Create a bar chart for each metric
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        values = [all_results[model].get(metric, 0) for model in model_names]
        
        bars = plt.bar(model_names, values)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{value:.4f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        metric_name = metric.replace('test_', '').replace('_', ' ').title()
        plt.title(f'Comparison of {metric_name} Across Models')
        plt.ylabel(metric_name)
        plt.ylim(0, 1.1)  # Most metrics range from 0 to 1
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'))
        plt.close()
    
    # Create a comprehensive comparison table as CSV
    import csv
    with open(os.path.join(output_dir, 'model_comparison.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['Model'] + [metric.replace('test_', '').replace('_', ' ').title() for metric in metrics])
        
        # Write data for each model
        for model_name in model_names:
            row = [model_name]
            for metric in metrics:
                row.append(f"{all_results[model_name].get(metric, 0):.4f}")
            writer.writerow(row)


def save_configs(hparams_data: Dict, hparams_model: Dict, output_dir: str) -> None:
    """
    Save configuration files to output directory.
    
    Args:
        hparams_data: Data configuration
        hparams_model: Model configuration
        output_dir: Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data config
    data_config_path = os.path.join(output_dir, "data_config.ini")
    with open(data_config_path, 'w') as f:
        for section, params in hparams_data.items():
            if isinstance(params, dict):
                f.write(f"[{section}]\n")
                for key, value in params.items():
                    f.write(f"{key} = {value}\n")
                f.write("\n")
    
    # Save model config
    model_config_path = os.path.join(output_dir, "model_config.ini")
    with open(model_config_path, 'w') as f:
        for section, params in hparams_model.items():
            if isinstance(params, dict):
                f.write(f"[{section}]\n")
                for key, value in params.items():
                    f.write(f"{key} = {value}\n")
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate custom models")
    
    # Required arguments
    parser.add_argument("--model_file", type=str, required=True,
                        help="Path to Python file containing model definition")
    parser.add_argument("--model_class", type=str, required=True,
                        help="Name of the model class in the file")
    
    # Task type
    parser.add_argument("--task_type", type=str, choices=["classification", "segmentation"],
                        help="Type of task (if not specified, read from config_model.ini)")
    
    # Mode
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "both"], default="both",
                        help="Mode: train, evaluate, or both")
    
    # Optional arguments for model instantiation
    parser.add_argument("--model_params", type=str, default=None,
                        help="JSON string of parameters to pass to model constructor")
    
    # Output directories
    parser.add_argument("--output_dir", type=str, default="model_results",
                        help="Directory to store results and saved models")
    parser.add_argument("--compare_with", type=str, default=None, nargs="+",
                        help="Paths to saved model checkpoints to compare with")
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model_class}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configurations
    config_data_path = "configs/config_data.ini"
    config_model_path = "configs/config_model.ini"
    
    # Load data config
    config_data = read_config_file(config_data_path)
    hparams_data = process_settings_data(config_data)
    
    # Load model config
    hparams_model = read_config_model(config_model_path)
    
    # Save copies of configs to output directory
    save_configs(hparams_data, hparams_model, output_dir)
    
    # Determine task type
    task_type = args.task_type
    if task_type is None:
        task_type = hparams_model['model']['mode'].strip("'").lower()
        print(f"Task type not specified, using {task_type} from config")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model class
    model_class = load_user_model(args.model_file, args.model_class)
    
    # Parse model parameters if provided
    model_params = {}
    if args.model_params:
        model_params = json.loads(args.model_params)
    
    # Instantiate the model
    model = model_class(**model_params)
    print(f"Model architecture:\n{model}")
    
    # Setup dataloaders
    train_loader, val_loader, test_loader = setup_dataloaders(
        task_type, hparams_data, hparams_model
    )
    
    # Define paths for saving
    model_path = os.path.join(output_dir, f"{args.model_class}_best.pt")
    
    # Define loss function based on task type
    if task_type == "classification":
        criterion = torch.nn.CrossEntropyLoss()
    else:  # segmentation
        ignore_index = hparams_data['class_fill_values'].get(
            hparams_model['model']['label'].strip("'"), 255)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    # Training mode
    results = {}
    if args.mode in ["train", "both"]:
        print("Starting training...")
        
        # Define optimizer
        lr = float(hparams_model['optimizer']['lr'])
        weight_decay = float(hparams_model['optimizer']['weight_decay'])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Define scheduler
        patience = int(hparams_model['optimizer']['reduce_lr_patience'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience, verbose=True
        )
        
        # Define early stopping parameters
        early_stopping_patience = int(hparams_model['train']['patience'])
        max_epochs = int(hparams_model['train']['epochs'])
        
        # Train the model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_epochs=max_epochs,
            model_save_path=model_path,
            early_stopping_patience=early_stopping_patience,
            scheduler=scheduler
        )
        
        # Save training history
        np.save(os.path.join(output_dir, "training_history.npy"), history)
        
        # Plot training curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'))
        plt.close()
    
    # Evaluation mode
    if args.mode in ["evaluate", "both"]:
        print("Starting evaluation...")
        
        # If we're only evaluating (not training), load the best model
        if args.mode == "evaluate" and os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        elif args.mode == "evaluate" and not os.path.exists(model_path):
            print(f"No model found at {model_path}. Please train the model first or provide a valid model path.")
            return
        
        model.to(device)
        model.eval()
        
        # Evaluate based on task type
        if task_type == "classification":
            results = evaluate_classification_model(model, test_loader, device)
        else:  # segmentation
            results = evaluate_segmentation_model(model, test_loader, device, hparams_data)
        
        # Print and save evaluation results
        print("\nEvaluation Results:")
        for metric, value in results.items():
            if not isinstance(value, np.ndarray):  # Skip confusion matrix for printing
                print(f"{metric}: {value:.4f}")
        
        # Save results
        np.save(os.path.join(output_dir, "evaluation_results.npy"), results)
        
        # Compare with other models if specified
        if args.compare_with:
            all_results = {args.model_class: results}
            
            for compare_path in args.compare_with:
                compare_name = os.path.basename(compare_path).split('_best.pt')[0]
                
                # Load the model for comparison
                compare_model = model_class(**model_params)  # Create a new instance
                checkpoint = torch.load(compare_path)
                compare_model.load_state_dict(checkpoint['model_state_dict'])
                
                # Evaluate the comparison model
                if task_type == "classification":
                    compare_results = evaluate_classification_model(compare_model, test_loader, device)
                else:  # segmentation
                    compare_results = evaluate_segmentation_model(compare_model, test_loader, device, hparams_data)
                
                all_results[compare_name] = compare_results
            
            # Plot comparison metrics
            plot_metrics(all_results, output_dir)
    
    print(f"All results saved to {output_dir}")


if __name__ == "__main__":
    main()