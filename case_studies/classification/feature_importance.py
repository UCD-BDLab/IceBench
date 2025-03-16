import os
from utils import *
import numpy as np
import torch
from torchvision.models.resnet import ResNet50_Weights, ResNet101_Weights , ResNet152_Weights
from src.classification.data_loader.data_loader_classification import *
from src.classification.models.model_cnn import CNN
from src.classification.models.model_densenet import ModifiedDenseNet121
from src.classification.models.model_resnet import CustomResNet50, CustomResNet101, CustomResNet152
from src.classification.models.model_alexnet import AlexNet , ModifiedAlexNet
from src.classification.models.model_VGG16 import CustomVGG16
from src.classification.data_loader.Data_module import *
from src.classification.training_evaluate.Model_module import *
from src.classification.models.model_resnet import *
from src.data_preprocessing.prepare_data import prepare_data
from src.utils.utils import *
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from captum.attr import GradientShap, DeepLiftShap , IntegratedGradients, FeatureAblation, FeatureAblation, GradientShap



def set_seed(seed_value=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    pl.seed_everything(seed_value)

import os
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchvision.models.resnet import ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from captum.attr import IntegratedGradients, FeatureAblation, GradientShap, DeepLiftShap

# Import local modules
from utils import read_config_file, process_settings_data, read_config_model, get_variable_options
from src.classification.data_loader.data_loader_classification import get_transform_list
from src.classification.models.model_cnn import CNN
from src.classification.models.model_densenet import ModifiedDenseNet121
from src.classification.models.model_resnet import CustomResNet50, CustomResNet101, CustomResNet152
from src.classification.models.model_alexnet import ModifiedAlexNet
from src.classification.models.model_VGG16 import CustomVGG16
from src.classification.data_loader.Data_module import MyDataModule
from src.classification.training_evaluate.Model_module import MyModel


def set_seed(seed_value=42):
    """Set random seed for reproducibility across all libraries."""
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    pl.seed_everything(seed_value)


def load_configs():
    """Load and process configuration files."""
    config_data_path = "config_data.ini"
    config_data = read_config_file(config_data_path)
    hparams_data = process_settings_data(config_data)

    config_model_path = "config_model.ini"
    hparams_model = read_config_model(config_model_path)
    
    # Load mean and std for normalization
    hparams_data['meanstd_file'] = 'global_mean_std.npy'
    hparams_data['mean_std_dict'] = np.load(hparams_data['meanstd_file'], allow_pickle=True).item()
    
    return hparams_data, hparams_model


def preprocess_data(hparams_data, hparams_model):
    """Preprocess training and testing data."""
    if len(hparams_data['amsr_env_variables']) > 0:
        print("Upsampling AMSR environmental variables...")
        hparams_data = get_variable_options(hparams_data, hparams_model)
    
    print("Preprocessing of train and test data is complete")
    
    # Set data directories
    hparams_data['dir_samples_labels_train'] = hparams_data['dir_train_with_icecharts'] 
    hparams_data['dir_samples_labels_val'] = hparams_data['dir_val_with_icecharts']
    hparams_data['dir_samples_labels_test'] = hparams_data['dir_test_with_icecharts']
    
    return hparams_data


def initialize_model(hparams_data, hparams_model):
    """Initialize model, optimizer, and scheduler based on configuration."""
    # Calculate input channels and classes
    chart_label = hparams_model['model']['label'].strip("'")
    n_classes = hparams_data['n_classes'][chart_label] - 1  # -1 to exclude background class
    num_input_channels = len(hparams_data['sar_variables']) + len(hparams_data['amsr_env_variables']) + 1  # +1 for month
    
    print(f"Number of input channels: {num_input_channels}")
    print(f"Number of classes: {n_classes}")
    
    # Define available models
    models = {
        'CNN': CNN(input_channels=num_input_channels, n_outputs=n_classes),
        'densenet121': ModifiedDenseNet121(pretrained=True, input_channels=num_input_channels, num_classes=n_classes),
        'resnet50': CustomResNet50(in_channels=num_input_channels, num_classes=n_classes, weights=ResNet50_Weights.DEFAULT),
        'resnet101': CustomResNet101(in_channels=num_input_channels, num_classes=n_classes, weights=ResNet101_Weights.DEFAULT),
        'resnet152': CustomResNet152(in_channels=num_input_channels, num_classes=n_classes, weights=ResNet152_Weights.DEFAULT),
        'alexnet': ModifiedAlexNet(in_channels=num_input_channels, num_classes=n_classes, pretrained=True),
        'VGG16': CustomVGG16(in_channels=num_input_channels, num_classes=n_classes, pretrained=True)
    }
    
    # Select model from config
    model_name = hparams_model['model']['model'].strip("'")
    if model_name not in models:
        raise ValueError(f"Invalid model specified: {model_name}")
    
    model = models[model_name]
    
    # Define optimizers
    learning_rate = float(hparams_model['Optimizer']['lr'])
    optimizers = {
        'Adam': torch.optim.Adam(model.parameters(), lr=learning_rate),
        'SGD': torch.optim.SGD(model.parameters(), lr=learning_rate),
    }
    
    # Select optimizer from config
    optimizer_name = hparams_model['Optimizer']['optimizer'].strip("'")
    if optimizer_name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    optimizer = optimizers[optimizer_name]
    
    # Define schedulers
    schedulers = {
        'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        ),
        'StepLR': torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        ),
        'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10
        ),
    }
    
    # Select scheduler from config
    scheduler_name = hparams_model['Optimizer']['scheduler_name'].strip("'")
    if scheduler_name not in schedulers:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    scheduler = schedulers[scheduler_name]
    criterion = torch.nn.CrossEntropyLoss()
    
    return model, criterion, optimizer, scheduler


def load_trained_model(model, criterion, optimizer, scheduler, hparams_model):
    """Load a trained model from checkpoint."""
    model_path = hparams_model['model']['checkpoint_path']
    
    model_pl = MyModel.load_from_checkpoint(
        model_path, 
        model=model, 
        criterion=criterion, 
        optimiser=optimizer, 
        scheduler=scheduler
    )
    
    model_pl.eval()
    return model_pl


def aggregate_attributions(attributions):
    """Aggregate attributions across spatial dimensions and batch."""
    return attributions.sum(dim=(2, 3)).mean(dim=0).cpu().detach().numpy()


def normalize_scores(scores):
    """Normalize attribution scores using L1 norm."""
    norm = np.linalg.norm(scores, ord=1)
    return scores / norm if norm != 0 else scores


def compute_feature_importance(model, test_loader, feature_names, device):
    """Compute feature importance using multiple attribution methods."""
    # Class names for plot titles
    class_names = {
        0: 'Open water',
        1: 'New Ice',
        2: 'Young ice',
        3: 'Thin FYI',
        4: 'Thick FYI',
        5: 'Old ice'
    }
    
    num_classes = len(class_names)
    
    # Wrapper function for attribution methods
    def model_forward(inputs):
        return model(inputs)
        
    # Initialize attribution methods
    integrated_gradients = IntegratedGradients(model_forward)
    feature_ablation = FeatureAblation(model_forward)
    gradient_shap = GradientShap(model_forward)
    deeplift_shap = DeepLiftShap(model)
    
    # Initialize score accumulators for each class and method
    accumulated_scores = {
        target: {
            'IG': np.zeros(len(feature_names)),
            'FA': np.zeros(len(feature_names)),
            'GS': np.zeros(len(feature_names)),
            'DL': np.zeros(len(feature_names))
        } for target in range(num_classes)
    }
    
    # Process all batches in the test loader
    num_batches = len(test_loader)
    for batch_idx, (inputs, _) in enumerate(test_loader):
        print(f"Processing batch {batch_idx+1}/{num_batches}")
        inputs = inputs.to(device)
        
        # Process each class
        for target in range(num_classes):
            # Create baselines for SHAP methods
            baseline_zeros = torch.zeros_like(inputs)
            baseline_ones = torch.ones_like(inputs)
            baselines = torch.cat([baseline_zeros, baseline_ones], dim=0)
            
            # 1. Integrated Gradients
            ig_attr = integrated_gradients.attribute(inputs, target=target)
            ig_attr_aggregated = aggregate_attributions(ig_attr)
            normalized_ig_scores = normalize_scores(ig_attr_aggregated)
            accumulated_scores[target]['IG'] += normalized_ig_scores
            
            # 2. Feature Ablation
            feature_mask = torch.arange(inputs.shape[1]).view(1, inputs.shape[1], 1, 1).to(device)
            ablation_attr = feature_ablation.attribute(inputs, target=target, feature_mask=feature_mask)
            ablation_aggregated = aggregate_attributions(ablation_attr)
            normalized_ablation_scores = normalize_scores(ablation_aggregated)
            accumulated_scores[target]['FA'] += normalized_ablation_scores
            
            # 3. Gradient SHAP
            torch.cuda.empty_cache()  # Clear GPU memory
            gs_attr = gradient_shap.attribute(inputs, baselines=baselines, target=target)
            gs_aggregated = aggregate_attributions(gs_attr)
            normalized_gs_scores = normalize_scores(gs_aggregated)
            accumulated_scores[target]['GS'] += normalized_gs_scores
            torch.cuda.empty_cache()
            
            # 4. DeepLift SHAP
            dl_attr = deeplift_shap.attribute(inputs, baselines=baselines, target=target)
            dl_aggregated = aggregate_attributions(dl_attr)
            normalized_dl_scores = normalize_scores(dl_aggregated)
            accumulated_scores[target]['DL'] += normalized_dl_scores
    
    # Average scores across all batches
    for target in range(num_classes):
        for method in accumulated_scores[target]:
            accumulated_scores[target][method] /= num_batches
            
    return accumulated_scores, class_names


def plot_feature_importance(accumulated_scores, feature_names, class_names, output_dir="plots_captum"):
    """Create visualization of feature importance for all classes and methods."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define plotting parameters
    num_classes = len(class_names)
    methods = {
        'IG': {'label': 'Integrated Gradient', 'color': '#1f77b4'},
        'FA': {'label': 'Feature Ablation', 'color': '#ff7f0e'},
        'GS': {'label': 'Gradient SHAP', 'color': '#2ca02c'},
        'DL': {'label': 'DeepLift SHAP', 'color': '#d62728'}
    }
    
    # Initialize figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12), sharex=True)
    axs = axs.flatten()
    
    # Plot each class
    for target in range(num_classes):
        ax = axs[target]
        width = 0.2  # Width of bars
        indices = np.arange(len(feature_names))
        
        # Plot each attribution method
        for i, (method, props) in enumerate(methods.items()):
            offset = (i - 1.5) * width
            ax.barh(
                indices + offset, 
                accumulated_scores[target][method], 
                width, 
                label=props['label'], 
                color=props['color']
            )
        
        # Customize plot appearance
        ax.set_yticks(indices)
        ax.set_yticklabels(feature_names, fontsize=12)
        ax.set_xlabel('Attribution Score', fontsize=16)
        ax.set_title(f'{class_names[target]}', fontsize=20)
        ax.invert_yaxis()  # Most important features on top
        ax.set_xlim(-1, 1)
        ax.tick_params(axis='x', labelsize=14)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add a single legend for the entire figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=16)
    
    # Adjust layout and save figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    model_name = "feature_importance_analysis"
    plt.savefig(f'{output_dir}/{model_name}_all_targets_4attr.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_dir}/{model_name}_all_targets_4attr.png")
    
    plt.show()


def main():
    """Main function to run the feature importance analysis pipeline."""
    print("Starting feature importance analysis...")
    
    # Set random seed for reproducibility
    seed_value = 42
    set_seed(seed_value)
    
    # Load configurations
    hparams_data, hparams_model = load_configs()
    
    # Preprocess data
    hparams_data = preprocess_data(hparams_data, hparams_model)
    
    # Print dataset statistics
    train_files = os.listdir(hparams_data['dir_samples_labels_train'])
    val_files = os.listdir(hparams_data['dir_samples_labels_val'])
    test_files = os.listdir(hparams_data['dir_samples_labels_test'])
    
    print(f"Number of samples - Training: {len(train_files)/2}, Validation: {len(val_files)/2}, Testing: {len(test_files)/2}")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize and load model
    transform_list = get_transform_list(hparams_model)
    model, criterion, optimizer, scheduler = initialize_model(hparams_data, hparams_model)
    model = load_trained_model(model, criterion, optimizer, scheduler, hparams_model)
    
    # Initialize data module and get test loader
    data_module = MyDataModule(hparams_data, hparams_model, transform_list)
    data_module.setup()
    test_loader = data_module.test_dataloader()
    
    # Define feature names
    feature_names = [
        'HH', 'HV', 'Incidence Angle', 'Distance map',
        'Latitude', 'Longitude',
        'AMSR2 18.7 H', 'AMSR2 18.7 V', 'AMSR2 36.5 H', 'AMSR2 36.5 V', 
        'Eastward 10m wind', 'Northward 10m wind', '2m air temperature',
        'Total column water vapor', 'Total column cloud liquid water', 'Month'
    ]
    
    # Compute feature importance
    accumulated_scores, class_names = compute_feature_importance(
        model, test_loader, feature_names, device
    )
    
    # Plot feature importance
    plot_feature_importance(accumulated_scores, feature_names, class_names)
    
    print("Feature importance analysis complete.")

