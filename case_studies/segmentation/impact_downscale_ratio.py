import os
from utils.utils import *
import numpy as np
import torch
from src.segmentation.data_loader.data_loader_segmentation import BenchmarkDataset , BenchmarkTestDataset 
from data_preprocessing.prepare_data import prepare_data
from src.segmentation.models.model_unet import UNet
from src.segmentation.models.model_deeplabv3 import DeepLabV3
from src.segmentation.training.train_segmentation import train_segmentation 
from utils.patch_extraction import PatchExtractor
from src.segmentation.training.evaluate_segmentation import test_segmentation

def set_seed(seed_value=42):
    """Set random seed for reproducibility across all libraries."""
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


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


def initialize_optimizer_and_scheduler(optimizer_name: str, scheduler_name: str, model: torch.nn.Module, hparams_model: dict) -> tuple:
    """
    Initialize both optimizer and learning rate scheduler.
    
    Args:
        optimizer_name: Name of the optimizer to use
        scheduler_name: Name of the scheduler to use
        model: The model whose parameters will be optimized
        hparams_model: Dictionary containing model hyperparameters
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    lr = float(hparams_model['Optimizer']['lr'])
    
    # Initialize optimizer
    optimizer_name = optimizer_name.lower().strip("'")
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=0.9, 
            weight_decay=float(hparams_model['Optimizer']['weight_decay'])
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Initialize scheduler
    scheduler_name = scheduler_name.lower().strip("'")
    if scheduler_name == 'reducelronplateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, min_lr=1e-8, factor=0.1, patience=10, verbose=True   
        )
    elif scheduler_name == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    elif scheduler_name == 'cosineannealinglr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=0
        )
    elif scheduler_name == 'cosineannealingwarmrestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20
        )
    elif scheduler_name == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return optimizer, scheduler



def impact_downscale_ratio():
    """
    Main function to train and evaluate sea ice segmentation models with different downscale ratios.
    
    This script performs the following key tasks:
    1. Data preparation with optional downscaling
    2. Model training
    3. Model evaluation
    
    Key features:
    1. Configurable downscaling: Set 'downsampling' to True and specify 'downsampling_factor' in config_data.ini
    2. Adaptive data paths: Automatically adjusts input/output directories based on downscale factor
    3. Flexible model architecture: Supports different models like UNet and DeepLabV3
    4. Comprehensive evaluation: Provides various performance metrics for the trained model
    
    Workflow:
    1. Load and process configuration settings from config files
    2. Prepare data, including optional downscaling (controlled by config settings)
    3. Set up data loaders for training, validation, and testing
    4. Initialize the chosen segmentation model, optimizer, and learning rate scheduler
    5. Train the model using the prepared data
    6. Evaluate the model's performance on the test set
    
    Usage:
    - To enable downscaling, set in config_data.ini:
        'downsampling': True
        'downsampling_factor': <your_chosen_factor> (e.g., 2, 5, etc.)
    - The script will automatically generate downsampled data during the prepare_data step
    - Training and evaluation will then proceed using the downsampled data
    
    Data Directory Setup:
    - If you have already generated the downsampled data, uncomment and use these lines:
        # hparams_data['dir_train_with_icecharts'] = 'out_dir_downsampled_{}_train'.format(downscale_factor)
        # hparams_data['dir_test_with_icecharts'] = 'out_dir_downsampled_{}_test'.format(downscale_factor)
    - If these lines are commented out, the prepare_data function will generate new downsampled data
      and save it in the default output directories
    
    Note:
    - Downscaling affects the input data resolution and may impact model performance
    - Different downscale factors allow for analyzing the trade-off between data resolution and model accuracy
    - Ensure sufficient computational resources when working with larger datasets or smaller downscale factors
    """
    hparams_data, hparams_model = load_configs()
    set_seed(int(hparams_model['datamodule']['seed']))
    dir_train_with_icecharts = hparams_data['dir_train_with_icecharts']


    # Data loader part
    np.random.seed(int(hparams_model['datamodule']['seed']))
    

    # preprocessing part - you can change the downscale factor in the config_data.ini file
    # hparams_data['data_preprocess_options']['downsampling'] = True
    # hparams_data['data_preprocess_options']['downsampling_factor'] = 2
    downscale_factor = hparams_data['data_preprocess_options']['downsampling_factor']

    hparams_data = prepare_data(hparams_data , hparams_model )
    print("preprocessing of train and test data is done!")
    
    hparams_data['dir_train_with_icecharts'] = 'out_dir_downsampled_{}_train'.format(downscale_factor)
    hparams_data['dir_test_with_icecharts'] = 'out_dir_downsampled_{}_test'.format(downscale_factor)
 

    ### data loader part
    val_num = int(hparams_model['datamodule']['num_val_scenes'])
    all_files_train = os.listdir(hparams_data['dir_train_with_icecharts'])
    train_list_all = [x for x in all_files_train if x.endswith(".nc")]

    validation_list = np.random.choice(train_list_all, size = val_num, replace = False)
    print("validation list: ", validation_list)
    train_list = [x for x in train_list_all if x not in validation_list]
    test_list = os.listdir(hparams_data['dir_test_with_icecharts'])
    
    # Update hparams_data
    hparams_data.update({
        'train_list': train_list,
        'validation_list': validation_list,
        'test_list': test_list,
        'meanstd_file': 'misc/global_meanstd.npy'
    })
    #patch extration method
    patch_with_stride = hparams_data['data_preprocess_options']['use_patch_with_stride']
    patch_with_randomcrop = hparams_data['data_preprocess_options']['use_patch_with_randomcrop']
    chart_label = hparams_model['model']['label'].strip("'")

    transform_list = get_transform_list(hparams_model)
    n_classes = hparams_data['n_classes'][chart_label]

    num_input_channels = len(hparams_data['sar_variables']) + len(hparams_data['amsr_env_variables']) + 1 # for month
    patch_size = int(hparams_data['data_preprocess_options']['patch_size'])



    if patch_with_randomcrop:

                train_dataset = BenchmarkDataset(hparams_data, hparams_model, hparams_data['train_list'],transform_list=transform_list)
                train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = None,  shuffle = True, num_workers = int(hparams_model['train']['num_workers']))
                val_dataset = BenchmarkTestDataset(hparams_data, hparams_model,hparams_data['validation_list'] , test = False)
                val_loader = torch.utils.data.DataLoader(val_dataset ,batch_size = None, num_workers =int(hparams_model['train']['num_workers']) )
                test_dataset = BenchmarkTestDataset(hparams_data, hparams_model,hparams_data['test_list'] , test = True)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = None ,num_workers = int(hparams_model['train']['num_workers']))

    
     # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is: ", device)
    model_name = hparams_model['model']['model'].strip("'")

    if model_name == 'Unet':
        print("Initializing U-Net model for semantic segmentation")
        num_filters = hparams_model['model']['num_filters']
        model = UNet(hparams_data, num_filters, out_channels=n_classes, in_channels=num_input_channels)
    elif model_name == 'deeplabv3':
        print("Initializing DeepLabV3 model for semantic segmentation")
        model = DeepLabV3(weights=None, num_classes=n_classes, in_channel=num_input_channels)
    else:
        raise ValueError("Invalid model specified.")

    # Training setup
    optimiser_name = hparams_model['Optimizer']['optimizer']
    scheduler_name = hparams_model['Optimizer']['scheduler']
    criterion = torch.nn.CrossEntropyLoss(ignore_index=int(hparams_data['class_fill_values'][chart_label]))
    model_path = f'checkpoints/segmentation/{model_name}_{downscale_factor}'
    optimiser, scheduler = initialize_optimizer_and_scheduler(optimiser_name, scheduler_name, model, hparams_model)

    # Train and test
    train_segmentation(hparams_data, hparams_model, model, model_name, criterion, optimiser, scheduler, train_loader, val_loader, device, load=False)
    test_f1, test_accuracy, test_precision, test_recall = test_segmentation(model, test_loader, criterion, model_name, device, hparams_data)

  
   

    