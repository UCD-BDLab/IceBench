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


def impact_data_prep ():

    """
    Assess the impact of various data preparation techniques on sea ice segmentation performance.

    This function is designed to evaluate how different data preprocessing steps affect 
    the accuracy of sea ice segmentation models. It allows for experimentation with:

    1. Distance to border calculation: Evaluates the effect of including distance to polygon borders.
    2. Data augmentation: Compares performance with and without data augmentation.
    3. Land masking: Assesses the impact of including or excluding land areas in the data.

    These preprocessing options can be configured in the 'config_data.ini' file under 
    the 'data_preprocess_options' section. By adjusting these parameters, users can:
    - Enable/disable distance to border calculation and set thresholds, use distance_to_border' and 'distance_border_threshold' for set up this variables
    - Toggle data augmentation on/off in this function
    - Choose to include or exclude land areas, use land_masking variable in the Data_preprocess_option

    The function performs the following steps:
    1. Loads configuration settings for data and model parameters.
    2. Prepares the dataset based on the specified preprocessing options.
    3. Splits data into training, validation
    4. Sets up data loaders and initializes the segmentation model.
    5. Trains the model and evaluates its performance.

    By running this function with different preprocessing configurations, users can 
    analyze how each technique impacts the sea ice segmentation results.

    Note: The function doesn't return values but saves the trained model and prints 
    evaluation metrics, allowing for comparison between different preprocessing strategies.
    """

    # Load configs first
    hparams_data, hparams_model = load_configs()
    
    # Log data preparation configuration
    print("\n=== Data Preparation Configuration ===")
    print(f"Land masking: {hparams_data['data_preprocess_options'].get('land_masking', False)}")
    print(f"Distance to border: {hparams_data['data_preprocess_options'].get('distance_to_border', False)}")
    print(f"Distance border threshold: {hparams_data['data_preprocess_options'].get('distance_border_threshold', 'Not set')}")

    print("=====================================\n")

    set_seed(int(hparams_model['datamodule']['seed']))
    dir_train_with_icecharts = hparams_data['dir_train_with_icecharts']

    # preprocessing part
    hparams_data = prepare_data(hparams_data, hparams_model)
    print("preprocessing of train and test data is done!")

    # Data loader part
    np.random.seed(int(hparams_model['datamodule']['seed']))
    val_num = hparams_model['datamodule']['num_val_scenes']
    all_files_train = os.listdir(hparams_data['dir_train_with_icecharts'])
    train_list_all = [x for x in all_files_train if x.endswith(".nc")]
    validation_list = np.random.choice(train_list_all, size=int(val_num), replace=False)
    
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
  
    downscale_factor = hparams_data['data_preprocess_options']['downsampling_factor']

    
    #based on the data prep you want to experiment, you need to change the following variables in the config_data.ini file
    #hparams_data['data_preprocess_options']['land_masking'] = True
    # hparams_data['data_preprocess_options']['distance_to_border'] = True
    # hparams_data['data_preprocess_options']['distance_border_threshold'] = 20
    transform_list = get_transform_list(hparams_model)
    #transform_list = []
   

    mode = hparams_model['model']['mode'].strip("'")
    #patch extration method
    patch_with_stride = hparams_data['data_preprocess_options']['use_patch_with_stride']
    patch_with_randomcrop = hparams_data['data_preprocess_options']['use_patch_with_randomcrop']
    chart_label = hparams_model['model']['label'].strip("'")
    n_classes = hparams_data['n_classes'][chart_label]
    num_input_channels = len(hparams_data['sar_variables']) + len(hparams_data['amsr_env_variables']) + 1 # for month
    patch_size = int(hparams_data['data_preprocess_options']['patch_size'])


    if patch_with_stride:
        print("you chose patch extraction with stride")
        extractor =  PatchExtractor(hparams_data , hparams_model , chart_label )
        hparams_data = extractor.process_files()
    
    if patch_with_randomcrop:
        print("this is multiclass segmentation")
        train_dataset = BenchmarkDataset(hparams_data, hparams_model, hparams_data['train_list'],transform_list=transform_list)
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = None,  shuffle = True, num_workers = int(hparams_model['train']['num_workers']))
        val_dataset = BenchmarkTestDataset(hparams_data, hparams_model,hparams_data['validation_list'] , test = False)
        val_loader = torch.utils.data.DataLoader(val_dataset ,batch_size = None, num_workers = int(hparams_model['train']['num_workers']))
        test_dataset = BenchmarkTestDataset(hparams_data, hparams_model,hparams_data['test_list'] , test = True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = None ,num_workers = int(hparams_model['train']['num_workers']))

    
    # model , optimiser, loss functi10on setup
    model_name = hparams_model['model']['model'].strip("'")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = hparams_model['train']['batch_size']
    num_filters =  hparams_model['model']['num_filters']

    print("num of classes: ", n_classes)
    print("num input channels: ", num_input_channels)
    # Initialize model based on model name
    if model_name.lower() == 'unet':
        print("Using UNet for segmentation")
        model = UNet(hparams_data, num_filters, out_channels=n_classes, in_channels=num_input_channels)
    elif model_name.lower() == 'deeplabv3':
        print("Using DeepLabV3 for segmentation") 
        model = DeepLabV3(weights=None, num_classes=n_classes, in_channel=num_input_channels)
    else:
        raise ValueError(f"Invalid model specified: {model_name}")

    # Initialize optimizer and scheduler
    scheduler_name = hparams_model['Optimizer']['scheduler'].strip("'")
    optimizer_name = hparams_model['Optimizer']['optimizer'].strip("'")
    
    optimiser, scheduler = initialize_optimizer_and_scheduler(
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name, 
        model=model,
        hparams_model=hparams_model
    )
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=int(hparams_data['class_fill_values'][chart_label]))
    
    model_path = f'checkpoints/segmentation/{model_name}'
    train_segmentation (hparams_data , hparams_model , model ,model_name, criterion,  optimiser ,scheduler,scheduler_name, train_loader, val_loader ,device,model_path, load = False)
    print("training is done!")
    test_f1, test_accuracy, test_precision, test_recall = test_segmentation(model, test_loader, criterion , model_name,device ,  hparams_data , model_path)
        
   



    