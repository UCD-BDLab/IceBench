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
from src.segmentation.training.evaluate_segmentation import test_segmentation , test_segmentation_for_two_seasosn

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





def impact_cryospheric_season_each_location():
    """
    Evaluate the impact of cryospheric seasons and locations on sea ice segmentation performance.

    This function assesses how different cryospheric seasons (melt and freeze) and specific 
    locations affect the accuracy of sea ice segmentation models. It uses pre-processed data 
    based on the NASA Earth Observatory's Arctic Sea Ice Melt dataset 
    (https://earth.gsfc.nasa.gov/cryo/data/arctic-sea-ice-melt).

    Key features:
    1. Season-specific data: Uses separate datasets for melt and freeze seasons
    2. Location-specific analysis: Allows for evaluation of different Arctic regions
    3. Pre-processed data loading: Utilizes JSON files containing pre-categorized file lists
    4. Flexible model architecture: Supports different models like UNet and DeepLabV3
    5. Comprehensive evaluation: Provides performance metrics for each season-location combination

    Workflow:
    1. Load configuration settings and prepare data
    2. Load pre-processed file lists for the specified season and location from JSON
    3. Set up data loaders for training, validation, and testing
    4. Initialize and train the segmentation model
    5. Evaluate the model's performance on the test set

    Usage:
    - Ensure that JSON files with pre-categorized file lists exist in the 'files_json' directory
    - JSON files should follow the naming convention: "{season}_{location}_train_files.json"
    - The function will automatically load the appropriate dataset based on the specified season and location

    Note:
    - This approach allows for detailed analysis of how sea ice segmentation performance 
      varies between melt and freeze seasons in different Arctic locations
    - Ensure that the NASA dataset has been properly pre-processed and categorized before running this function
    
    """
    hparams_data, hparams_model = load_configs()

    set_seed(int(hparams_model['datamodule']['seed']))

    # Data Preparation
    season = hparams_data ['train_data_options']['season'].strip("'") 
    loc = hparams_data['train_data_options']['location']
    

    # Model Setup
    model_name = hparams_model['model']['model'].strip("'")
    num_input_channels = len(hparams_data['sar_variables']) + len(hparams_data['amsr_env_variables']) + 1 # 1 for month
    n_classes = hparams_data['n_classes'][hparams_model['model']['label'].strip("'")]
    num_filters = hparams_model['model']['num_filters']
    transform_list = get_transform_list(hparams_model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # preprocessing part
    hparams_data = prepare_data(hparams_data , hparams_model )#, hparams_data['dir_train_without_icecharts'], hparams_data['dir_train_with_icecharts'])
    print("preprocessing of train and test data is done!")

    all_files_train = os.listdir(hparams_data['dir_train_with_icecharts'])
    train_list = [x for x in all_files_train if x.endswith(".nc")]
    
    #find all location_Files name for each category
    current_loc_categories = hparams_data['train_data_options']['location']
    print("current_loc_categories: ", current_loc_categories)
    
    json_filename = f"files_json/{season}_{loc}_train_files.json"
    print("json file name: ", json_filename)
    season_loc_files = read_filenames_from_json(json_filename.strip("'"))
    print(f"Location {current_loc_categories} and {season} and length :", len(season_loc_files))


    num_val_scenes = int(hparams_model['datamodule']['num_val_scenes'])
    validation_list  = np.random.choice(season_loc_files, size=num_val_scenes, replace=False)
    
    print(f"Validation list for {season} season, category {current_loc_categories}:" , validation_list , len(validation_list))

    train_list = [x for x in season_loc_files if x not in validation_list]
    test_list = os.listdir(hparams_data['dir_test_with_icecharts'])
    

    # Update hparams_data
    hparams_data.update({
        'train_list': train_list,
        'validation_list': validation_list,
        'test_list': test_list,
        'meanstd_file': 'misc/global_meanstd.npy'
        
    })

    print(f"len of train list: {len(train_list)} and len of validation list: {len(validation_list)} and len of test list: {len(test_list)}")

    mode = hparams_model['model']['mode'].strip("'")
    #patch extration method
    patch_with_stride = hparams_data['data_preprocess_options']['use_patch_with_stride']
    patch_with_randomcrop = hparams_data['data_preprocess_options']['use_patch_with_randomcrop']
    chart_label = hparams_model['model']['label'].strip("'")

    n_classes = hparams_data['n_classes'][chart_label]

    num_input_channels = len(hparams_data['sar_variables']) + len(hparams_data['amsr_env_variables']) + 1 # for month

    if patch_with_randomcrop:
        print("The patch with random crop is used")
        train_dataset = BenchmarkDataset(hparams_data, hparams_model, hparams_data['train_list'], transform_list=transform_list)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=int(hparams_model['train']['num_workers']))
        val_dataset = BenchmarkTestDataset(hparams_data, hparams_model, hparams_data['validation_list'], test=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=None, num_workers=int(hparams_model['train']['num_workers']))
        test_dataset = BenchmarkTestDataset(hparams_data, hparams_model, hparams_data['test_list'], test=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None, num_workers=int(hparams_model['train']['num_workers']))
    else:
        raise ValueError("Invalid mode specified.")


    model_name = hparams_model['model']['model'].strip("'")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = hparams_model['train']['batch_size']
    num_filters = hparams_model['model']['num_filters']


    print(f"Number of classes: {n_classes}")
    print(f"Number of input channels: {num_input_channels}")

    # Initialize model, optimizer and scheduler
    model_name = hparams_model['model']['model'].strip("'")
    scheduler_name = hparams_model['Optimizer']['scheduler'].strip("'") 
    optimiser_name = hparams_model['Optimizer']['optimizer'].strip("'")

    if model_name == 'Unet':
        print("you chose Unet for segmentation with CE loss")
        num_filters = hparams_model['model']['num_filters']
        model = UNet(hparams_data, num_filters, out_channels=n_classes, in_channels=num_input_channels)
    elif model_name == 'deeplabv3':
        print("you chose deeplabv3 for segmentation")
        model = DeepLabV3(weights=None, num_classes=n_classes, in_channel=num_input_channels)
    else:
        raise ValueError("Invalid model specified.")
    
    optimiser, scheduler = initialize_optimizer_and_scheduler(
        optimizer_name=optimiser_name,
        scheduler_name=scheduler_name,
        model=model,
        hparams_model=hparams_model
    )
    
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=int(hparams_data['class_fill_values'][chart_label]))

    model_path = f'checkpoints/segmentation/{model_name}_{season}_{loc}'
    print("start training the model")
    train_segmentation (hparams_data , hparams_model , model ,model_name, criterion,  optimiser ,scheduler,scheduler_name, train_loader, val_loader ,device,model_path, load = False)
                    
    print("training is done!")
    #teest the model with the test_segmentaion and mention on all the test list
    print("testing the model on all the test list")
    test_f1, test_accuracy, test_precision, test_recall = test_segmentation(model, test_loader, criterion, model_name, device, hparams_data, model_path)


    #print("testing the model with the two seasons and four locations")
    #f1 , accuracy, percision , recall = test_segmentation_for_two_seasosn(model, test_loader, criterion , model_name, device , hparams_data , model_path, season_prediction=True, location_prediction=True)



    

