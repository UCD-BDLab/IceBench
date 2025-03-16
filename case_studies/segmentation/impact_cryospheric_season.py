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


def initialize_model(model_name, num_input_channels, n_classes, num_filters):
    """
    Initialize a segmentation model based on the model name.

    Args:
        model_name (str): Name of the model ('unet', 'deeplabv3').
        num_input_channels (int): Number of input channels.
        n_classes (int): Number of output classes.
        num_filters (list): List of filter sizes for the model.

    Returns:
        nn.Module: Initialized model.
    """
    if model_name.lower() == 'unet':
        return UNet(in_channels=num_input_channels, out_channels=n_classes, num_filters=num_filters)
    elif model_name.lower() == 'deeplabv3':
        return DeepLabV3(weights=None, num_classes=n_classes, in_channel=num_input_channels)
    else:
        raise ValueError(f"Unsupported model: {model_name}")



def impact_cryospheric_season(season):
    """
    Main function to train and evaluate the model based on the season (melt or freeze).
    
    Args:
        season (str): The season to analyze ('melt' or 'freeze')
    """
    # Load configs
    hparams_data, hparams_model = load_configs()
    season = hparams_data['train_data_options']['season'].strip("'")
    set_seed(int(hparams_model['datamodule']['seed']))
    
    # Load season-specific files
    print("season: ", season)
    season = season.lower()
    if season == 'melt':
        file_names = read_filenames_from_json("melt_files.json")
        print("Loaded filenames from melt_files.json")
    elif season == 'freeze':
        file_names = read_filenames_from_json("freeze_files.json")
        print("Loaded filenames from freeze_files.json")
    else:
        raise ValueError("Invalid season specified.")

    # Data preprocessing
    hparams_data = prepare_data(hparams_data, hparams_model)
    print("preprocessing of train and test data is done!")


    ### data loader part
    
    val_num = int(hparams_model['datamodule']['num_val_scenes'])
    train_list_all = file_names

    validation_list  = np.random.choice(train_list_all, size=val_num, replace=False)
    print(f"Selected validation scenes: {', '.join(validation_list)}")
    train_list_season = [x for x in train_list_all if x not in validation_list]
    test_list = os.listdir(hparams_data['dir_test_with_icecharts'])

    # update thehparams_data
    hparams_data.update({
        'train_list': train_list_season,
        'validation_list': validation_list,
        'test_list': test_list,
        'meanstd_file': 'misc/global_meanstd.npy'
    })
    print(f"len of train list: {len(train_list_season)} and len of validation list: {len(validation_list)} and len of test list: {len(test_list)}")

    mode = hparams_model['model']['mode'].strip("'")
    #patch extration method
    patch_with_stride = hparams_data['data_preprocess_options']['use_patch_with_stride']
    patch_with_randomcrop = hparams_data['data_preprocess_options']['use_patch_with_randomcrop']
    pure_polygons_condition = hparams_data['SIC_config']['pure_polygon']
    chart_label = hparams_model['model']['label'].strip("'")

    transform_list = get_transform_list(hparams_model)
    n_classes = hparams_data['n_classes'][chart_label]

    num_input_channels = len(hparams_data['sar_variables']) + len(hparams_data['amsr_env_variables']) + 1 # for month
    patch_size = int(hparams_data['data_preprocess_options']['patch_size'])

    
    if mode == "segmentation":
        if patch_with_randomcrop:
            print("this is multiclass segmentation")
            train_dataset = BenchmarkDataset(hparams_data, hparams_model, hparams_data['train_list'], transform_list=transform_list)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=int(hparams_model['train']['num_workers']))
            val_dataset = BenchmarkTestDataset(hparams_data, hparams_model, hparams_data['validation_list'], test=False)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=None, num_workers=int(hparams_model['train']['num_workers']))
            test_dataset = BenchmarkTestDataset(hparams_data, hparams_model, hparams_data['test_list'], test=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None, num_workers=int(hparams_model['train']['num_workers']))
        else:
            raise ValueError("Patch with random crop is not used")
    

    model_name = hparams_model['model']['model'].strip("'")
    model_name = model_name.lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = hparams_model['train']['batch_size']
    num_filters = hparams_model['model']['num_filters']

    print(f"Number of classes in segmentation model: {n_classes}")
    print(f"Number of input channels in model: {num_input_channels}")

    if model_name == 'Unet':
        print("you chose Unet for segmentation with CE loss")
        num_filters = hparams_model['model']['num_filters']
        model = UNet(hparams_data, num_filters, out_channels=n_classes, in_channels=num_input_channels)
    elif model_name == 'deeplabv3':
        print("you chose deeplabv3 for segmentation")
        model = DeepLabV3(weights=None, num_classes=n_classes, in_channel=num_input_channels)
    else:
        raise ValueError("Invalid model specified.")

    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=int(hparams_data['class_fill_values'][chart_label]))
    # Initialize optimizer and scheduler
    optimiser_name = hparams_model['Optimizer']['optimizer'].strip("'")
    scheduler_name = hparams_model['Optimizer']['scheduler'].strip("'")
    lr = float(hparams_model['Optimizer']['lr'])
    model_path = os.path.join('models', f'{model_name}_{optimiser_name}_{scheduler_name}_{lr}.pth')

    optimiser, scheduler = initialize_optimizer_and_scheduler(
        optimizer_name=optimiser_name,
        scheduler_name=scheduler_name, 
        model=model,
        hparams_model=hparams_model
    )
    print("start training the model")
    train_segmentation (hparams_data , hparams_model , model ,model_name, criterion,  optimiser ,scheduler,scheduler_name, train_loader, val_loader ,device,model_path, load = False)                  
    print("training is done!")

    test_f1, test_accuracy, test_precision, test_recall = test_segmentation(model, test_loader, criterion, model_name, device, hparams_data, model_path)

    #print("testing the model with the two seasons and four locations")
    #f1 , accuracy, percision , recall = test_segmentation_for_two_seasosn(model, test_loader, criterion , model_name, device , hparams_data , model_path, season_prediction=True, location_prediction=True)



    

