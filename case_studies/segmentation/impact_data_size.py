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



def impact_datasize():
    """
    Evaluate the impact of dataset size on sea ice segmentation performance.

    This function assesses how varying amounts of training data affect the accuracy 
    of sea ice segmentation models. It uses a unique approach to data selection:

    1. Patch generation: Patches are first generated from all available training files.
    2. Random selection: A specified number of patches (defined by 'data_size') are 
       randomly selected from this pool for training.
    3. Consistent validation: The validation set remains the same as the base model, 
       ensuring consistent evaluation across different data size experiments.
    The function performs these main steps:
    1. Loads and processes configuration settings.
    2. Prepares the dataset, selecting a subset based on the specified data size.
    3. Sets up data loaders and initializes the segmentation model.
    4. Trains the model on the selected data subset.
    5. Evaluates the model's performance, providing metrics for the given data size.

    Note: This function uses pre-generated patches stored in a specific directory,
    which allows for consistent and repeatable experiments across different data sizes.
    The validation set remains constant to ensure fair comparisons between experiments.
    """

    # Load configs first
    hparams_data, hparams_model = load_configs()
    set_seed(int(hparams_model['datamodule']['seed']))
    dir_train_with_icecharts = hparams_data['dir_train_with_icecharts']

    # preprocessing part
    hparams_data = prepare_data(hparams_data, hparams_model)
    print("preprocessing of train and test data is done!")

    # Data loader part
    np.random.seed(int(hparams_model['datamodule']['seed']))
    val_num = hparams_model['datamodule']['num_val_scenes']

    hparams_data['data_preprocess_options']['use_patch_with_stride'] = True
    hparams_data['data_preprocess_options']['use_patch_with_randomcrop'] = False

    
    data_size = int(hparams_model['datamodule']['data_size'])
    #hparams_data['dir_samples_labels_train'] = 
   
    test_list = os.listdir(hparams_data['dir_test_with_icecharts'])


    mode = hparams_model['model']['mode'].strip("'")
    #patch extration method
    patch_with_stride = hparams_data['data_preprocess_options']['use_patch_with_stride']
    patch_with_randomcrop = hparams_data['data_preprocess_options']['use_patch_with_randomcrop']
    chart_label = hparams_model['model']['label'].strip("'")

    transform_list = get_transform_list(hparams_model)
    n_classes = hparams_data['n_classes'][chart_label]

    num_input_channels = len(hparams_data['sar_variables']) + len(hparams_data['amsr_env_variables']) + 1 # for month
    patch_size = int(hparams_data['data_preprocess_options']['patch_size'])
    batch_size = int(hparams_model['datamodule']['batch_size'])
    hparams_data['meanstd_file'] = 'misc/global_meanstd.npy'


    loss_func = hparams_model['loss']['loss'].strip("'")
    

    

    if patch_with_randomcrop:
        train_dir = hparams_data['dir_train_with_icecharts']
        train_files = os.listdir(train_dir)
        train_list = [x for x in train_files if x.endswith(".nc")]
        hparams_data['train_list'] = train_list
        validation_list = np.random.choice(train_list, size=int(val_num), replace=False)
        hparams_data['validation_list'] = validation_list
        test_list = os.listdir(hparams_data['dir_test_with_icecharts'])
        hparams_data['test_list'] = test_list   
        # update thehparams_data
        hparams_data['train_list'] = hparams_data['sample_files']
        hparams_data['validation_list'] = validation_list
        hparams_data['test_list'] = test_list


        train_dataset = BenchmarkDataset(hparams_data, hparams_model, hparams_data['train_list'],transform_list=transform_list)
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = None,  shuffle = True, num_workers = int(hparams_model['train']['num_workers']))
        val_dataset = BenchmarkTestDataset(hparams_data, hparams_model,hparams_data['validation_list'] , test = False)
        val_loader = torch.utils.data.DataLoader(val_dataset ,batch_size = None, num_workers = int(hparams_model['train']['num_workers']))
        test_dataset = BenchmarkTestDataset(hparams_data, hparams_model,hparams_data['test_list'] , test = True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = None ,num_workers = int(hparams_model['train']['num_workers']))

    elif patch_with_stride:
        if patch_with_stride:
            print("you chose patch extraction with stride")
            extractor =  PatchExtractor(hparams_data , hparams_model , chart_label )
            out_dir = 'data_segmentation/patch_extraction_stride'
            hparams_data = extractor.process_files(out_dir)

        train_dir = hparams_data['dir_samples_labels_train']
        dir_list = os.listdir(train_dir)
        samples_list = [x for x in dir_list if x.endswith("_samples.npy")]
        #samples_selected = np.random.choice(samples_list, data_size, replace=False)
        samples_selected = np.random.choice(samples_list, data_size, replace=True)
        hparams_data['sample_files'] = samples_selected.tolist()

        train_dataset = BenchmarkDataset_Segmentation_directory(hparams_data, hparams_model,hparams_data['dir_samples_labels_train'],transform_list=transform_list)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = int(hparams_model['train']['num_workers']))
        val_dataset = BenchmarkTestDataset(hparams_data, hparams_model,hparams_data['validation_list'] , test = False)
        val_loader = torch.utils.data.DataLoader(val_dataset ,batch_size = None, num_workers = int(hparams_model['train']['num_workers']))
        test_dataset = BenchmarkTestDataset(hparams_data, hparams_model,hparams_data['test_list'] , test = True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = None ,num_workers = int(hparams_model['train']['num_workers']))

    else:
        raise ValueError("Invalid patch extraction method specified.")
        

    
    # model , optimiser, loss functi10on setup
    model_name = hparams_model['model']['model'].strip("'")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model_path = f'checkpoints/segmentation/{model_name}_{data_size}'
    optimiser, scheduler = initialize_optimizer_and_scheduler(optimiser_name, scheduler_name, model, hparams_model)

    # Train and test
    train_segmentation(hparams_data, hparams_model, model, model_name, criterion, optimiser, scheduler, train_loader, val_loader, device, load=False)
    test_f1, test_accuracy, test_precision, test_recall = test_segmentation(model, test_loader, criterion, model_name, device, hparams_data)

   

    
    

