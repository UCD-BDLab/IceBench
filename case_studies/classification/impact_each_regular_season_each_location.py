import os
from utils.utils import *
import numpy as np
import torch
from torchvision.models.resnet import ResNet50_Weights, ResNet101_Weights , ResNet152_Weights
from src.classification.data_loader.data_loader_classification import *
from src.classification.models.model_cnn import CNN
from src.classification.models.model_densenet import ModifiedDenseNet121
from src.classification.models.model_resnet import CustomResNet50, CustomResNet101, CustomResNet152
from src.classification.models.model_alexnet import AlexNet , ModifiedAlexNet
from src.classification.models.model_VGG16 import CustomVGG16
from src.classification.training_evaluate.train_classification import train_and_evaluate_classification
from src.classification.training_evaluate.evaluate_classification import test_classification
from src.classification.data_loader.Data_module import *
from src.classification.training_evaluate.Model_module import *
from src.classification.models.model_resnet import *
from src.utils.patch_extraction import PatchExtractor
from src.data_preprocessing.prepare_data import prepare_data
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


def set_seed(seed_value=42):
    """Set random seed for reproducibility."""
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

def initialize_model(model_name, num_input_channels, n_classes, hparams_model):
    """Initialize the specified model with the given input channels and output classes."""
    print(f"Initializing model: {model_name}")
    
    # Get pretrained setting from config, default to True if not specified
    pretrained = hparams_model.get('model', {}).get('pretrained', 'True').lower() == 'true'
    
    models = {
        'CNN': CNN(input_channels=num_input_channels, n_outputs=num_input_channels),
        'densenet121': ModifiedDenseNet121(pretrained=pretrained, input_channels=num_input_channels, num_classes=n_classes),
        'resnet50': CustomResNet50(in_channels=num_input_channels, num_classes=n_classes, 
                                 weights=ResNet50_Weights.DEFAULT if pretrained else None),
        'resnet101': CustomResNet101(in_channels=num_input_channels, num_classes=n_classes,
                                   weights=ResNet101_Weights.DEFAULT if pretrained else None),
        'resnet152': CustomResNet152(in_channels=num_input_channels, num_classes=n_classes,
                                   weights=ResNet152_Weights.DEFAULT if pretrained else None),
        'alexnet': ModifiedAlexNet(in_channels=num_input_channels, num_classes=n_classes, pretrained=pretrained),
        'VGG16': CustomVGG16(in_channels=num_input_channels, num_classes=n_classes, pretrained=pretrained)
    }

    if model_name not in models:
        raise ValueError(f"Invalid model specified: {model_name}")

    return models[model_name]

def initialize_optimizer_scheduler(optimiser_name, scheduler_name, model, lr):
    """Initialize optimizer and scheduler based on the configuration."""
    print(f"Initializing optimizer: {optimiser_name} and scheduler: {scheduler_name}")
    
    # weight_decay = float(hparams_model['Optimizer'].get('weight_decay', 0.01))
    # reduce_lr_patience = int(hparams_model['Optimizer'].get('reduce_lr_patience', 4))

    optimizers = {
        'Adam': torch.optim.Adam(model.parameters(), lr=lr),
        'SGD': torch.optim.SGD(model.parameters(), lr=lr)
    }
    if optimiser_name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimiser_name}")

    schedulers = {
        'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[optimiser_name], mode='min', factor=0.1, patience=5, verbose=True),
        'StepLR': torch.optim.lr_scheduler.StepLR(optimizers[optimiser_name], step_size=10, gamma=0.1),
        'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[optimiser_name], T_max=10),
    }
    if scheduler_name not in schedulers:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return optimizers[optimiser_name], schedulers[scheduler_name]


def each_location_each_regular_season():
    """
    Trains and evaluates a classification model for each specified season and location.

    This function sets up and executes a pipeline for training and evaluating a classification model.
    It supports training on different seasons (e.g., fall, spring) and locations. The function handles data preprocessing,
    model selection, training, and evaluation based on configurations specified in the configuration files.

    Configuration for seasons and locations is managed via the configuration files (config_data.ini and config_model.ini).
    Ensure that these files are properly set up with the required parameters for different seasons and locations.

    Returns:
        None
    """
    hparams_data, hparams_model = load_configs()

    hparams_data['meanstd_file'] = 'global_mean_std.npy'
    hparams_data['mean_std_dict'] = np.load(hparams_data['meanstd_file'], allow_pickle=True).item()
    
    
    # preprocessing part
    if len(hparams_data['amsr_env_variables']) > 0:
        print("We need to upsample the amsr_env_variables")
        hparams_data = get_variable_options(hparams_data , hparams_model)
    print("preprocessing of train and test data is done!")


    seed_value = int( hparams_model['datamodule']['seed'])
    set_seed(seed_value)
    
    val_num = hparams_model['datamodule']['num_val_scenes']

    seasonality = hparams_data ['train_data_options']['season'].strip("'")
    location = hparams_data ['train_data_options']['location'].strip("'")
    print(" this is training for season : ", seasonality)

    mode = hparams_model['model']['mode'].strip("'")
    
    #patch extration method
    patch_with_stride = hparams_data['data_preprocess_options']['use_patch_with_stride']
    patch_with_randomcrop = hparams_data['data_preprocess_options']['use_patch_with_randomcrop']
    chart_label = hparams_model['model']['label'].strip("'")

    transform_list = get_transform_list(hparams_model)
    n_classes = hparams_data['n_classes'][chart_label] -1 # -1 because we are not considering the background class

    num_input_channels = len(hparams_data['sar_variables']) + len(hparams_data['amsr_env_variables']) + 1 # for month


    if patch_with_stride:
        print("you chose patch extraction with stride")
        extractor =  PatchExtractor(hparams_data , hparams_model , chart_label )
        hparams_data = extractor.process_files_season_location(seasonality , location[0])
    
    
    hparams_data['dir_samples_labels_train'] = hparams_data['dir_train_with_icecharts'] 
    hparams_data['dir_samples_labels_val'] = hparams_data['dir_val_with_icecharts']
    hparams_data['dir_samples_labels_test'] = hparams_data['dir_test_with_icecharts']

    train_files = hparams_data['dir_samples_labels_train']

    # model , optimiser, loss function and scheduler
    model_name = hparams_model['model']['model'].strip("'")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_input_channels = len(hparams_data['sar_variables']) + len(hparams_data['amsr_env_variables'])  + 1
    downscale_ratio = int(hparams_data['data_preprocess_options']['downsampling_factor'])
    patch_size = int(hparams_data['data_preprocess_options']['patch_size'])
    lr = float(hparams_model['Optimizer']['lr'])
    batch_size = int(hparams_model['train']['batch_size'])
    optimiser_name = hparams_model['Optimizer']['optimizer'].strip("'")
    scheduler_name = hparams_model['Optimizer']['scheduler'].strip("'")


    model = initialize_model(model_name, num_input_channels, n_classes , hparams_model)
    model = model.to(device)

    optimiser, scheduler = initialize_optimizer_scheduler(optimiser_name, scheduler_name, model, lr)

    criterion = torch.nn.CrossEntropyLoss()
    
    data_module = MyDataModule(hparams_data, hparams_model, transform_list)
    data_module.setup()

    checkpoint_dir = f'./checkpoints/{model_name}/{seasonality}_{location[0]}'
    checkpoint_dir_f1 = f'./checkpoints/{model_name}/{seasonality}_{location[0]}'
    model_file_name = '{}_best_val_loss_{}'.format(model_name, seasonality, location[0])
    model_file_name_f1 = '{}_best_val_f1_{}'.format(model_name, seasonality, location[0])
                    
    tb_logger = TensorBoardLogger(save_dir='./lightning_logs', name=f'{model_name}_{seasonality}_{location[0]}')


    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=int(hparams_model['train']['patience']), mode='min')
                    #early_stop_callback = pl.callbacks.EarlyStopping( monitor='val_f1', patience=int(hparams_model['train']['patience']), mode='max')
    checkpoint_callback_loss = pl.callbacks.ModelCheckpoint(
                        dirpath=checkpoint_dir,
                        filename= model_file_name,
                        save_top_k=1,
                        monitor='val_loss',
                        mode='min'
                    )
    checkpoint_callback_f1 = pl.callbacks.ModelCheckpoint(
                        dirpath=checkpoint_dir_f1,
                        filename= model_file_name_f1,
                        save_top_k=1,
                        monitor='val_f1',
                        mode='max'
                    )

                    
    model_pl = MyModel(model, criterion, optimiser, scheduler)
    data_module.setup('fit')
    trainer = pl.Trainer(
                        max_epochs=int(hparams_model['train']['epochs']),
                        callbacks=[early_stop_callback, checkpoint_callback_loss, checkpoint_callback_f1],
                        logger=tb_logger,
                        devices=1 if torch.cuda.is_available() else None
                        
                    )
    trainer.fit(model_pl,datamodule=data_module)
                    
    
    data_module.setup('test')
    model_load_loss = os.path.join(checkpoint_dir, model_file_name)
    model_load_loss = model_load_loss + '.ckpt'
    best_model_loss = MyModel.load_from_checkpoint(checkpoint_path=model_load_loss, model=model,
                        criterion=criterion,
                        optimiser=optimiser,
                        scheduler=scheduler)

                    # Load the best model based on validation F1 score
    model_load_f1 = os.path.join(checkpoint_dir_f1, model_file_name_f1)
    model_load_f1 = model_load_f1 + '.ckpt'
    best_model_f1 = MyModel.load_from_checkpoint(checkpoint_path=model_load_f1 ,    model=model,
                     criterion=criterion,
                    optimiser=optimiser,
                    scheduler=scheduler)

    # Test the best models
    test_result_loss = trainer.test(best_model_loss, datamodule=data_module)
    test_result_f1 = trainer.test(best_model_f1, datamodule=data_module)

    # Print the test results for the best models
    print("Best model based on validation loss:")
    print(f"Test Results: F1 Score: {test_result_loss[0]['test_f1']:.4f}, Accuracy: {test_result_loss[0]['test_accuracy']:.4f}, Precision: {test_result_loss[0]['test_precision']:.4f}, Recall: {test_result_loss[0]['test_recall']:.4f}")

    print("Best model based on validation F1 score:")
    print(f"Test Results:F1 Score: {test_result_f1[0]['test_f1']:.4f}, Accuracy: {test_result_f1[0]['test_accuracy']:.4f}, Precision: {test_result_f1[0]['test_precision']:.4f}, Recall: {test_result_f1[0]['test_recall']:.4f}")




    

  