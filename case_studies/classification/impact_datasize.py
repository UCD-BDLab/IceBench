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




def train_and_evaluate(hparams_data, hparams_model, transform_list, model, criterion, optimiser, scheduler, model_name , datasize):
    """Train and evaluate the classification model."""
    data_module = MyDataModule(hparams_data, hparams_model, transform_list)
    data_module.setup()


    checkpoint_dir = f'./checkpoints/datasize'
    checkpoint_dir_f1 = f'./checkpoints/datasize'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    

    model_file_name = '{}_best_val_loss_{}'.format(model_name , datasize)
    model_file_name_f1 = '{}_best_val_f1_{}'.format(model_name , datasize)
                    
    tb_logger = TensorBoardLogger(save_dir='./lightning_logs', name=f'{model_name}_{datasize}')


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
          
                    
    model_load_loss = os.path.join(checkpoint_dir, model_file_name)
    model_load_loss = model_load_loss + '.ckpt'
    best_model_loss = MyModel.load_from_checkpoint(checkpoint_path=model_load_loss, model=model,
                            criterion=criterion,
                            optimiser=optimiser,
                            scheduler=scheduler ,
                            #class_weights=class_weights
                            )

                        # Load the best model based on validation F1 score
    model_load_f1 = os.path.join(checkpoint_dir_f1, model_file_name_f1)
    model_load_f1 = model_load_f1 + '.ckpt'
    best_model_f1 = MyModel.load_from_checkpoint(checkpoint_path=model_load_f1 ,    model=model,
                         criterion=criterion,
                        optimiser=optimiser,
                        scheduler=scheduler )

                        # Test the best models
    data_module.setup('test')
    test_result_loss = trainer.test(best_model_loss, datamodule=data_module)
    test_result_f1 = trainer.test(best_model_f1, datamodule=data_module)


    print("Best model based on validation loss:")
    print(f"Test Results: F1 Score: {test_result_loss[0]['test_f1']:.4f}, Accuracy: {test_result_loss[0]['test_accuracy']:.4f}, Precision: {test_result_loss[0]['test_precision']:.4f}, Recall: {test_result_loss[0]['test_recall']:.4f}")

    print("Best model based on validation F1 score:")
    print(f"Test Results:F1 Score: {test_result_f1[0]['test_f1']:.4f}, Accuracy: {test_result_f1[0]['test_accuracy']:.4f}, Precision: {test_result_f1[0]['test_precision']:.4f}, Recall: {test_result_f1[0]['test_recall']:.4f}")


def imapct_datasize():
    hparams_data, hparams_model = load_configs()

    datasize = int(hparams_model['datamodule']['datasize'])
    
    # preprocessing part
    if len(hparams_data['amsr_env_variables']) > 0:
        print("We need to upsample the amsr_env_variables")
        hparams_data = get_variable_options(hparams_data , hparams_model)
    print("preprocessing of train and test data is done!")

    seed_value = int( hparams_model['datamodule']['seed'])
    set_seed(seed_value)
      

    mode = hparams_model['model']['mode'].strip("'")
    #patch extration method
    patch_with_stride = hparams_data['data_preprocess_options']['use_patch_with_stride']
    patch_with_randomcrop = hparams_data['data_preprocess_options']['use_patch_with_randomcrop']
    chart_label = hparams_model['model']['label'].strip("'")

    transform_list = get_transform_list(hparams_model)
    n_classes = hparams_data['n_classes'][chart_label] - 1

    num_input_channels = len(hparams_data['sar_variables']) + len(hparams_data['amsr_env_variables']) + 1 # for month
    #print("number of input channels: ", num_input_channels)
    patch_size = int(hparams_data['data_preprocess_options']['patch_size'])
    
    hparams_data['dir_samples_labels_train'] = hparams_data['dir_train_with_icecharts'] 
    hparams_data['dir_samples_labels_val'] = hparams_data['dir_val_with_icecharts']
    hparams_data['dir_samples_labels_test'] = hparams_data['dir_test_with_icecharts']

   
    train_files = hparams_data['dir_samples_labels_train']
    all_sample_files = [file for file in os.listdir(train_files) if file.endswith("_samples.npy")]
    number_train_files = len(all_sample_files)

    if datasize < number_train_files:
        # Randomly pick datasize samples from the train data
        random_indices = np.random.choice(number_train_files, datasize, replace=False)
        selected_sample_files = [all_sample_files[i] for i in random_indices]
        selected_label_files = [file.replace("_samples.npy", "_labels.npy") for file in selected_sample_files]
        hparams_data['dir_samples_labels_train'] = train_files
        hparams_data['selected_train_files'] = selected_sample_files
        hparams_data['selected_label_files'] = selected_label_files
    else:
        print("The number of samples in the training data is less than the specified datasize. break it")
        return None

    print(f"the number of samples for training:{len(selected_sample_files)} validation: {len(os.listdir(hparams_data['dir_samples_labels_val']))/2} and test: {len(os.listdir(hparams_data['dir_samples_labels_test']))/2} ")
      
    
    # model , optimiser, loss functi10on setup
    model_name = hparams_model['model']['model'].strip("'")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_input_channels = len(hparams_data['sar_variables']) + len(hparams_data['amsr_env_variables'])  + 1
    lr = float(hparams_model['Optimizer']['lr'])
    optimiser_name = hparams_model['Optimizer']['optimizer'].strip("'")
    scheduler_name = hparams_model['Optimizer']['scheduler'].strip("'")

    model = initialize_model(model_name, num_input_channels, n_classes , hparams_model)
    model = model.to(device)
    optimiser, scheduler = initialize_optimizer_scheduler(optimiser_name, scheduler_name, model, lr)

    criterion = torch.nn.CrossEntropyLoss()
    
    train_and_evaluate(hparams_data, hparams_model, transform_list, model, criterion, optimiser, scheduler, model_name , datasize)
  








