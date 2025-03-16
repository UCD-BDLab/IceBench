import os
from src.utils.utils import *
import numpy as np
import torch
from typing import Dict, Any, Tuple
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


def base_model (model_name):
    config_data_path = "config_data.ini"
    config_data = read_config_file(config_data_path)
    hparams_data = process_settings_data(config_data)

    config_model_path = "config_model.ini"
    hparams_model = read_config_model(config_model_path)

    dir_train_with_icecharts = hparams_data['dir_train_with_icecharts']
    hparams_model['train']['batch_size'] = 16
    hparams_data['dir_train_with_icecharts'] = 'data/downscale_5/random_val/samples_labels_train_classification_224'
    hparams_data['dir_test_with_icecharts'] = 'data/downscale_5/random_val/samples_labels_test_classification_224'
    hparams_data['dir_val_with_icecharts'] = 'data/downscale_5/random_val/samples_labels_val_classification_224'
    hparams_data['meanstd_file'] = 'global_mean_std.npy'
    hparams_data['mean_std_dict'] = np.load(hparams_data['meanstd_file'], allow_pickle=True).item()
    downscale_factor = hparams_data['data_preprocess_options']['downsampling_factor']
    
    # preprocessing part
    if len(hparams_data['amsr_env_variables']) > 0:
        print("We need to upsample the amsr_env_variables")
        hparams_data = get_variable_options(hparams_data , hparams_model)



    mode = hparams_model['model']['mode'].strip("'")
    #patch extration method
    patch_with_stride = hparams_data['data_preprocess_options']['use_patch_with_stride']
    patch_with_randomcrop = hparams_data['data_preprocess_options']['use_patch_with_randomcrop']
    pure_polygons_condition = hparams_data['SIC_config']['pure_polygon']
    binary_SIC = hparams_data['SIC_config']['binary_label']
    ice_threshold_enabled = hparams_data['SIC_config']['ice_threshold_enabled']
    chart_label = hparams_model['model']['label'].strip("'")

    transform_list = get_transform_list(hparams_model)
    chart_group = str(chart_label) + "_groups"
    n_classes = len(hparams_data[chart_group]) 

    num_input_channels = len(hparams_data['sar_variables']) + len(hparams_data['amsr_env_variables']) + 1 # for month
    patch_size = int(hparams_data['data_preprocess_options']['patch_size'])

    hparams_data['meanstd_file'] = 'misc/global_meanstd.npy'


    if patch_with_stride:
        print("you chose patch extraction with stride")
        extractor =  PatchExtractor(hparams_data , hparams_model , chart_label )
        hparams_data = extractor.process_files()
    
    hparams_data['dir_samples_labels_train'] = hparams_data['dir_train_with_icecharts'] 
    hparams_data['dir_samples_labels_val'] = hparams_data['dir_val_with_icecharts']
    hparams_data['dir_samples_labels_test'] = hparams_data['dir_test_with_icecharts']
    print("here is the directory of the samples and labels for training: ", hparams_data['dir_samples_labels_train'])
    loss_func = hparams_model['loss']['loss'].strip("'")
    train_files = hparams_data['dir_samples_labels_train']
    number_train_files = os.listdir(train_files)
    print(f"the number of samples for training:{len(number_train_files)/2} validation: {len(os.listdir(hparams_data['dir_samples_labels_val']))/2} and test: {len(os.listdir(hparams_data['dir_samples_labels_test']))/2} ")
        
    if mode == "classification":
        if patch_with_randomcrop:
            # caution: check the directory of the samples and labels before running the code
            print("this is classification with patch extraction with randomcrop")
            train_dataset = ClassificationBenchmarkDataset(hparams_data,hparams_model, hparams_data['dir_train_with_icecharts'], transform_list=transform_list)
            train_loader = torch.utils.data.DataLoader(train_dataset, shuffle = True, num_workers = int(hparams_model['train']['num_workers']))
            val_dataset = BenchmarkDataset_directory(hparams_data, hparams_model,hparams_data['dir_samples_labels_val'])
            val_loader = torch.utils.data.DataLoader(val_dataset, num_workers = int(hparams_model['train']['num_workers']))
            test_dataset = BenchmarkDataset_directory(hparams_data, hparams_model,hparams_data['dir_samples_labels_test'])
            test_loader = torch.utils.data.DataLoader(test_dataset ,num_workers = int(hparams_model['train']['num_workers']))

        elif patch_with_stride:
            print("this is classification with patch extraction with stride")
            train_dataset = BenchmarkDataset_directory(hparams_data, hparams_model,hparams_data['dir_samples_labels_train'],transform_list=transform_list)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = int(hparams_model['train']['batch_size']), shuffle = True, num_workers = int(hparams_model['train']['num_workers']))
            val_dataset = BenchmarkDataset_directory(hparams_data, hparams_model,hparams_data['dir_samples_labels_val'])
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = int(hparams_model['train']['batch_size']), num_workers = int(hparams_model['train']['num_workers']))
            test_dataset = BenchmarkDataset_directory(hparams_data, hparams_model,hparams_data['dir_samples_labels_test'])
            test_loader = torch.utils.data.DataLoader(test_dataset ,batch_size = int(hparams_model['train']['batch_size']), num_workers = int(hparams_model['train']['num_workers']))
        else:
            raise ValueError("Invalid patch extraction method specified.")
    else:
        raise ValueError("Invalid mode specified.")
    
    # model , optimiser, loss function setup
    model_name = hparams_model['model']['model'].strip("'")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is: ", device)


    #print(f"you chose {model_name} for classification")
    print("n class" , n_classes)
    num_input_channels = len(hparams_data['sar_variables']) + len(hparams_data['amsr_env_variables'])  + 1

    downscale_ratio = int(hparams_data['data_preprocess_options']['downsampling_factor'])
    patch_size = int(hparams_data['data_preprocess_options']['patch_size'])
    lr = float(hparams_model['Optimizer']['lr'])
    batch_size = int(hparams_model['train']['batch_size'])
    optimiser_name = hparams_model['Optimizer']['optimizer'].strip("'")
    scheduler_name = hparams_model['Optimizer']['scheduler_name'].strip("'")
    criterion_name = hparams_model['loss']['loss'].strip("'")

    

    models = {
        'CNN': CNN(input_channels=num_input_channels, n_outputs=n_classes),
        'densenet121': ModifiedDenseNet121(pretrained=True, input_channels=num_input_channels, num_classes=n_classes),
        'resnet50': CustomResNet50(in_channels=num_input_channels, num_classes=n_classes, weights=ResNet50_Weights.DEFAULT),
        'resnet101': CustomResNet101(in_channels = num_input_channels, num_classes=n_classes , weights=ResNet101_Weights.DEFAULT),
        'resnet152': CustomResNet152 (in_channels=num_input_channels, num_classes=n_classes , weights= ResNet152_Weights.DEFAULT),
        'alexnet': ModifiedAlexNet(in_channels=num_input_channels, num_classes=n_classes , pretrained=True),
        'VGG16' : CustomVGG16(in_channels=num_input_channels, num_classes=n_classes, pretrained=True),
        #'swin' : ModifiedSwin( in_channels=num_input_channels, num_classes=n_classes, pretrained=True) ,
        #'vit' : ModifiedViT(in_channels=num_input_channels, num_classes=n_classes, pretrained=True)
    }
    if model_name not in models:
        raise ValueError("Invalid model specified.")
    
    model = models[model_name]
    optimizers = {
    'Adam': torch.optim.Adam(model.parameters(), lr=float(hparams_model['Optimizer']['lr'])),
    'SGD': torch.optim.SGD(model.parameters(), lr=float(hparams_model['Optimizer']['lr']), momentum=0.9, weight_decay= float(hparams_model['Optimizer']['weight_decay'])    ),
    }
    if optimiser_name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimiser_name}")
    optimiser = optimizers[optimiser_name]

    schedulers = {
    'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, min_lr= 1e-8, factor=0.1, patience=10, verbose=True),
    'StepLR': torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.1),
    'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=10),
    }

    if scheduler_name not in schedulers:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    scheduler = schedulers[scheduler_name]

    if criterion_name == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {criterion_name}")


    model_path = 'find_base_model/downscale_{}/patchsize_{}/{}_best_val_loss_{}_{}_{}_{}'.format(downscale_ratio, patch_size,model_name ,lr, batch_size, optimiser_name, scheduler_name)
   
    # create the directory
    model_directory = os.path.dirname(model_path)
    print("model directory is: ", model_directory)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
    print(f"Configuration of Training and Data: Model: {model_name}, Patch Size: {patch_size}, Downscale Ratio: {downscale_ratio}, Optimizer: {optimiser_name}, Scheduler: {scheduler_name}, Learning Rate: {lr}, Batch Size: {batch_size}")
    

    val_loss, f1_score_val =train_and_evaluate_classification (hparams_model , model, model_name, criterion, optimiser,scheduler, train_loader, val_loader ,model_path, device )
    test_accuracy, test_precision, test_recall, test_f1 = test_classification(model, model_name, criterion, test_loader, model_path,device)
       

