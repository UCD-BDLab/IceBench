import torch
import numpy as np
import os
import matplotlib.pyplot as plt
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
from captum.attr import  FeatureAblation , GradientShap ,  IntegratedGradients,  DeepLift
import torch.nn as nn

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

def feature_importance():

    seed_value = int(hparams_model['datamodule']['seed'])

    # Set the seed for PyTorch
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    config_data_path = "config_data.ini"
    config_data = read_config_file(config_data_path)
    hparams_data = process_settings_data(config_data)

    config_model_path = "config_model.ini"
    hparams_model = read_config_model(config_model_path)
    
    dir_train_with_icecharts = hparams_data['dir_train_with_icecharts']

    hparams_data['meanstd_file'] = 'global_mean_std.npy'
    hparams_data['mean_std_dict'] = np.load(hparams_data['meanstd_file'], allow_pickle=True).item()
    

  
    # preprocessing part
    hparams_data = prepare_data(hparams_data , hparams_model )#, hparams_data['dir_train_without_icecharts'], hparams_data['dir_train_with_icecharts'])
    print("preprocessing of train and test data is done!")

 

    ### data loader part
    np.random.seed(int(hparams_model['datamodule']['seed']))
    
    test_list = os.listdir(hparams_data['dir_test_with_icecharts'])

  
    hparams_data['test_list'] = test_list

    mode = hparams_model['model']['mode'].strip("'")
    #patch extration method
    chart_label = hparams_model['model']['label'].strip("'")
    n_classes = hparams_data['n_classes'][chart_label]

    num_input_channels = len(hparams_data['sar_variables']) + len(hparams_data['amsr_env_variables']) + 1 # for month

    test_files = os.listdir(hparams_data['dir_test_with_icecharts'])
    
    test_dataset = BenchmarkTestDataset(hparams_data, hparams_model, hparams_data['test_list'], test = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = None ,num_workers = int(hparams_model['train']['num_workers']))

        
    
    model_name = hparams_model['model']['model'].strip("'")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    num_input_channels = len(hparams_data['sar_variables']) + len(hparams_data['amsr_env_variables']) + 1 # for month

    num_filters = hparams_model['model']['num_filters']
    # based on the model name, we can determine the model
    if model_name == 'Unet':
        model = UNet(hparams_data,num_filters, out_channels= n_classes , in_channels= num_input_channels)
    elif model_name == 'deeplabv3':
        model = DeepLabV3(hparams_data,num_filters, out_channels= n_classes , in_channels= num_input_channels)
    
    save_path = 'captum_plots'
    # check if it is not available then create a new one
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_path = hparams_model['model']['checkpoint_path'].strip("'")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)

    model.eval()

    #features_name = ['HH', 'HV', 'Incidence Angle', 'Distance map',
    #                 'Latitude', 'Longitude',  
    #                'AMSR2 18.7 H', 'AMSR2 18.7 V', 'AMSR2 36.5 H', 'AMSR2 36.5 V', 'Eastward 10m wind',
    #                  'Northward 10m wind', '2m air temperature',  'Total column water vapor', 'Total column cloud liquid water' , 'Month']
    features_name = hparams_data['sar_variables'] + hparams_data['amsr_env_variables'] + ['Month']
    print("features name: ", features_name)


    def agg_segmentation_wrapper(inp):
        model_out = model(inp)
        # Creates binary matrix with 1 for original argmax class for each pixel
        # and 0 otherwise. Note that this may change when the input is ablated
        # so we use the original argmax predicted above, out_max.
        out_max = model_out.argmax(dim=1)
        selected_inds = torch.zeros_like(model_out).scatter_(1, out_max.unsqueeze(1), 1)
         # Ignore areas with label 255
        ignore_mask = (out_max != 255).unsqueeze(1).float()
        selected_inds = selected_inds * ignore_mask
        return (model_out * selected_inds).sum(dim=(2, 3))
    
    class SegmentationWrapper(nn.Module):
            def __init__(self, model):
                super(SegmentationWrapper, self).__init__()
                self.model = model

            def forward(self, inp):
                model_out = self.model(inp)
                out_max = model_out.argmax(dim=1)
                selected_inds = torch.zeros_like(model_out).scatter_(1, out_max.unsqueeze(1), 1)
                ignore_mask = (out_max != 255).unsqueeze(1).float()
                selected_inds = selected_inds * ignore_mask
                return (model_out * selected_inds).sum(dim=(2, 3))
    
    def aggregate_attributions(attributions):

        return attributions.sum(dim=(2, 3)).mean(dim=0).cpu().detach().numpy()

    def clear_cuda():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    # Prepare a sample input from the test_loader
    input_data, _ , = test_loader[0]
    input_data = input_data.to(device)
    

    accumulated_scores = {target: {method: np.zeros(num_input_channels) for method in ['IG', 'FA', 'GS', 'DL']} for target in range(6)}
    # Verify the number of input channels and feature names
    num_input_channels = input_data.shape[1]
    assert len(features_name) == num_input_channels, "Number of features names does not match number of input channels"
    class_names =  {
                    0: 'Open water',
                    1: 'New Ice',
                    2: 'Young ice',
                    3: 'Thin FYI',
                    4: 'Thick FYI',
                    5: 'Old ice' }
    # Initialize the figure for multiple subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()

    ig = IntegratedGradients(agg_segmentation_wrapper)
    feature_ablation = FeatureAblation(agg_segmentation_wrapper)
    gradient_shap = GradientShap(agg_segmentation_wrapper)
    segmentation_wrapper = SegmentationWrapper(model)
    deeplift = DeepLift(segmentation_wrapper)

    num_features = len(features_name)
    num_classes = 6
    aggregated_attributions = {
    'ig': np.zeros((num_classes, num_features)),
    'ablation': np.zeros((num_classes, num_features)),
    'gs_shap': np.zeros((num_classes, num_features)),
    'deeplift': np.zeros((num_classes, num_features))
}

    num_batches = len(test_loader)

    for batch_idx, (input_data, _) in enumerate(test_loader):
        print(f"Processing batch {batch_idx + 1}/{num_batches}")
        if batch_idx == 5:
            break
        
        input_data = input_data.to(device)
        
        for target in range(num_classes):
            # Integrated Gradients
            ig_attr = ig.attribute(input_data, target=target)
            ig_aggregated = aggregate_attributions(ig_attr)
            # norm = np.linalg.norm(ig_aggregated, ord=1)
            # if norm != 0:
            #     ig_aggregated /= norm
            aggregated_attributions['ig'][target] += ig_aggregated

            # Feature Ablation
            ablation_attr = feature_ablation.attribute(input_data, target=target, 
                                                    feature_mask=torch.arange(num_features).view(1, num_features, 1, 1).to(device),
                                                    perturbations_per_eval=1)
            attr_ablation_aggregated = aggregate_attributions(ablation_attr)
            # norm = np.linalg.norm(attr_ablation_aggregated, ord=1)
            # if norm != 0:
            #     attr_ablation_aggregated /= norm
            aggregated_attributions['ablation'][target] += attr_ablation_aggregated

            # Gradient SHAP
            clear_cuda()
            baseline_zeros = torch.zeros_like(input_data)
            baseline_ones = torch.ones_like(input_data)
            baselines = torch.cat([baseline_zeros, baseline_ones], dim=0)
            gs_shap_attr = gradient_shap.attribute(input_data, baselines=baselines, target=target)
            gs_shap_aggregated = aggregate_attributions(gs_shap_attr)
            # norm = np.linalg.norm(gs_shap_aggregated, ord=1)
            # if norm != 0:
            #     gs_shap_aggregated /= norm
            aggregated_attributions['gs_shap'][target] += gs_shap_aggregated

            # DeepLift
            clear_cuda()
            deeplift_attr = deeplift.attribute(input_data, target=target, baselines=baseline_zeros)
            deeplift_aggregated = aggregate_attributions(deeplift_attr)
            # norm = np.linalg.norm(deeplift_aggregated, ord=1)
            # if norm != 0:
            #     deeplift_aggregated /= norm
            aggregated_attributions['deeplift'][target] += deeplift_aggregated

        clear_cuda()

    # Average the attributions
    # for method in aggregated_attributions:
    #     aggregated_attributions[method] /= num_batches

    # Normalize the averaged attributions
    for method in aggregated_attributions:
        for target in range(num_classes):
            norm = np.linalg.norm(aggregated_attributions[method][target], ord=1)
            if norm != 0:
                aggregated_attributions[method][target] /= norm

    # Plotting
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()

    for target in range(num_classes):
        ax = axs[target]
        width = 0.2  # width of the bars
        indices = np.arange(num_features)
        
        ax.barh(indices - 1.5*width, aggregated_attributions['ig'][target], width, label='Integrated Gradient', color='#1f77b4')
        ax.barh(indices - 0.5*width, aggregated_attributions['ablation'][target], width, label='Feature Ablation', color='#ff7f0e')
        ax.barh(indices + 0.5*width, aggregated_attributions['gs_shap'][target], width, label='Gradient SHAP', color='#2ca02c')
        ax.barh(indices + 1.5*width, aggregated_attributions['deeplift'][target], width, label='DeepLift', color='#d62728')
        
        ax.set_yticks(indices)
        ax.set_yticklabels(features_name, fontsize=12)
        ax.set_xlabel('Attribution Score', fontsize=16)
        ax.set_title(f'{class_names[target]}', fontsize=20)
        ax.invert_yaxis()
        ax.set_xlim(-1, 1)
        ax.tick_params(axis='x', labelsize=14)
        ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Adding legend to the figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #ceate a folder for the plots
    if not os.path.exists('captum_plots'):
        os.makedirs('captum_plots')
    plt.savefig('captum_plots/feature_importance_{}.png'.format(model_name))
