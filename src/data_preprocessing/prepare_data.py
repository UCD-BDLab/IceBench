import os
from src.utils.utils import (downsampling_files, process_nc_files_and_generate_maps, downsampling_files_mode,
                   get_variable_options)

def generate_ice_charts_if_empty(directory, directory_with_icechart, hparams_data):
    """
    Generate ice charts if the specified directory is empty.

    Args:
        directory (str): Path to the input directory containing NC files.
        directory_with_icechart (str): Path to the output directory for ice charts.
        hparams_data (dict): Hyperparameters for data processing.

    Returns:
        None
    """
    if not os.path.exists(directory_with_icechart) or not os.listdir(directory_with_icechart):
        print(f"{directory_with_icechart} is empty. Creating ice charts for the scenes.")
        process_nc_files_and_generate_maps(directory, directory_with_icechart, hparams_data)
    else:
        print(f"{directory_with_icechart} is not empty. No need to create ice charts for the scenes.")

def prepare_data(hparams_data, hparams_model):
    """
    Prepare data for training and testing, including ice chart generation and downsampling.

    Args:
        hparams_data (dict): Hyperparameters for data processing.
        hparams_model (dict): Hyperparameters for the model.

    Returns:
        dict: Updated hparams_data with potentially modified directories.
    """
    # Extract directories from hyperparameters
    dir_train = hparams_data['dir_train_without_icecharts']
    dir_train_with_icecharts = hparams_data['dir_train_with_icecharts']
    dir_test = hparams_data['dir_test']
    dir_test_with_icecharts = hparams_data['dir_test_with_icecharts']
    mode = hparams_model['model']['mode']

    # Generate ice charts if necessary
    generate_ice_charts_if_empty(dir_train, dir_train_with_icecharts, hparams_data)
    generate_ice_charts_if_empty(dir_test, dir_test_with_icecharts, hparams_data)

    # Downsampling data if specified in hyperparameters
    downsampling_condition = hparams_data['data_preprocess_options']['downsampling']
    factor = hparams_data['data_preprocess_options']['downsampling_factor']
    parent_directory = os.path.dirname(dir_train_with_icecharts)
    out_downsample_dir_train = os.path.join(parent_directory, f"out_dir_downsampled_{factor}_train")
    out_downsample_dir_test = os.path.join(parent_directory, f"out_dir_downsampled_{factor}_test")

    if downsampling_condition and factor > 0:
        if not os.path.exists(out_downsample_dir_train):
            print("Downsampling train data...")
            hparams_data = downsampling_files_mode(hparams_data, mode='train')
            hparams_data['dir_train_with_icecharts'] = out_downsample_dir_train

        if not os.path.exists(out_downsample_dir_test):
            print("Downsampling test data...")
            hparams_data = downsampling_files_mode(hparams_data, mode='test')
            hparams_data['dir_test_with_icecharts'] = out_downsample_dir_test

    # Process AMSR environmental variables if specified
    if hparams_data['amsr_env_variables']:
        print("Upsampling AMSR environmental variables...")
        hparams_data = get_variable_options(hparams_data, hparams_model)

    return hparams_data
