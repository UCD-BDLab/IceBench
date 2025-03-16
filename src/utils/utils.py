import configparser
import re
import os
import netCDF4
import xarray as xr
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import griddata
from torchvision import transforms
import json


from src.data_preprocessing.Create_label_charts import ice_chart_mapping

def read_config_file(config_file_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)
    return config

def process_settings_data(config):
    paths = config['PATHS']
    data_preprocess = config['data_preprocess']
    #data_augmentation = config['Data_augmentation']
    variables = config['variables']
    ice_chart_config = config['IceChartConfigurations']

    dir_train_without_icecharts = paths['dir_train_without_icecharts']
    dir_test = paths['dir_test']
    out_dir_with_icecharts = paths['dir_train_with_icecharts']
    out_test_with_icecharts = paths['dir_test_with_icecharts']


    data_preprocess_options = eval(config['data_preprocess']['Data_preprocess_options'])
    train_data_options = eval(config['data_preprocess']['Train_data_options'])
    downsample_methods = eval(config['data_preprocess']['Downsample_methods'])
    patch_stride_config = eval(config['data_preprocess']['Patch_stride_config'])
    patch_randomcrop_config = eval(config['data_preprocess']['Patch_randomcrop_config'])
    #data_augmentation_options = eval(config['Data_augmentation']['Data_augmentation_options'])

    # Extract sar_variables and process the complex format
    sar_variables_string = variables['Sar_Variables']
    sar_variables = re.findall(r"'(.*?)'", sar_variables_string)
   
    # Extract amsr_env_variables and process the complex format
    amsr_env_variables_string = variables['Amsr_env_variables']
    amsr_env_variables = re.findall(r"'(.*?)'", amsr_env_variables_string)

    CHARTS = eval(ice_chart_config['CHARTS'])
    train_fill_value = int(ice_chart_config['train_fill_value'])
    polygon_idx = int(ice_chart_config['polygon_idx'])

    SIC_config_dict = eval(ice_chart_config['SIC_Config'])
    SIC_classes = eval(ice_chart_config['SIC_classes'])
    SIC_groups = eval(ice_chart_config['SIC_groups'])

    #SoD
    SOD_config_dict = eval(ice_chart_config['SOD_Config'])
    SOD_classes = eval(ice_chart_config['SOD_classes'])
    SOD_groups = eval(ice_chart_config['SOD_groups'])

    #FLOE
    FLOE_config_dict = eval(ice_chart_config['FLOE_Config'])
    FLOE_classes = eval(ice_chart_config['FLOE_classes'])
    FLOE_groups = eval(ice_chart_config['FLOE_groups'])
    
    #
    ICECHART_NOT_FILLED_VALUE = int(ice_chart_config['ICECHART_NOT_FILLED_VALUE'])
    ICECHART_UNKNOWN = int(ice_chart_config['ICECHART_UNKNOWN'])
    LOOKUP_NAMES = {
        'SIC': SIC_config_dict ,
        'SOD': SOD_config_dict,
        'FLOE': FLOE_config_dict
    }

    ICE_STRINGS = eval(ice_chart_config['ICE_STRINGS'])

    COLOURS = eval(config['Colors']['COLOURS'])
    class_fill_values = { 
        'SIC': SIC_config_dict['mask'],
        'SOD': SOD_config_dict['mask'],
        'FLOE': FLOE_config_dict['mask'] }
    n_classes = {
        'SIC': SIC_config_dict['n_classes'],
        'SOD': SOD_config_dict['n_classes'],
        'FLOE': FLOE_config_dict['n_classes'] }

    return {
        'dir_train_without_icecharts': dir_train_without_icecharts,
        'dir_test': dir_test,
        'dir_train_with_icecharts': out_dir_with_icecharts,
        'dir_test_with_icecharts': out_test_with_icecharts,
        'data_preprocess_options': data_preprocess_options,
        'train_data_options': train_data_options,
        'downsample_methods': downsample_methods, 
        'patch_stride_config': patch_stride_config,
        'patch_randomcrop_config': patch_randomcrop_config,
        #'data_augmentation_options': data_augmentation_options,
        'sar_variables': sar_variables,
        'amsr_env_variables': amsr_env_variables,
        'polygon_idx': polygon_idx,
        'train_fill_value': train_fill_value,
        'charts' : CHARTS,
        'SIC_config': SIC_config_dict,
        'SIC_classes': SIC_classes,
        'SIC_groups': SIC_groups,
        'SOD_config': SOD_config_dict,
        'SOD_classes': SOD_classes,
        'SOD_groups': SOD_groups,
        'FLOE_config': FLOE_config_dict,
        'FLOE_classes': FLOE_classes,
        'FLOE_groups': FLOE_groups,
        'ICECHART_NOT_FILLED_VALUE': ICECHART_NOT_FILLED_VALUE,
        'ICECHART_UNKNOWN': ICECHART_UNKNOWN,
        'ICE_STRINGS': ICE_STRINGS,
        'LOOKUP_NAMES': LOOKUP_NAMES,
        'COLOURS': COLOURS,
        'class_fill_values': class_fill_values ,
        'n_classes': n_classes

    }
    
def read_config_model(config_file_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)
    
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for option in config.options(section):
            config_dict[section][option] = config.get(section, option)
    
    return config_dict


def check_labels_ready(dir_train, settings_data):
    all_files_have_labels = True  # Assume all files have labels initially
    files_with_missing_charts = []
    for scene in os.listdir(dir_train):
        if scene.endswith(".nc"):
            scene_path = os.path.join(dir_train, scene)
            
            nc_file = netCDF4.Dataset(scene_path)
            scene_content = nc_file.variables
            missing_charts = []
            for chart in settings_data["charts"]:
                if chart not in scene_content:
                    missing_charts.append(chart)
            if missing_charts:
                    #print(f"File '{scene}' is missing charts: {', '.join(missing_charts)}")
                all_files_have_labels = False 
                files_with_missing_charts.append((scene, missing_charts))  
                    #break
    return all_files_have_labels, files_with_missing_charts

def calculate_distance_to_border(ds , settings):
    """
    Calculate the distance from each pixel in a polygon ice chart to its nearest border.
    
    Parameters:
        ds (xarray.Dataset): Input dataset containing 'polygon_icechart', 'sar_lines', and 'sar_samples'.
        settings (dict): A dictionary of settings.
        
    Returns:
        xarray.Dataset: The input dataset with the 'distance_to_border' variable added.
    """
    
    data = ds['polygon_icechart'].values

    non_nan_mask = ~np.isnan(data)
    unique_values = np.unique(data[non_nan_mask])

    distances = np.zeros(data.shape)

    for i in unique_values:
        if not np.isnan(i):
            
            temp_arr = np.zeros(data.shape)
            temp_arr[data == i] = 1
            temp_arr[data != i] = 0
            temp_arr = distance_transform_edt(temp_arr)
            distances[temp_arr != 0] = temp_arr[temp_arr != 0]
    
    distances[np.isnan(ds['polygon_icechart'].values)] = np.nan
    ds = ds.assign({'distance_to_border': xr.DataArray(distances , dims=ds['polygon_icechart'].dims)}) 
    #ds['distance_to_border'] = xr.DataArray(distances , dims = ['sar_lines' , 'sar_samples'])
    

    return ds

def reshape_grid_data(ds  , variable_name , output_name):
    '''
    Reshape the sar incidence angle to the same shape as other variables.
    Parameters:
        ds (xarray.Dataset): Input dataset containing 'sar_grid_incidenceangle', 'sar_grid_line', 'sar_grid_sample', 'sar_lines', and 'sar_samples'.

    Returns:
        xarray.Dataset: The input dataset with the 'sar_incidenceangle' variable added.
    '''


    sar_grid_incidence = ds[variable_name].values
    sar_grid_line = ds['sar_grid_line'].values
    sar_grid_sample = ds['sar_grid_sample'].values

    sar_lines = len(ds['sar_lines'])
    sar_samples = len(ds['sar_samples'])

    # Create a 2D grid of indices
    index_grid = np.meshgrid(np.arange(sar_lines), np.arange(sar_samples), indexing='ij')
    index_grid = np.stack(index_grid, axis=-1)

    # Interpolate the incidence data onto the new grid
    sar_incidence_reshape = griddata(
        np.column_stack((sar_grid_line, sar_grid_sample)),
        sar_grid_incidence,
        index_grid,
        method='linear'
    )

    ds = ds.assign({output_name: xr.DataArray(sar_incidence_reshape, dims=('sar_lines', 'sar_samples'))})

    return ds


def update_sar_based_on_polygonicechart(ds , settings):
    """
    Find the nan values in the polygon ice chart and find those pixels in sar variables and make them nan.
    
    Parameters:
        ds (xarray.Dataset): Input dataset containing 'polygon_icechart' and sar variables in settings
        settings (dict): A dictionary of settings.
        
    Returns:
        xarray.Dataset: The updated sar variables in input dataset
    """
    sar_variables = settings['sar_variables']

    variables_to_remove = [ 'distance_map', 'sar_latitude' , 'sar_longitude']
    sar_variables = [var for var in sar_variables if var not in variables_to_remove]
   
    polygon_icechart = ds['polygon_icechart'].values
    # Find nan values in the ice chart
    icechart_mask = np.isnan(polygon_icechart)
    
    # Iterate through SAR variables and update pixels based on the ice chart nan mask
    for var_name in sar_variables:
        if var_name in ds:
            
            sar_variable = ds[var_name].values
            sar_variable[icechart_mask] = np.nan
            #sar_variable[np.isnan(sar_variable)] = 255

            ds[var_name].values = sar_variable
    
    return ds


def decimal_to_scaled_value(ds , settings):
    """
    Convert SAR variables from decimal to scaled value.
    
    Parameters:
        ds (xarray.Dataset): Input dataset containing SAR variables.
        settings (dict): A dictionary of settings.
        
    Returns:
        xarray.Dataset: The updated SAR variables in input dataset
    """
    sar_variables = settings['sar_variables']

    # Variables to remove
    variables_to_remove = ['sar_incidenceangle', 'distance_map',     'sar_latitude', 'sar_longitude']

    # Remove the specified variables from sar_variables
    sar_variables = [var for var in sar_variables if var not in variables_to_remove]

    # Iterate through SAR variables and update pixels based on the ice chart nan mask
    for var_name in sar_variables:
        if var_name in ds:
          
            # Add 10 to each pixel
            ds[var_name] = ds[var_name] + 10
            
            # Divide each pixel by 20
            ds[var_name] = ds[var_name] / 20
    
    return ds

def normalize_channels(ds , settings):
    """
    Normalize SAR variables.
    
    Parameters:
        ds (xarray.Dataset): Input dataset containing SAR variables.
        settings (dict): A dictionary of settings.
        
    Returns:
        xarray.Dataset: The updated SAR variables in input dataset
    """
    variables = settings['sar_variables'] + settings['amsr_env_variables']
    # remove_variable = ['distance_map', 'sar_latitude' , 'sar_longitude']
    # variables = [var for var in variables if var not in remove_variable]
    mean_std_dict = settings['mean_std_dict']
    for var_name in variables:
        if var_name in ds:
            ds[var_name] = (ds[var_name] - mean_std_dict[var_name]['mean']) / mean_std_dict[var_name]['std']
            # put 0 for nan values
            ds[var_name] = ds[var_name].fillna(0)


    return ds


def process_nc_files_and_generate_maps(directory, out_path, settings):
    """
    Process NetCDF files in the given directory, calculate distance to border, generate ice chart mapping scenes,
    and save the scenes as NetCDF files in the output path.

    Args:
        directory (str): Path to the directory containing NetCDF files.
        out_path (str): Path to the directory where output NetCDF files will be saved.
        settings (dict): Dictionary containing settings for distance calculation and ice chart mapping.

    Returns:
        None
    """
    compression_settings = {
    'nersc_sar_primary': {'zlib': True, 'complevel': 1},
    'nersc_sar_secondary': {'zlib': True, 'complevel': 1},
    'sar_incidenceangle': {'zlib': True, 'complevel': 4},
    'sar_latitude': {'zlib': True, 'complevel': 4},
    'sar_longitude': {'zlib': True, 'complevel': 4},
    'distance_map': {'zlib': True, 'complevel': 4},
    }
    for filename in os.listdir(directory):
        
        if filename.endswith(".nc"):
            file_path = os.path.join(directory, filename)
            try:
                ds = xr.open_dataset(file_path)
                ds = reshape_grid_data(ds , 'sar_grid_incidenceangle' , 'sar_incidenceangle')
                ds = reshape_grid_data(ds , 'sar_grid_latitude' , 'sar_latitude')
                ds = reshape_grid_data(ds , 'sar_grid_longitude' , 'sar_longitude')
                
                ds = update_sar_based_on_polygonicechart(ds , settings)
                
                #ds = decimal_to_scaled_value(ds , settings)
                ds = normalize_channels(ds , settings)
                ds_with_distance_to_border  = calculate_distance_to_border(ds , settings)
              
                scene = ice_chart_mapping(ds_with_distance_to_border , settings)
                
                
                output_filename = filename
                # if the out_path is not exist, create it
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                output_path_file = os.path.join(out_path, output_filename) 
                scene = scene.drop_vars(['polygon_icechart','sar_grid_line', 'sar_grid_sample', 'sar_grid_height','sar_grid_incidenceangle' , 'amsr2_swath_map' , 'swath_segmentation',])
                #scene.to_netcdf(output_path_file)
                # Convert specific variables to the desired data types
                scene['SIC'] = scene['SIC'].astype('uint8')
                scene['SOD'] = scene['SOD'].astype('uint8')
                scene['FLOE'] = scene['FLOE'].astype('uint8')

                # Convert the rest of the variables to float32
                for var_name in scene.data_vars:
                    if var_name not in ['SIC', 'SOD', 'FLOE']:
                        scene[var_name] = scene[var_name].astype('float32')

                scene.to_netcdf(output_path_file, format='NETCDF4', encoding=compression_settings)
                print(f"Processed {filename}")
                del scene 
                ds.close()
            except Exception as e:
                print(f"Error processing {filename}: {e}")


def downsample_variable(ds, variable , factor , method):
    '''
    Downsample a variable in a dataset using a specified factor and method.
    
    Parameters:
        ds (xr.Dataset): Input dataset.
        variable (str): Variable name to downsample.
        factor (int): Downsampling factor.
        method (str): Downsampling method ('mean' or 'max').
        
    Returns:
        xr.Dataset: Downsampled dataset.
    '''
    if factor <= 0:
        raise ValueError("Downsampling factor must be greater than 0.")
    variable = ds[variable]
    if method == 'mean':
        var_downsampled = variable.coarsen(sar_lines=factor, sar_samples=factor , boundary='trim' , side= 'left').mean()
    elif method == 'max':
        var_downsampled = variable.coarsen(sar_lines=factor, sar_samples=factor , boundary='trim' , side= 'left').max()
    elif method == 'min':
        var_downsampled = variable.coarsen(sar_lines=factor, sar_samples=factor , boundary='trim' , side= 'left').min()
    else:
        raise ValueError(f"Downsampling method '{method}' not recognised.")
    return var_downsampled


def downsampling_files (settings_data):
    """
        Downsample specified variables in NetCDF files and save the downsampled dataset to new files.
        
        Parameters:
            settings_data (dict): A dictionary containing settings and options for data preprocessing.
            
        Returns:
            dict: The updated settings_data dictionary with the 'dir_train_with_icecharts_downsampled' key added.
    """
    
    dir_data_icecharts = settings_data['dir_train_with_icecharts']
    data_preprocess_options = settings_data['data_preprocess_options']
    factor = data_preprocess_options['downsampling_factor']
    downsample_methods = settings_data['downsample_methods']
    amsr_env_variables = settings_data['amsr_env_variables']
    pixel_space = data_preprocess_options['original_pixel_spacing']

    for filename in os.listdir(dir_data_icecharts):
        
        if filename.endswith(".nc"):
            file_path = os.path.join(dir_data_icecharts, filename)
            try:
                ds = xr.open_dataset(file_path )
                ds_downsampled = xr.Dataset()
                
                for variable in ds.data_vars:
                    if variable in downsample_methods.keys():
                    
                        method = downsample_methods[variable]
                        ds_downsampled[variable] = downsample_variable(ds, variable , factor , method)
                         
                ds.close()
                parent_directory = os.path.dirname(dir_data_icecharts)
                
                dir_data_downsampled = os.path.join(parent_directory , f"out_dir_downsampled_{factor}")
                if not os.path.exists(dir_data_downsampled):
                    os.makedirs(dir_data_downsampled)
                
                
                settings_data['dir_train_with_icecharts_downsampled'] = dir_data_downsampled
                out_path = dir_data_downsampled
                output_filename = filename#[:-3] + f"_downsampled_{factor}.nc"
                output_path = os.path.join(out_path, output_filename)   
                
                for variable_ in amsr_env_variables:
                    ds_downsampled[variable_] = ds[variable_]
                # remove some variables from downsampled files

                ds_downsampled.attrs = ds.attrs
                ds_downsampled.attrs['downsampling_factor'] = factor
                ds_downsampled.attrs['pixel_spacing'] = factor * pixel_space 
                ds_downsampled.to_netcdf(output_path)
                print(f"Downsampled {filename}")
            except Exception as e:
                print(f"Error downsampling {filename}: {e}")
    
    settings_data['dir_train_with_icecharts'] = dir_data_downsampled
    return settings_data

def downsampling_files_mode (settings_data, mode="train"):
    """
        Downsample specified variables in NetCDF files and save the downsampled dataset to new files.
        
        Parameters:
            settings_data (dict): A dictionary containing settings and options for data preprocessing.
            mode (str): The mode to use, either "train" or "test".
            
        Returns:
            dict: The updated settings_data dictionary with the 'dir_train_with_icecharts_downsampled' key added.
    """
    
    dir_data_icecharts = settings_data['dir_{}_with_icecharts'.format(mode)]
    data_preprocess_options = settings_data['data_preprocess_options']
    factor = data_preprocess_options['downsampling_factor']
    downsample_methods = settings_data['downsample_methods']
    amsr_env_variables = settings_data['amsr_env_variables']
    pixel_space = data_preprocess_options['original_pixel_spacing']

    for filename in os.listdir(dir_data_icecharts):
        
        if filename.endswith(".nc"):
            file_path = os.path.join(dir_data_icecharts, filename)
            try:
                ds = xr.open_dataset(file_path )
                ds_downsampled = xr.Dataset()
                
                for variable in ds.data_vars:
                    if variable in downsample_methods.keys():
                    
                        method = downsample_methods[variable]
                        ds_downsampled[variable] = downsample_variable(ds, variable , factor , method)
                         
                ds.close()
                parent_directory = os.path.dirname(dir_data_icecharts)
                
                dir_data_downsampled = os.path.join(parent_directory , f"out_dir_downsampled_{factor}_{mode}")
                if not os.path.exists(dir_data_downsampled):
                    os.makedirs(dir_data_downsampled)
                
                
                settings_data['dir_{}_with_icecharts_downsampled'.format(mode)] = dir_data_downsampled
                out_path = dir_data_downsampled
                output_filename = filename#[:-3] + f"_downsampled_{factor}.nc"
                output_path = os.path.join(out_path, output_filename)   
                
                for variable_ in amsr_env_variables:
                    ds_downsampled[variable_] = ds[variable_]

                ds_downsampled.attrs = ds.attrs
                ds_downsampled.attrs['downsampling_factor'] = factor
                ds_downsampled.attrs['pixel_spacing'] = factor * pixel_space 
                ds_downsampled.to_netcdf(output_path)
                print(f"Downsampled {filename}")
            except Exception as e:
                print(f"Error downsampling {filename}: {e}")
    
    settings_data['dir_{}_with_icecharts'.format(mode)] = dir_data_downsampled
    return settings_data

def get_transform_list(settings_model):
    transform_list = []
    data_augmentation_options = settings_model['Data_augmentation_options']
    rotation = data_augmentation_options['rotation'].strip("'")
    rotation_angle = int(data_augmentation_options['rotation_angle'])
    flip = data_augmentation_options['flip'].strip("'")
    flip_axis = int(data_augmentation_options['flip_axis'])
    if rotation:
        transform_list.append(transforms.RandomRotation(degrees=rotation_angle))
    if flip:
        if flip_axis == 0:
            transform_list.append(transforms.RandomVerticalFlip())
        elif flip_axis == 1:
            transform_list.append(transforms.RandomHorizontalFlip())
    
    return transform_list


def get_variable_options(settings_data , settings_model):
    """
    Get amsr and env grid options, crop shape and upsampling shape.

    Parameters
    ----------
    settings_data: dict
        Dictionary with training options.
    
    Returns
    -------
    settings_data: dict
        Updated with amsrenv options.
    """
    if settings_data['data_preprocess_options']['downsampling'] :
        new_pixel_spacing = int(settings_data['data_preprocess_options']['original_pixel_spacing']) * int(settings_data['data_preprocess_options']['downsampling_factor'])
    else:
        new_pixel_spacing = int(settings_data['data_preprocess_options']['original_pixel_spacing'])
    settings_data['amsrenv_delta'] = 50 / (new_pixel_spacing // 40)
    settings_data['amsrenv_patch'] = settings_data['data_preprocess_options']['patch_size'] / settings_data['amsrenv_delta']
    settings_data['amsrenv_patch_dec'] = int(settings_data['amsrenv_patch'] - int(settings_data['amsrenv_patch']))
    settings_data['amsrenv_upsample_shape'] = (int(settings_data['data_preprocess_options']['patch_size']  + \
                                                   settings_data['amsrenv_patch_dec'] * \
                                                   settings_data['amsrenv_delta']),
                                               int(settings_data['data_preprocess_options']['patch_size'] +  \
                                                   settings_data['amsrenv_patch_dec'] * \
                                                   settings_data['amsrenv_delta']))
   
    
    return settings_data

def files(options_data ):
    train_dir_list = 'datalist/train_list.json'
    test_dir_list = 'datalist/test_list.json'
    save_filenames_to_jason(options_data['dir_train_without_icecharts'] , train_dir_list)
    save_filenames_to_jason(options_data['dir_test'] , test_dir_list)

def save_filenames_to_jason(dir_path , jason_file_path):
    """
    Read the test files and return the list of files.
    Parameters
    ----------
    dir_test: str
        Path to the directory containing test files.
    
    Returns
    -------
        None
    """
    list_files = []
 
    for filename in os.listdir(dir_path):
        if filename.endswith(".nc"):
            file_path = os.path.join(dir_path, filename)
            list_files.append(file_path)

    # with open(jason_file_path, 'w') as f:
    #     json.dump(list_files, f)
    json_data = json.dumps(list_files, indent=4)
    try:
        with open(jason_file_path, 'w') as json_file:
            json_file.write(json_data)
        print("JSON data written to 'data.json'")
    except Exception as e:
        print("An error occurred:", e)





def compute_channel_stats(setting_data , files):
    channel_means = []
    channel_stds = []
        

    for file in files:
        scene = xr.open_dataset(os.path.join(setting_data['dir_train_with_icecharts'], file))

            # Compute mean and std for each channel in this scene
        for variable in setting_data['full_variables']:
            channel_data = scene[variable].values
            mean = np.mean(channel_data)
            std = np.std(channel_data)

            channel_means.append(mean)
            channel_stds.append(std)

        # Compute the overall mean and std across all scenes
    overall_mean = np.mean(channel_means)
    overall_std = np.mean(channel_stds)

    return overall_mean, overall_std

def get_month(filename):
    date = os.path.basename(filename).split('_')[5].split('.')[0].split('T')[0]
    month = date[4:6]
    
    return month
def get_month_files(files_list , season):
    if season == 'winter':
        winter_files = [f for f in files_list if get_month(f) in [ '12', '01', '02']]
        return winter_files
    elif season == 'summer':
        summer_files = [f for f in files_list if get_month(f) in ['06', '07', '08']]
        return summer_files
    elif season == 'spring':
        spring_files = [f for f in files_list if get_month(f) in ['03', '04', '05']]
        return spring_files
    elif season == 'fall':
        fall_files = [f for f in files_list if get_month(f) in ['09', '10', '11']]
        return fall_files
def get_season(file_name):
    month = get_month(file_name)
    if month in ['12', '01', '02']:
        return 'winter'
    elif month in ['06', '07', '08']:
        return 'summer'
    elif month in ['03', '04', '05']:
        return 'spring'
    elif month in ['09', '10', '11']:
        return 'fall'
    return None

def is_season(filename, season):
    month = get_month(filename)
    
    season_months = {
        'winter': ['12', '01', '02'],
        'spring': ['03', '04', '05'],
        'summer': ['06', '07', '08'],
        'fall': ['09', '10', '11'],
        'fourseasons': ['12', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11'],
    }

    # Check if the month extracted from the filename is in the list of months for the specified season
    if month in season_months.get(season, []):
        return True
    return False

def location(filename):
    provider = os.path.basename(filename).split('_')[10]
    if provider =='dmi':
        location = os.path.basename(filename).split('_')[12]
    else:
        location = os.path.basename(filename).split('_')[11]
    return location

def get_location(file_name):
    locations_dic = {
    'cat1' : ['CapeFarewell', 'CentralEast','NorthAndCentralEast', 'NorthEast','SouthEast','SGRDIWA'] ,
    'cat2' :  ['CentralWest', 'NorthWest' , 'Qaanaaq','SGRDIFOXE', 'SouthWest'] ,
    'cat3' : ['SGRDIEA',  'SGRDIHA', 'SGRDIMID', 'SGRDINFLD']   ,
    'cat4' : ['North']
    }
    provider = os.path.basename(file_name).split('_')[10]
    if provider =='dmi':
        location = os.path.basename(file_name).split('_')[12]
    else:
        location = os.path.basename(file_name).split('_')[11]

    for key, value in locations_dic.items():
        if location in value:
            return key
    return None
    
def read_filenames_from_json(file_path):
    """Read filenames from a JSON file and return them as a list."""
    with open(file_path, 'r') as file:
        filenames = json.load(file)
    return filenames