import os
import numpy as np
import math
import torch
import os
import numpy as np
import xarray as xr
import torch
from itertools import product


class PatchExtractor:
    """
    PatchExtractor is responsible for extracting patches from spatial data for training, validation, and testing.

    This class provides methods to process and filter spatial data based on predefined rules and configurations,
    ensuring that valid patches are extracted for use in classification or segmentation tasks.

    Attributes:
    - data_options (dict): Configuration options related to data preprocessing and patch extraction.
    - model_options (dict): Configuration options for the model, such as batch size and mode.
    - chart_label (str): The type of label used in the ice charts (e.g., 'SIC', 'SOD', 'FLOE').
    - patch_size (int): Size of the square patch to be extracted.
    - batch_size (int): Batch size for saving patches.
    - stride (int): Step size between patches.
    - padding (int): Amount of padding applied during extraction.
    - task (str): Task type (e.g., 'classification' or 'segmentation').
    - samples (dict): A dictionary to store extracted samples for each scene.
    """

    def __init__(self, data_options, model_options, chart_label):
        # Initialize configuration parameters
        self.data_options = data_options
        self.patch_size = int(data_options['data_preprocess_options']['patch_size'])
        self.batch_size = int(model_options['train']['batch_size'])
        self.stride = int(data_options['patch_stride_config']['stride'])
        self.padding = int(data_options['patch_stride_config']['padding'])
        self.task = model_options['model']['mode'].strip("'")
        self.chart = chart_label
        self.data_options['full_variables'] = data_options['sar_variables']
        self.samples = {}

    def extract_month_number(self, file_name):
        """
        Extracts the month number from a given filename.

        Args:
            file_name (str): The input filename from which the month number is to be extracted.

        Returns:
            int or None: The extracted month number (1 to 12) if valid; None if the format is not recognized.
        """
        try:
            if file_name.startswith('S1'):
                month_number = int(file_name[21:23])  # Extract month number from specific indices
            elif file_name.endswith('_dmi_prep.nc'):
                month_number = int(file_name[4:6])
            else:
                month_number = int(file_name[20:28][4:6])  # Generic month extraction for other formats

            if 1 <= month_number <= 12:
                return month_number
            return None  # Invalid month number
        except IndexError:
            return None  # Filename does not match expected format

    def check_patch_conditions(self, scene, row_rand, col_rand, threshold=0.3):
        """
        Checks if a given patch in the scene satisfies specified conditions.

        Args:
            scene (xarray.Dataset): The scene containing various data layers.
            row_rand (int): Row coordinate of the patch's starting point.
            col_rand (int): Column coordinate of the patch's starting point.
            threshold (float): Threshold for valid pixels in the patch.

        Returns:
            bool: True if all conditions are met, False otherwise.
        """
        sample_y = scene[self.chart].values[row_rand: row_rand + self.patch_size, col_rand: col_rand + self.patch_size]
        patch_x = scene['nersc_sar_primary'].values[row_rand: row_rand + self.patch_size, col_rand: col_rand + self.patch_size]
        
        # Check for valid conditions in the patch
        num_nan = np.sum(np.isnan(patch_x))
        condition_patch_x = num_nan == 0
        condition_patch_y = np.sum(sample_y == self.data_options['class_fill_values'][self.chart]) / sample_y.size <= 0.3
        condition_valid_pixels = condition_patch_x and condition_patch_y

        # Additional conditions based on distance to border and land masking
        distance_to_border_enabled = self.data_options['data_preprocess_options']['distance_to_border']
        land_mask_enabled = self.data_options['data_preprocess_options']['land_masking']
        final_condition = condition_valid_pixels

        if distance_to_border_enabled or land_mask_enabled:
            patch_distance_to_border = scene['distance_to_border'].values[row_rand: row_rand + self.patch_size, col_rand: col_rand + self.patch_size]
            patch_dist_map = scene['distance_map'].values[row_rand: row_rand + self.patch_size, col_rand: col_rand + self.patch_size]

            if distance_to_border_enabled:
                all_distance_border = np.all(patch_distance_to_border > self.data_options['data_preprocess_options']['distance_border_threshold'])
                condition_distance = all_distance_border
                final_condition = final_condition and condition_distance

            if land_mask_enabled:
                condition_land = np.all(patch_dist_map > 0)
                final_condition = final_condition and condition_land

        return final_condition

    def check_label(self, scene, row, col, threshold):
        """
        Validate a patch's label based on criteria such as fill value and pixel consistency.

        Args:
            scene (xarray.Dataset): The scene containing label data.
            row (int): Row index of the patch's starting point.
            col (int): Column index of the patch's starting point.
            threshold (float): Threshold for valid labels.

        Returns:
            (bool, int/array): Tuple of validation status and the patch's label.
        """
        not_filled = self.data_options['class_fill_values'][self.chart]
        y_patch = scene[self.chart].values[row: row + self.patch_size, col: col + self.patch_size]
        npsum = np.sum(y_patch == not_filled)
        condition_valid_pixels = npsum / y_patch.size <= threshold

        if self.task == 'classification':
            if np.all(y_patch == y_patch[0][0]) and npsum == 0:
                return True, y_patch[0][0]
            return False, not_filled

        elif self.task == 'segmentation':
            return condition_valid_pixels, y_patch if condition_valid_pixels else not_filled

        return False, not_filled

    def extract_patches(self, scene, scene_name):
        """
        Extract patches from a given scene based on predefined configurations.

        Args:
            scene (xarray.Dataset): The input scene data.
            scene_name (str): The name of the scene being processed.

        Returns:
            (np.array, np.array): A tuple of extracted patches and their corresponding labels.
        """
        nrows, ncols = len(scene['sar_lines']), len(scene['sar_samples'])
        row_indices = range(0, nrows - self.patch_size, self.stride)
        col_indices = range(0, ncols - self.patch_size, self.stride)

        # Initialize arrays to store extracted patches
        samples_list, labels_list = [], []
        patch = np.zeros((len(self.data_options['full_variables']) + len(self.data_options['amsr_env_variables']) + 1, self.patch_size, self.patch_size))  # +1 for month
        month = self.extract_month_number(scene_name)
        threshold = 0.3

        # Binary conversion for 'SIC' label if configured
        if self.chart == 'SIC' and self.data_options['SIC_config']['binary_label']:
            scene, threshold = self.apply_threshold_to_binary(scene)

        # Iterate through patches in the scene and apply conditions
        for row, col in product(row_indices, col_indices):
            patch_valid = self.check_patch_conditions(scene, row, col, threshold)
            label_valid, label = self.check_label(scene, row, col, threshold)

            if patch_valid and label_valid:
                # Populate the patch array with valid samples
                patch[:-1, :, :] = scene[self.data_options['full_variables']].isel(
                    sar_lines=slice(row, row + self.patch_size), sar_samples=slice(col, col + self.patch_size)
                ).to_array().values

                month_array = np.full((self.patch_size, self.patch_size), month)
                patch[-1, :, :] = month_array
                samples_list.append(patch)
                labels_list.append(label)

        if not samples_list:
            print(f"⚠️ No valid patches found for scene '{scene_name}'.")
            return None, None

        samples = np.stack(samples_list, axis=0)
        labels = np.stack(labels_list, axis=0)
        return samples, labels

def generate_patches_for_season_loc(hparams_data , hparams_model, season, loc, out_dir):
    """
    Generate training, validation, and test patches for a specified season and location.

    """
    # Set global statistics file for normalization
    hparams_data['meanstd_file'] = 'global_mean_std.npy'
    hparams_data['mean_std_dict'] = np.load(hparams_data['meanstd_file'], allow_pickle=True).item()
    
    # Preprocess data to upsample AMSR variables if applicable
    if len(hparams_data['amsr_env_variables']) > 0:
        print(" Upsampling AMSR environmental variables for enhanced resolution...")
        hparams_data = get_variable_options(hparams_data, hparams_model)
    print(" Data preprocessing for training and testing completed.")

    # Data Loader Configuration
    np.random.seed(int(hparams_model['datamodule']['seed']))
    
    # Filter files for training by season and location
    all_files_train = [f for f in os.listdir(hparams_data['dir_train_with_icecharts']) if f.endswith(".nc")]
    print(f"Total available training files: {len(all_files_train)}")
    
    # Set season and location settings for filtering
    hparams_data['train_data_options']['location'] = [loc]
    hparams_data['train_data_options']['season'] = season
    seasona = hparams_data['train_data_options']['season'].strip("'")
    category_location = hparams_data['train_data_options']['location'][0]

    # Filter the files based on specified season and location
    specific_locations = hparams_data['train_data_options'][category_location]
    season_loc_files = [f for f in all_files_train if location(f) in specific_locations and is_season(f, seasona)]
    print(f" For season '{seasona}' and location category '{category_location}', filtered training files: {len(season_loc_files)}")

    # Generate a validation list from the filtered files
    val_num = math.ceil(len(season_loc_files) * 0.1)  # 10% validation split
    validation_list = np.random.choice(season_loc_files, val_num, replace=False).tolist()
    print(f"Validation list for season '{seasona}' and location '{category_location}': {len(validation_list)} samples.")

    # Create a training list by excluding validation samples
    train_list = [x for x in season_loc_files if x not in validation_list]
    print(f" Training list for season '{seasona}': {len(train_list)} samples.")

    # Prepare test files by filtering based on location and season
    all_test_files = os.listdir(hparams_data['dir_test_with_icecharts'])
    season_loc_test = [f for f in all_test_files if location(f) in specific_locations and is_season(f, seasona)]
    test_list = season_loc_test
    print(f"Test list for season '{seasona}' and location '{category_location}': {len(test_list)} samples.")

    # Update hparams_data with the new train, validation, and test lists
    hparams_data['train_list'] = train_list
    hparams_data['validation_list'] = validation_list
    hparams_data['test_list'] = test_list

    # Print data splits for reference
    print(f"Data Split Summary:\n - Training: {len(train_list)} files\n - Validation: {len(validation_list)} files\n - Test: {len(test_list)} files")


    # Patch Extraction Logic
    if patch_with_stride:
        print(f"Initiating patch extraction with stride for season '{season}' and location '{loc}'...")
        extractor = PatchExtractor(hparams_data, hparams_model, hparams_model['model']['label'].strip("'"))
        hparams_data = extractor.process_files_season_loc(season, loc, out_dir)
        print(f" Patch extraction completed for season '{season}' and location '{loc}'. Patches saved in: {out_dir}")
        return hparams_data
    else:
        print(f" Patch extraction skipped as 'patch_with_stride' is set to False.")
