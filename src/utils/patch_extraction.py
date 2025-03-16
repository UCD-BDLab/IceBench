import os
import numpy as np
import math
import os
import xarray as xr
import torch
from itertools import product
from utils import *
import math
from season_based_arctic_Seaice import check_season , return_season
from src.data_preprocessing.prepare_data import *

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
    
    def process_scene_season_loc(self , files_list , data_type ,season , location , out_dir):

        if data_type == 'train':
            #files_list = np.random.choice(files_list, 50)
            dir = self.data_options['dir_train_with_icecharts']
            parent_dir = os.path.dirname(dir)
            #out_dir = os.path.join(parent_dir, f"samples_labels_train_{self.task}_{self.patch_size}_{season}_{location}")
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            
            self.data_options['dir_samples_labels_train'] = out_dir


        elif data_type == 'test':
            dir = self.data_options['dir_test_with_icecharts']
            parent_dir = os.path.dirname(dir)
            #out_dir = os.path.join(parent_dir, f"samples_labels_test_{self.task}_{self.patch_size}_{season}_{location}")
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            self.data_options['dir_samples_labels_test'] = out_dir

        elif data_type == 'validation':
            dir = self.data_options['dir_train_with_icecharts']
            parent_dir = os.path.dirname(dir)
            #out_dir = os.path.join(parent_dir, f"samples_labels_val_{self.task}_{self.patch_size}_{season}_{location}")
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            
            self.data_options['dir_samples_labels_val'] = out_dir

            
        #files_list = self.files
        counter = 0

        
        for i in range(len(files_list)):
       
            scene = xr.open_dataset(os.path.join(dir, files_list[i]))

            #extract the scene name befor the .nc extension
            scene_name = files_list[i].split('.')[0]

   
            try:
                samples , labels = self.extract_patches(scene , scene_name )

            except:
                print("No valid patches found for this scene {} ".format(scene_name))
                continue
            
            if samples is not None and labels is not None:
                # Reshape labels to (32, 1)
                if self.task == 'classification':
                    labels = np.reshape(labels, (-1, 1))

                print("there are {} samples in {} scene".format(int(samples.shape[0] ) , scene_name))
                num_batches = samples.shape[0] / self.batch_size
                num_samples = samples.shape[0]
                
                #option to save the samples and labels in batches
                # for j in range(int(num_batches)):
                #     np.save(os.path.join(out_dir, f"{counter}_samples.npy"), samples[j * self.batch_size: (j + 1) * self.batch_size])
                #     np.save(os.path.join(out_dir, f"{counter}_labels.npy"), labels[j * self.batch_size: (j + 1) * self.batch_size])
                #     counter += 1

                # save all the samples and files with counter
                for i in range(num_samples):
                    np.save(os.path.join(out_dir, f"{scene_name}_{counter}_samples.npy"), samples[i])
                    np.save(os.path.join(out_dir, f"{scene_name}_{counter}_labels.npy"), labels[i])
                    counter += 1


            
            scene.close()
            
        print("Done extracting patches from all scenes for {} data".format(data_type))

    def process_files_season_loc (self , season , location , out_dir_parent):
  
        dir = self.data_options['dir_train_with_icecharts']
        #parent_dir = os.path.dirname(dir)
        parent_dir = out_dir_parent
        out_dir = os.path.join(parent_dir, f"samples_labels_train_{self.task}_{self.patch_size}_{season}_{location}")
        if not os.path.exists(out_dir):
            
            self.process_scene_season_loc(self.data_options['train_list'] , 'train' , season , location , out_dir) 
            
        else:
            self.data_options['dir_samples_labels_train'] = out_dir
        

        dir = self.data_options['dir_test_with_icecharts']
        #parent_dir = os.path.dirname(dir)
        parent_dir = out_dir_parent
        out_dir = os.path.join(parent_dir, f"samples_labels_test_{self.task}_{self.patch_size}_{season}_{location}")

        if not os.path.exists(out_dir):
            
            self.process_scene_season_loc(self.data_options['test_list'] , 'test' , season , location , out_dir)
        else:
            self.data_options['dir_samples_labels_test'] = out_dir
     

        dir = self.data_options['dir_train_with_icecharts']
        #parent_dir = os.path.dirname(dir)
        parent_dir = out_dir_parent
        out_dir = os.path.join(parent_dir, f"samples_labels_val_{self.task}_{self.patch_size}_{season}_{location}")
        if not os.path.exists(out_dir):
            
            self.process_scene_season_loc(self.data_options['validation_list'] , 'validation' , season, location , out_dir)
        else:
            self.data_options['dir_samples_labels_val'] = out_dir
      
        return self.data_options
    

    #save the sample and labes with batch size

    def process_scene_with_dir(self , files_list , data_type , out_dir):

        if data_type == 'train':
            #files_list = np.random.choice(files_list, 50)
            dir = self.data_options['dir_train_with_icecharts']
            parent_dir = os.path.dirname(dir)
            #out_dir = os.path.join(parent_dir, f"samples_labels_train_{self.task}_{self.patch_size}")
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            
            self.data_options['dir_samples_labels_train'] = out_dir


        elif data_type == 'test':
            dir = self.data_options['dir_test_with_icecharts']
            parent_dir = os.path.dirname(dir)
            #out_dir = os.path.join(parent_dir, f"samples_labels_test_{self.task}_{self.patch_size}")
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            self.data_options['dir_samples_labels_test'] = out_dir

        elif data_type == 'validation':
            dir = self.data_options['dir_train_with_icecharts']
            parent_dir = os.path.dirname(dir)
            #out_dir = os.path.join(parent_dir, f"samples_labels_val_{self.task}_{self.patch_size}")
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            self.data_options['dir_samples_labels_val'] = out_dir

            
        #files_list = self.files
        counter = 0

        
        for i in range(len(files_list)):
       
            scene = xr.open_dataset(os.path.join(dir, files_list[i]))

            #extract the scene name befor the .nc extension
            scene_name = files_list[i].split('.')[0]

   
            try:
                samples , labels = self.extract_patches(scene , scene_name )

            except:
                print("No valid patches found for this scene {} ".format(scene_name))
                continue
            
            if samples is not None and labels is not None:
                # Reshape labels to (32, 1)
                if self.task == 'classification':
                    labels = np.reshape(labels, (-1, 1))

                print("there are {} samples in {} scene".format(int(samples.shape[0] ) , scene_name))
                num_batches = samples.shape[0] / self.batch_size
                num_samples = samples.shape[0]
                
                #option to save the samples and labels in batches
                # for j in range(int(num_batches)):
                #     np.save(os.path.join(out_dir, f"{counter}_samples.npy"), samples[j * self.batch_size: (j + 1) * self.batch_size])
                #     np.save(os.path.join(out_dir, f"{counter}_labels.npy"), labels[j * self.batch_size: (j + 1) * self.batch_size])
                #     counter += 1

                # save all the samples and files with counter
                for i in range(num_samples):
                    np.save(os.path.join(out_dir, f"{scene_name}_{counter}_samples.npy"), samples[i])
                    np.save(os.path.join(out_dir, f"{scene_name}_{counter}_labels.npy"), labels[i])
                    counter += 1

            # free the memory
            # remove samples labels
            del samples , labels       
            
            scene.close()
            
        print("Done extracting patches from all scenes for {} data".format(data_type))

   

    def process_files_with_dir (self , out_parent_dir):
  
        dir = self.data_options['dir_train_with_icecharts']
        parent_dir = out_parent_dir
        out_dir = os.path.join(parent_dir, f"samples_labels_train_{self.task}_{self.patch_size}")
        if not os.path.exists(out_dir):
            self.process_scene_with_dir(self.data_options['train_list'] , 'train' , out_dir) 
            
        else:
            self.data_options['dir_samples_labels_train'] = out_dir
        

        dir = self.data_options['dir_test_with_icecharts']
        parent_dir = out_parent_dir
        out_dir = os.path.join(parent_dir, f"samples_labels_test_{self.task}_{self.patch_size}")

        if not os.path.exists(out_dir):
            self.process_scene_with_dir(self.data_options['test_list'] , 'test' , out_dir)
        else:
            self.data_options['dir_samples_labels_test'] = out_dir
     

        dir = self.data_options['dir_train_with_icecharts']
        parent_dir = out_parent_dir
        out_dir = os.path.join(parent_dir, f"samples_labels_val_{self.task}_{self.patch_size}")
        if not os.path.exists(out_dir):
            self.process_scene_with_dir(self.data_options['validation_list'] , 'validation' , out_dir)
        else:
            self.data_options['dir_samples_labels_val'] = out_dir
      
        return self.data_options
    


