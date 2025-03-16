import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BenchmarkDataset_directory(Dataset):
    def __init__(self , data_options, model_options ,data_dir, transform_list=None):
        self.data_options = data_options
        self.model_options = model_options
        self.data_dir = data_dir
        self.sample_files = [file for file in os.listdir(self.data_dir) if file.endswith("_samples.npy")]
        self.label_files = [file.replace("_samples.npy", "_labels.npy") for file in self.sample_files]
        
        # Ensure that transform_list is not None
        if transform_list is None:
            transform_list = []
       
        self.transform = transforms.Compose(transform_list)


    def __len__(self):
       
        return len(self.sample_files)

    def __getitem__(self, idx):
        
        sample_file = os.path.join(self.data_dir, self.sample_files[idx])
        label_file = os.path.join(self.data_dir, self.label_files[idx])
        sample = np.load(sample_file)
        label = np.load(label_file)
        sample = torch.from_numpy(sample).float()
        label = torch.from_numpy(label).long()
        #sample[-1, :,:] = (sample[-1, :, :] - 6.5) / 3.5

        if self.transform:
            sample = self.transform(sample)
            

        return sample, label

class BenchmarkTestDataset_directory(Dataset):
    def __init__(self , data_options, model_options ,data_dir ):
        self.data_options = data_options
        self.model_options = model_options
        self.data_dir = data_dir
        self.sample_files = [file for file in os.listdir(self.data_dir) if file.endswith("_samples.npy")]
        self.label_files = [file.replace("_samples.npy", "_labels.npy") for file in self.sample_files]
        self.batch_size = int(self.model_options['train']['batch_size'])
       
        
    def __len__(self):
      
        return len(self.sample_files) 
    
    def __getitem__(self, idx):
        sample_file = self.sample_files[idx]
        label_file = self.label_files[idx]
        sample = np.load(os.path.join(self.data_dir, sample_file))
        label = np.load(os.path.join(self.data_dir, label_file))
        sample = torch.from_numpy(sample).float()
        label = torch.from_numpy(label).long()
        return sample, label


class BenchmarkDataset_datasize_directory(Dataset):
    def __init__(self, data_options, model_options, data_dir, selected_sample_files, selected_label_files, transform_list=None):
            self.sample_files = selected_sample_files
            self.label_files = selected_label_files
            self.data_dir = data_dir
            self.data_options = data_options
            self.model_options = model_options
            # Ensure that transform_list is not None
            if transform_list is None:
                transform_list = []
            self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
            sample_file = os.path.join(self.data_dir, self.sample_files[idx])
            label_file = os.path.join(self.data_dir, self.label_files[idx])

            sample = np.load(sample_file)
            label = np.load(label_file)

            sample = torch.from_numpy(sample).float()
            label = torch.from_numpy(label).long()

            if self.transform:
                sample = self.transform(sample)

            return sample, label

class BenchmarkTestDataset_mask_directory:
    def __init__(self , data_options, model_options ,data_dir ):
        self.data_options = data_options
        self.model_options = model_options
        self.data_dir = data_dir
        self.sample_files = [file for file in os.listdir(self.data_dir) if file.endswith("_samples.npy")]
        self.label_files = [file.replace("_samples.npy", "_labels.npy") for file in self.sample_files]
        self.mask_files = [file.replace("_samples.npy", "_masks.npy") for file in self.sample_files]
        self.batch_size = int(self.model_options['train']['batch_size'])
       
        
    def __len__(self):
      
        return len(self.sample_files) 
    
    def __getitem__(self, idx):
        sample_file = self.sample_files[idx]
        label_file = self.label_files[idx]
        mask_file = self.mask_files[idx]
        sample = np.load(os.path.join(self.data_dir, sample_file))
        label = np.load(os.path.join(self.data_dir, label_file))
        mask = np.load(os.path.join(self.data_dir, mask_file))
        sample = torch.from_numpy(sample).float()
        label = torch.from_numpy(label).long()
        mask = torch.from_numpy(mask).bool()
        return sample, label , mask
    


class ClassificationBenchmarkDataset(Dataset):
    """Pytorch dataset for loading batches of patches of scenes from the ASID V2 data set for classification task."""
    
    def __init__(self , data_options, model_options ,files, transform_list=None):
        self.data_options = data_options
        self.model_options = model_options
        self.files = files
        self.patch_size = int(self.data_options['data_preprocess_options']['patch_size'])
        self.batch_size = int(self.model_options['train']['batch_size'])
        self.chart = self.model_options['model']['label'].strip("'")
        
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return int(self.model_options['datamodule']['epoch_len'])
    
    def check_patch_conditions(self, scene, row_rand, col_rand , threshold):
        """
            Checks conditions for the validity of a scene patch based on given criteria.

            Args:
                scene (xarray): A xarray containing scene data, including 'SIC', 'distance_to_border', and 'distance_map'.
                row_rand (int): Row coordinate of the patch's starting point.
                col_rand (int): Column coordinate of the patch's starting point.

            Returns:
                bool: True if all specified conditions are met, False otherwise.
        """
        # make the area patch dist map if hase a distance from land     
        num_pixels_valid_threshold = int(self.data_options['patch_randomcrop_config']['num_pixels_valid'])
        num_pixels_non_valid = int(self.data_options['patch_randomcrop_config']['num_pixels_non_valid'])

        #sample_y = scene['SIC'].values[row_rand: row_rand + self.patch_size, 
        #                              col_rand: col_rand + self.patch_size]
        #condition_valid_pixels = np.sum(sample_y == self.data_options['class_fill_values']['SIC']) / sample_y.ravel().shape[0] <= threshold #< num_pixels_non_valid
        patch_distance_to_border = scene['distance_to_border'].values[row_rand: row_rand + self.patch_size, 
                                                                          col_rand: col_rand + self.patch_size]
        
        patch_dist_map = scene['distance_map'].values[row_rand: row_rand + self.patch_size, 
                                                          col_rand: col_rand + self.patch_size]

        all_distance_border = np.all(patch_distance_to_border > self.data_options['data_preprocess_options']['distance_border_threshold']) 
        condition_land = np.all(patch_dist_map > 0)
        condition_distance = all_distance_border 
        #condition_valid_pixels = npsum > num_pixels_valid_threshold 

        
        distance_to_border_enabled = self.data_options['data_preprocess_options']['distance_to_border']
        land_mask_enabled = self.data_options['data_preprocess_options']['land_masking']


        if distance_to_border_enabled and land_mask_enabled:

            final_condition = condition_land and condition_distance 
        elif distance_to_border_enabled:
            final_condition = condition_distance 
        elif land_mask_enabled:
            final_condition = condition_land
        else:
            final_condition = True

        if final_condition:
            return True
        else:
            return False

    def check_label(self, scene, row, col , threshold):
        '''
            Checks conditions for the validity of a scene patch based on given criteria.

        '''
      
        not_filled = self.data_options['class_fill_values'][self.chart]

        y_patch = scene[self.chart].values[row: row + self.patch_size, 
                                      col: col + self.patch_size]
        npsum = np.sum(y_patch == self.data_options['class_fill_values'][self.chart])
       

        if np.all(y_patch == y_patch[0][0]) and npsum == 0:
            return True , y_patch[0][0]
        else:
            return False , not_filled
        
    def apply_threshold_to_binary(self ,scene):
        binary_task = self.data_options['SIC_config']['binary_label']
        pure_polygon = self.data_options['SIC_config']['pure_polygon']
            
        if binary_task and pure_polygon:
                    
            
            scene['binary'] = scene['SIC'].copy()
            replacement_value = self.data_options['class_fill_values']['SIC']
            modified_array = scene['binary'].values.copy()
            indices_to_replace = np.logical_and(modified_array != 0, modified_array != 10)
            modified_array[indices_to_replace] = replacement_value
            scene['binary'].values = modified_array
            
            scene['binary'].values[scene['binary'].values == 10] = 1
                
            scene['binary'].values[scene['SIC'].values == self.data_options['class_fill_values']['SIC']] = self.data_options['class_fill_values']['SIC']
                
            threshold = 0
        elif binary_task and  self.data_options['SIC_config']['ice_threshold_enabled'] :
            
            scene['binary'] = scene['SIC'].copy()
            scene['binary'].values[scene['binary'].values > self.data_options['SIC_config']['ice_threshold']] = 1
            scene['binary'].values[scene['SIC'].values == scene['class_fill_values']['SIC']] = self.data_options['class_fill_values']['SIC']            

            #copy the scene['binary'] to scene['SIC']
        scene['SIC'] = scene['binary'].copy()
        return scene , threshold    


    def random_crop(self, scene):
        
        max_attempts = 100
        threshold = 0
        if self.data_options['SIC_config']['binary_label']:
            scene , threshold = self.apply_threshold_to_binary(scene)

        #self.data_options['full_variables'] = self.data_options['sar_variables'] + self.data_options['charts']
        self.data_options['full_variables'] = self.data_options['sar_variables'] 
        patch = np.zeros((len(self.data_options['full_variables']) + len(self.data_options['amsr_env_variables']) 
                          , self.patch_size, self.patch_size))
        y_patch = np.zeros((self.patch_size, self.patch_size))

        for attempts in range(max_attempts):
            row_rand = np.random.randint(low=0, high=scene['SIC'].values.shape[0] -  self.patch_size)
            col_rand = np.random.randint(low=0, high=scene['SIC'].values.shape[1] -  self.patch_size)
            
            condition = self.check_patch_conditions(scene, row_rand, col_rand)
            condition_valid_label , label = self.check_label(scene, row_rand, col_rand , threshold = threshold)


            if condition and condition_valid_label:

                patch[0:len(self.data_options['full_variables']), :, :] = scene[self.data_options['full_variables']].isel(
                    sar_lines=range(row_rand, row_rand + self.patch_size),
                    sar_samples=range(col_rand, col_rand + self.patch_size)).to_array().values
                
                
                # Equivalent in amsr and env variable grid.
                if len(self.data_options['amsr_env_variables']) > 0:
                    amsrenv_row = row_rand / self.data_options['amsrenv_delta']
                    amsrenv_row_dec = int(amsrenv_row - int(amsrenv_row))  # Used in determining the location of the crop in between pixels.
                    amsrenv_row_index_crop = amsrenv_row_dec * self.data_options['amsrenv_delta'] * amsrenv_row_dec
                    amsrenv_col = col_rand / self.data_options['amsrenv_delta']
                    amsrenv_col_dec = int(amsrenv_col - int(amsrenv_col))
                    amsrenv_col_index_crop = amsrenv_col_dec * self.data_options['amsrenv_delta'] * amsrenv_col_dec
                    # Crop and upsample low resolution variables.
                    patch[len(self.data_options['full_variables']):, :, :] = torch.nn.functional.interpolate(
                    input=torch.from_numpy(scene[self.data_options['amsr_env_variables']].to_array().values[
                            :, 
                            int(amsrenv_row): int(amsrenv_row + np.ceil(self.data_options['amsrenv_patch'])),
                            int(amsrenv_col): int(amsrenv_col + np.ceil(self.data_options['amsrenv_patch']))]
                        ).unsqueeze(0),
                        size=self.data_options['amsrenv_upsample_shape'],
                        mode=self.data_options['data_preprocess_options']['loader_upsampling']).squeeze(0)[
                        :,
                        int(np.around(amsrenv_row_index_crop)): int(np.around(amsrenv_row_index_crop + self.patch_size)),
                        int(np.around(amsrenv_col_index_crop)): int(np.around(amsrenv_col_index_crop + self.patch_size))].numpy()
                
                return patch, label


        patch , label = None , None
        return patch , label
    



    def __getitem__(self, idx):
        sample = np.zeros((self.batch_size, self.patch_c,
                            self.patch_size, self.patch_size))
        label = np.zeros((self.batch_size,))
        sample_n = 0

        
        scene_id = np.random.randint(low=0, high=len(self.files), size=1).item()

        while sample_n < 2:
            # - Open memory location of scene. Uses 'Lazy Loading'.
            scene_id = np.random.randint(low=0, high=len(self.files), size=1).item()

            # - Load scene
            scene = xr.open_dataset(os.path.join(self.data_options['dir_train_with_icecharts'], self.files[scene_id]))
            # - Extract patches
            try:
                sample , label = self.random_crop(scene)
                sample_n += 1
            except:
                print(f"Cropping in {self.files[scene_id]} failed.")
                print(f"Scene size: {scene['SIC'].values.shape} for crop shape: ({self.patch_size}, {self.patch_size})")
                print('Skipping scene.')
                continue


        label = np.reshape(label, (-1, 1))
        sample = torch.from_numpy(sample)
        label = torch.from_numpy(label)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

