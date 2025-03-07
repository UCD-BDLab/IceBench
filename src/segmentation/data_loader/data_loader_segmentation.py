import os
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
from torchvision import transforms


class BenchmarkDataset(Dataset):
    
    def __init__(self , data_options, model_options , files,transform_list = None):
        self.data_options = data_options
        self.model_options = model_options
        self.files = files
        self.patch_size = int(self.data_options['data_preprocess_options']['patch_size'])
        self.batch_size = int(self.model_options['train']['batch_size'])
        self.class_label = self.model_options['model']['label'].strip("'")
        
        
        if transform_list is None:
            transform_list = []

        #transform_list.append(transforms.Normalize(mean=mean_values, std=std_values))

        self.transform = transforms.Compose(transform_list)

        self.patch_c = len( self.data_options['sar_variables'])  + len(self.data_options['amsr_env_variables']) + 1 # for adding month number

        
        

    def __len__(self):
        return int(self.model_options['datamodule']['epoch_len'])
    
    def check_patch_conditions(self, scene, row_rand, col_rand ):
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

        distance_to_border_enabled = self.data_options['data_preprocess_options']['distance_to_border']
        land_mask_enabled = self.data_options['data_preprocess_options']['land_masking']

        sample_y = scene[self.class_label].values[row_rand: row_rand + self.patch_size, col_rand: col_rand + self.patch_size]
        ####################################################################################
        # patch_x = scene['nersc_sar_primary'].values[row_rand: row_rand + self.patch_size, col_rand: col_rand + self.patch_size]
        # num_nan = np.sum(np.isnan(patch_x))
        # condition_patch_x = num_nan == 0
        condition_patch_y= np.sum(sample_y == self.data_options['class_fill_values'][self.class_label]) / sample_y.ravel().shape[0] <= 0.3 #< num_pixels_non_valid
        

        condition_valid_pixels = condition_patch_y

        if not distance_to_border_enabled and not land_mask_enabled:
            return condition_valid_pixels
        ####################################################################################
        patch_distance_to_border = scene['distance_to_border'].values[row_rand: row_rand + self.patch_size, 
                                                                          col_rand: col_rand + self.patch_size]
        
        patch_dist_map = scene['distance_map'].values[row_rand: row_rand + self.patch_size, 
                                                          col_rand: col_rand + self.patch_size]

        all_distance_border = np.all(patch_distance_to_border > self.data_options['data_preprocess_options']['distance_border_threshold']) 
        mean_std_dict = self.data_options['mean_std_dict']
        land_remove_value = 0 - mean_std_dict['distance_map']['mean'] / mean_std_dict['distance_map']['std']
        condition_land = np.all(patch_dist_map > land_remove_value)
        condition_distance = all_distance_border 
        #condition_valid_pixels = npsum > num_pixels_valid_threshold 

        del  patch_distance_to_border, patch_dist_map
    
        if distance_to_border_enabled and land_mask_enabled:
            final_condition = condition_land and condition_distance and condition_valid_pixels
            
        elif distance_to_border_enabled:
            final_condition = condition_distance and condition_valid_pixels

        elif land_mask_enabled:
            final_condition = condition_land and condition_valid_pixels
        else:
            final_condition = condition_valid_pixels

        if final_condition:
            return True
        else:
            return False


    
    def random_crop(self, scene):
        max_attempts = 100
        #self.data_options['full_variables'] = self.data_options['sar_variables'] + self.data_options['charts']
        self.data_options['full_variables'] = self.data_options['sar_variables'] 
        patch = np.zeros((len(self.data_options['sar_variables']) + len(self.data_options['amsr_env_variables']) 
                          , self.patch_size, self.patch_size))
        y_patch = np.zeros((self.patch_size, self.patch_size))

        for attempts in range(max_attempts):
            row_rand = np.random.randint(low=0, high=scene['SIC'].values.shape[0] -  self.patch_size)
            col_rand = np.random.randint(low=0, high=scene['SIC'].values.shape[1] -  self.patch_size)
            
            condition = self.check_patch_conditions(scene, row_rand, col_rand)
           

            if condition:
            
                patch[0:len(self.data_options['sar_variables']), :, :] = scene[self.data_options['sar_variables']].isel(
                    sar_lines=range(row_rand, row_rand + self.patch_size),
                    sar_samples=range(col_rand, col_rand + self.patch_size)).to_array().values
              
                y_patch = scene[self.class_label].values[row_rand: row_rand + self.patch_size, col_rand: col_rand + self.patch_size]
              
                # Equivalent in amsr and env variable grid.
                if len(self.data_options['amsr_env_variables']) > 0:
                 
                    amsrenv_row = row_rand / self.data_options['amsrenv_delta']
                 
                    amsrenv_row_dec = int(amsrenv_row - int(amsrenv_row))  # Used in determining the location of the crop in between pixels.
                  
                    amsrenv_row_index_crop = amsrenv_row_dec * self.data_options['amsrenv_delta'] * amsrenv_row_dec
                 
                    amsrenv_col = col_rand / self.data_options['amsrenv_delta']
               
                    amsrenv_col_dec = int(amsrenv_col - int(amsrenv_col))
                 
                    amsrenv_col_index_crop = amsrenv_col_dec * self.data_options['amsrenv_delta'] * amsrenv_col_dec
                   
                    # Crop and upsample low resolution variables.
                    patch[len(self.data_options['sar_variables']):, :, :] = torch.nn.functional.interpolate(
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
                    
                return patch, y_patch

  
        patch , y_patch = None , None
        return patch , y_patch
    

    
    def extract_month_number(self,file_name):
        """
        Extracts the month number from a given filename.

        Args:
            file_name (str): The input filename from which the month number is to be extracted.

        Returns:
            int or None: The extracted month number (1 to 12) if successfully extracted and valid.
                        Returns None if the filename format is not recognized or the month number is out of range.
        """
        try:
            if file_name.startswith('S1'):
                # Extract the month number from characters 21 and 22
                month_number = int(file_name[21:23])
            elif file_name.endswith('_dmi_prep.nc'):
                month_number = int(file_name[4:6])
            else:
                
                date_portion = file_name[20:28]
                date_string = date_portion[:8]
                month_number = int(date_string[4:6])

            if 1 <= month_number <= 12:
                month_number = (month_number - 6.5) / 3.5
                return month_number
            else:
                return None  # Invalid month number

        except IndexError:
            return None  # The file name doesn't have the expected format
    
    def prep_dataset(self, patches , ys):
        """
        Convert patches from 4D numpy array to 4D torch tensor.
        Convert masks from 3D numpy array to 3D torch tensor.

        Parameters:
        patches (numpy.ndarray): 4D array of patches.
        masks (numpy.ndarray): 3D array of masks.

        Returns:
        torch.Tensor, torch.Tensor: Converted patches and masks as torch tensors.
        """
        # Convert numpy arrays to torch tensors
        #patches_tensor = torch.tensor(patches, dtype=torch.float32)  # Adjust dtype if needed
        #y_tensor = torch.tensor(ys, dtype=torch.long)      # Adjust dtype if needed
        patches_tensor = torch.from_numpy(patches).type(torch.float32)
        y_tensor = torch.from_numpy(ys).type(torch.long)

        return patches_tensor, y_tensor

    def __getitem__(self, idx):
        """
        Get batch. Function required by Pytorch dataset.

        Returns
        -------
        x :
            4D torch tensor; ready training data.
        y : Dict
            Dictionary with 3D torch tensors for each chart; reference data for training data x.
        """
        # Placeholder to fill with data.
        
        patches = np.zeros((self.batch_size, self.patch_c,
                            self.patch_size, self.patch_size))
        y = np.zeros((self.batch_size, self.patch_size, self.patch_size))
        sample_n = 0

        # Continue until batch is full.
        while sample_n < self.batch_size:
            # - Open memory location of scene. Uses 'Lazy Loading'.
            scene_id = np.random.randint(low=0, high=len(self.files), size=1).item()
          
            month = self.extract_month_number(self.files[scene_id])
            month_array = np.full((self.patch_size, self.patch_size), month)
            
            # - Load scene
            scene = xr.open_dataset(os.path.join(self.data_options['dir_train_with_icecharts'], self.files[scene_id]))
            # - Extract patches
            try:
                scene_patch , scene_y = self.random_crop(scene)

            except:
                print(f"Cropping in {self.files[scene_id]} failed.")
                print(f"Scene size: {scene['SIC'].values.shape} for crop shape: ({self.patch_size}, {self.patch_size})")
                print('Skipping scene.')
                continue
            
            if scene_patch is not None:
                # -- Stack the scene patches in patches
                #patches[sample_n, :, :, :] = scene_patch
                patches[sample_n, :-1, :, :] = scene_patch  # Fill all channels except the last one
                patches[sample_n, -1, :, :] = month_array  # Fill the last channel with the month data
           
                y[sample_n, :, :] = scene_y
                sample_n += 1 # Update the index.
            scene.close()

        # Prepare training arrays
        x, y_ = self.prep_dataset(patches=patches , ys = y)

        if self.transform is not None:
            x = self.transform(x)

        return x, y_

class BenchmarkTestDataset(Dataset):
    """Pytorch dataset for loading full scenes from the ASID ready-to-train challenge dataset for inference."""

    def __init__(self, data_options, model_options, files, test=False):
        self.data_options = data_options
        self.model_options = model_options
        self.files = files
        self.test = test
        self.class_label = self.model_options['model']['label'].strip("'")
        


    def __len__(self):
        """
        Provide the number of iterations. Function required by Pytorch dataset.

        Returns
        -------
        Number of scenes per validation.
        """
        return len(self.files)
    
    
    def extract_month_number(self, file_name):
        """
        Extracts the month number from a given filename.

        Args:
            file_name (str): The input filename from which the month number is to be extracted.

        Returns:
            int or None: The extracted month number (1 to 12) if successfully extracted and valid.
                        Returns None if the filename format is not recognized or the month number is out of range.
        """
        try:
            file_name = os.path.basename(file_name)
            if file_name.startswith('S1'):
                # Extract the month number from characters 21 and 22
                month_number = int(file_name[21:23])
            elif file_name.endswith('_dmi_prep.nc'):
                month_number = int(file_name[4:6])
            else:
                
                date_portion = file_name[20:28]
                date_string = date_portion[:8]
                month_number = int(date_string[4:6])

            if 1 <= month_number <= 12:
                month_number = (month_number - 6.5) / 3.5
                return month_number
            else:
                return None  # Invalid month number

        except IndexError:
            return None  # The file name doesn't have the expected format
        
    def prep_scene(self, scene , idx):
        """
        Upsample low resolution to match charts and SAR resolution. Convert patches from 4D numpy array to 4D torch tensor.

        Parameters
        ----------
        scene :

        Returns
        -------
        x :
            4D torch tensor, ready training data.
        y :
            Dict with 3D torch tensors for each reference chart; reference inference data for x. None if test is true.
        """
        if len(self.data_options['amsr_env_variables']) > 0:
            x = torch.cat((torch.from_numpy(scene[self.data_options['sar_variables']].to_array().values).unsqueeze(0),
                          torch.nn.functional.interpolate(
                              input=torch.from_numpy(scene[self.data_options['amsr_env_variables']].to_array().values).unsqueeze(0),
                              size=scene['nersc_sar_primary'].values.shape,
                              mode=self.data_options['data_preprocess_options']['loader_upsampling'])),
                          axis=1)
        else:
            x = torch.from_numpy(scene[self.data_options['sar_variables']].to_array().values).unsqueeze(0)

        
        month = self.extract_month_number(self.files[idx])

        month_array = torch.full_like(x[:, :1, :, :], fill_value=month)
        # Concatenate the month data as the last channel
        x = torch.cat((x, month_array), axis=1)

        x = x.to(dtype=torch.float32) 
   
        
        #if there is np.nan in x put 0 instead
        #x = torch.where(torch.isnan(x), torch.tensor(0.0, dtype=torch.float32), x)

        #y = torch.from_numpy(scene[self.class_label].values).unsqueeze(0).to(dtype=torch.long)
        y = torch.from_numpy(scene[self.class_label].values).to(dtype=torch.long)
      
        return x, y

    def __getitem__(self, idx):
        """
        Get scene. Function required by Pytorch dataset.

        Returns
        -------
        x :
            4D torch tensor; ready inference data.
        y :
            Dict with 3D torch tensors for each reference chart; reference inference data for x. None if test is true.
        masks :
            Dict with 2D torch tensors; mask for each chart for loss calculation. Contain only SAR mask if test is true.
        name : str
            Name of scene.

        """
        if self.test:
            dir = self.data_options['dir_test_with_icecharts']
        else:
            dir = self.data_options['dir_train_with_icecharts']
        
        scene = xr.open_dataset(os.path.join(dir, self.files[idx]))
        x, y = self.prep_scene(scene , idx)
        name = self.files[idx]

        masks = (y == self.data_options['class_fill_values'][self.class_label]).squeeze()

        return x, y, masks, name
