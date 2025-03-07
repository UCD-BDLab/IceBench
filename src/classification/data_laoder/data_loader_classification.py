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

class BenchmarkTestDataset_directory:
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
    


