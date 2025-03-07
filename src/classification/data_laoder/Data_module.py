import os
from utils import *
import numpy as np
import torch
from loader_from_directory import BenchmarkDataset_directory, BenchmarkTestDataset_directory
from sklearn.metrics import f1_score , recall_score, precision_score, accuracy_score
import pytorch_lightning as pl

class MyDataModule(pl.LightningDataModule):
    def __init__(self, hparams_data, hparams_model, transform_list):
        super().__init__()
        self.hparams_data = hparams_data
        self.hparams_model = hparams_model
        self.transform_list = transform_list

    def setup(self, stage=None):
        if stage == 'fit' :
            
            self.train_dataset = BenchmarkDataset_directory(self.hparams_data, self.hparams_model, self.hparams_data['dir_samples_labels_train'],transform_list=self.transform_list)
            self.val_dataset = BenchmarkDataset_directory(self.hparams_data, self.hparams_model, self.hparams_data['dir_samples_labels_val'], transform_list=self.transform_list)
        
        if stage == 'test' :
            self.test_dataset = BenchmarkTestDataset_directory(self.hparams_data, self.hparams_model, self.hparams_data['dir_samples_labels_test'])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=int(self.hparams_model['train']['batch_size']), shuffle=True, num_workers=int(self.hparams_model['train']['num_workers']))

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=int(self.hparams_model['train']['batch_size']), num_workers=int(self.hparams_model['train']['num_workers']))

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=int(self.hparams_model['train']['batch_size']),  num_workers=int(self.hparams_model['train']['num_workers']))




