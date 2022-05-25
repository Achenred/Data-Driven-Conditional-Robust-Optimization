#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:00:43 2021

@author: nymisha
"""
from torch.utils.data.dataset import Dataset
from base.torchvision_dataset import TorchvisionDataset
import pandas as pd
import numpy as np
import torch
from utils.config import Config
import logging
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


class Generated_Dataset_no_side(TorchvisionDataset):

    def __init__(self, root,test,normal_class,xp_path):
        
        self.train_set=Generated(root,train=True,normal_class=normal_class,xp_path=xp_path)
        self.test_set =Generated(test,train=False,normal_class=normal_class,xp_path=xp_path)
    

class Generated:
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, root, train,normal_class,xp_path):
        super(Generated, self).__init__()
        
        cfg = Config(locals().copy())
        logger = logging.getLogger()
        def scaler(df):
            x=df.values
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df = pd.DataFrame(x_scaled)
            return df
        self.train=train
        if train:
            train_set = pd.read_csv(root, sep=",", header=None)
            self.data_train_set=torch.tensor(train_set.iloc[:,2:].values.astype(np.float32))
            self.side_train_set=torch.zeros(self.data_train_set.shape[0])
        else:
            test_set = pd.read_csv(root, sep=",", header=None)
            
            self.data_test_set=torch.tensor(test_set.iloc[:,2:].values.astype(np.float32))
            self.side_test_set=torch.zeros(self.data_test_set.shape[0])

    def __getitem__(self, index):
        if self.train:
            side, data = self.side_train_set[index], self.data_train_set[index]
        else:
            side, data = self.side_test_set[index], self.data_test_set[index]
        return (side, data, index)
    
    def __len__(self):
        if self.train:
            return len(self.data_train_set)
        else:
            return len(self.data_test_set)
        
        
        