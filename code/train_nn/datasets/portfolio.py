#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:00:43 2021

@author: nymisha
"""
from torch.utils.data.dataset import Dataset
from base.torchvision_dataset import TorchvisionDataset
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import logging


class Portfolio_Dataset(TorchvisionDataset):

    def __init__(self, root,test,normal_class):
        
        self.train_set=MyPortfolio(root,train=True,normal_class=normal_class)
        self.test_set =MyPortfolio(test,train=False,normal_class=normal_class)

def scaler(df):
    x=df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = 100*pd.DataFrame(x_scaled)
    return df
    
class MyPortfolio:
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, root, train,normal_class):
        super(MyPortfolio, self).__init__()
        logger = logging.getLogger()
        self.train=train
        if train:
            train_set = pd.read_csv(root)
            side_cols=[i for i in train_set.columns if '_SI' in i]
            main_cols=[i for i in train_set.columns if '_SI' not in i]

            self.train_set=train_set
            self.data_train_set=torch.tensor(scaler(train_set[main_cols]).values.astype(np.float32))
            self.side_train_set=torch.tensor(scaler(train_set[side_cols]).values.astype(np.float32))

        else:
            test_set = pd.read_csv(root)

            side_cols=[i for i in test_set.columns if '_SI' in i]
            main_cols=[i for i in test_set.columns if '_SI' not in i]

            self.test_set=test_set
            self.data_test_set=torch.tensor(scaler(test_set[main_cols]).values.astype(np.float32))
            self.side_test_set=torch.tensor(scaler(test_set[side_cols]).values.astype(np.float32))

    def __getitem__(self, index):
        if self.train:
            side, data = self.side_train_set[index], self.data_train_set[index]
        else:
            side, data = self.side_test_set[index], self.data_test_set[index]
        return (side, data, index)
    
    
    def __len__(self):
        if self.train:
            return len(self.train_set)
        else:
            return len(self.test_set)
        
        