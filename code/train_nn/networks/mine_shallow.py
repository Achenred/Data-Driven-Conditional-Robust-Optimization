import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet

torch.manual_seed(0)
class MINE_SHALLOW(BaseNet):

    
    def __init__(self):
        super().__init__()

        rep = 16
        self.rep_dim = rep
        
        self.fc1 = nn.Linear(59, 30, bias=False)           
        self.fc3 = nn.Linear(30, rep, bias=False)   
            
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc3(x)
        
        return x
    
