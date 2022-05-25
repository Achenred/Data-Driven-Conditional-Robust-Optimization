import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base.base_net import BaseNet


class PORT_GEN(BaseNet):

    def __init__(self,main_size,out_size):
        super().__init__()

        rep = out_size
        self.rep_dim = rep
        
        
        self.fc1 = nn.Linear(main_size, int(np.ceil(main_size/2)), bias=False) 
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(int(np.ceil(main_size/2)), rep, bias=False)  
        # self.relu2 = nn.ReLU()
        
        # self.fc3 = nn.Linear(rep, rep, bias=False)       

        # Initialization
        nn.init.uniform_(self.fc1.weight, 0.,1.)
        nn.init.uniform_(self.fc2.weight, 0.,1.)
        # nn.init.uniform_(self.fc3.weight, 0.,1.)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # x = self.relu2(x)
        # x = self.fc3(x)
        
        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight,0.,1.)

#Chicago week/weekend nueral net

    # def __init__(self):
    #     super().__init__()

    #     rep = 50
    #     self.rep_dim = rep
        
        
    #     self.fc1 = nn.Linear(144, rep, bias=False)       
    #     self.fc2 = nn.Linear(rep, rep, bias=False)       
    #     self.fc3 = nn.Linear(rep, rep, bias=False)       
        
    #     self.relu1 = nn.ReLU()
    #     self.relu2 = nn.ReLU()
        
    # def forward(self, x):
    #     x = x.view(x.size(0), -1)
    #     x = self.fc1(x)
    #     x = self.relu1(x)
    #     x = self.fc2(x)
    #     x = self.relu2(x)
    #     x = self.fc3(x)
        
    #     return x


