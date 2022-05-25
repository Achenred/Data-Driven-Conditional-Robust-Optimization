import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet
import logging


class conditional_assign(BaseNet):

    def __init__(self):
        super().__init__()

        rep = 5
        self.rep_dim = rep
        
        
        self.fc1 = nn.Linear(35, rep, bias=False) 
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        
        return x
    

def block_net(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f, bias=False) ,
        nn.ReLU(),
        nn.Linear(out_f, out_f, bias=False) # ,
        # nn.ReLU(),
        # nn.Linear(out_f, out_f, bias=False) 
    )
class main_network(BaseNet):
    def __init__(self,n_class):
        super().__init__()
        logger = logging.getLogger()
        rep = 5
        self.rep_dim = rep
        in_f=32
        out_f=rep
        # self.trace = []
        self.net_blocks = nn.ModuleList([block_net(in_f, out_f) for _ in range(n_class)])
        self.net_blocks.apply(self._init_weights)
        # logger.info(self.net_blocks)

        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out=[]
        for net in self.net_blocks:
            out.append(net(x))
        return out
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight,0.,1.)

        



