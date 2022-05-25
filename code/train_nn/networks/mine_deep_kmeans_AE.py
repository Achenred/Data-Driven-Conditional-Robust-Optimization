import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet
import logging


class Encoder(nn.Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()

        self.rep_dim = output_size

        self.fc2 = nn.Linear(input_size, 10)

        self.fc4 = nn.Linear(10, output_size)

        nn.init.uniform_(self.fc2.weight, -1.,1.)
        # nn.init.uniform_(self.fc3.weight, 0.,0.5)
        nn.init.uniform_(self.fc4.weight, -1.,1.)
    
    def forward(self, x):
        out = x.view(x.size(0), -1)
        # out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        # out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

class Decoder(nn.Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(output_size, 10)
        # self.fc2 = nn.Linear(hidden_sizes[1], hidden_sizes[0])
        self.fc3 = nn.Linear(10, input_size)

        nn.init.uniform_(self.fc1.weight, -1.,1.)
        # nn.init.uniform_(self.fc2.weight, 0.,0.5)
        nn.init.uniform_(self.fc3.weight, -1.,1.)

    
    def forward(self, x):
        out = F.relu(self.fc1(x))
        # out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
#         out = F.relu(self.fc4(out))
        out = out.view(out.size(0), -1)
        return out

class KMeansCriterion(nn.Module):
    
    def __init__(self, lmbda):
        super().__init__()

        self.lmbda = lmbda
    
    def forward(self, embeddings, centroids):
        distances = self.lmbda*torch.sum(torch.abs(embeddings[:, None, :] - centroids)**2, 2)
        cluster_distances, cluster_assignments = distances.max(1)
        loss = self.lmbda * cluster_distances.sum()
        return loss, cluster_assignments.detach()
    
    

def block_net(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f, bias=False) ,
        nn.ReLU(),
        nn.Linear(out_f, out_f, bias=False) # ,
        # nn.ReLU(),
        # nn.Linear(out_f, out_f, bias=False) 
    )
class main_net_AE(BaseNet):

    def __init__(self):
        super().__init__()

        rep = 5
        self.rep_dim = rep
        
        
        self.fc1 = nn.Linear(15, 10, bias=False) 
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(10, rep, bias=False)  
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

        



