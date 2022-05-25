import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet
import logging


class Encoder3(nn.Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()

        self.rep_dim = output_size
#         self.fc1 = nn.Linear(784, 1000)
#         self.fc2 = nn.Linear(1000, 250)
        self.fc2 = nn.Linear(input_size, 15)
        # self.fc3 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc4 = nn.Linear(15, output_size)

        nn.init.uniform_(self.fc2.weight, 0.,1.)
        # nn.init.uniform_(self.fc3.weight, 0.,0.5)
        nn.init.uniform_(self.fc4.weight, 0.,1.)
    
    def forward(self, x):
        out = x.view(x.size(0), -1)
        # out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        # out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

class Decoder3(nn.Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(output_size, 15)
        # self.fc2 = nn.Linear(hidden_sizes[1], hidden_sizes[0])
        self.fc3 = nn.Linear(15, input_size)

        nn.init.uniform_(self.fc1.weight, 0.,1.)
        # nn.init.uniform_(self.fc2.weight, 0.,0.5)
        nn.init.uniform_(self.fc3.weight, 0.,1.)
#         self.fc3 = nn.Linear(250, 1000)
#         self.fc4 = nn.Linear(1000, 784)
    
    def forward(self, x):
        out = F.relu(self.fc1(x))
        # out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
#         out = F.relu(self.fc4(out))
        out = out.view(out.size(0), -1)
        return out

class Soft_KMeansCriterion3(nn.Module):
    
    def __init__(self,beta, lmbda):
        super().__init__()
        self.beta=beta
        self.lmbda = lmbda
    
    def forward(self, embeddings, centroids):
        # distances = self.lmbda*torch.sum(torch.abs(embeddings[:, None, :] - centroids), 2)
        
        distances = self.lmbda*torch.sqrt(torch.sum((embeddings[:, None, :] - centroids)**2, 2))
        m = nn.Softmax(dim=1)
        cluster_assignments = m(-self.beta*(torch.sub(distances, distances.max(1)[0].view(distances.shape[0], 1))))
        
        loss=torch.mul(cluster_assignments,distances).sum(1).mean()
        return loss, cluster_assignments.detach()    

def block_net(in_f, out_f,h1,h2):
    return nn.Sequential(
        nn.Linear(in_f, h1, bias=False) ,
        nn.ReLU(),
        nn.Linear(h1, h2, bias=False) ,
        nn.ReLU(),
        nn.Linear(h2, out_f, bias=False) 
    )
class main_net_AE3(BaseNet):
    def __init__(self,n_class):
        super().__init__()
        logger = logging.getLogger()
        rep = 5
        self.rep_dim = rep
        in_f=13
        out_f=rep
        # self.trace = []
        self.net_blocks = nn.ModuleList([block_net(in_f, out_f,10,8) for _ in range(n_class)])
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

        



