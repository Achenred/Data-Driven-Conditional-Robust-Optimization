from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from soft_assign import soft_assign
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import logging
import time
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import sys


class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, n_class, R, c,beta,alpha, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        self.n_class=n_class
        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu
        self.beta=beta
        self.alpha=alpha

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        self.assignment=None
        self.centroids=None

    def train(self, dataset: BaseADDataset, encoder: BaseNet, decoder: BaseNet, soft_kmeans: BaseNet,net_main: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net_cond = nn.Sequential(encoder, decoder) 
        net_main = net_main.to(self.device)
        
        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        net=nn.Sequential(encoder, decoder,net_main)
        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
    
        
        # Training
        # step 1: conditional network
        # step 2: main network
        logger.info('Starting training...')
        start_time = time.time()
        net_cond.train()
        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)

        self.centroids = Variable(self.centroid_init(train_loader,encoder).data)


        logger.info('centroids:%s'%self.centroids)   
        def scaler(df):
            x = df #returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
          
            return x_scaled
        for epoch in range(self.n_epochs):
            scheduler.step()
            loss_epoch = 0.0
            loss_main = 0.0
            loss_cond = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            alpha=self.alpha

            # for ec in range(1):
            optimizer.zero_grad()
            loss_epoch_cond=0.0
            centroid_sums = torch.zeros_like(self.centroids)

            for data in train_loader:
                optimizer.zero_grad()
                side,inputs, _= data

                embeddings = encoder(side) #net_cond(side)
                X_hat = decoder(embeddings)
                recon_loss = F.mse_loss(X_hat, side)
                # logger.info(embeddings[:5])
                cluster_loss, cluster_assignments = soft_kmeans(embeddings, self.centroids)
                cond_loss = recon_loss + cluster_loss

                self.centroids  = self.update_clusters(centroid_sums,cluster_assignments,embeddings)
                
                loss_epoch_cond += cond_loss.item()
                 
            self.assignment=torch.transpose(cluster_assignments,0,1)

            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            # Initialize hypersphere center c (if c not loaded)
            if self.c is None:
                logger.info('Initializing center c...')
                self.c = self.init_center_c(train_loader,self.assignment,net_main)
                logger.info('Center c initialized.')

            for data in train_loader:
                
                side,inputs, _= data
                inputs = inputs.to(self.device)

                torch.autograd.set_detect_anomaly(True)

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net_main(inputs)

                loss=torch.tensor([0.0], requires_grad=True)
                loss_list=[]
                for k in range(self.n_class):
                    dist = torch.sum((outputs[k] - self.c[k]) ** 2, dim=1)
                   
                    if self.objective == 'soft-boundary':
                        scores = dist - self.R ** 2
                        loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                    else:
                        loss = loss+torch.sum(torch.mul(self.assignment[k],dist))/self.assignment[k].sum()
                        loss_list.append(torch.sum(torch.mul(self.assignment[k],dist))/self.assignment[k].sum().item())
                        
                loss/=self.n_class
                loss_main+=loss.item()
                loss_cond+=cond_loss.item()
                
                loss=(1-alpha)*loss+alpha*cond_loss
               
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
           
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            if epoch%100==0:

                logger.info('  Epoch {}/{}\t Time: {:.3f}\t Total Loss: {:.8f}\t Main Loss: {:.8f}\t Conditional Loss: {:.8f}'
                            .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches, loss_main/ n_batches, loss_cond/ n_batches))
                logger.info(loss_list)
                logger.info('cluster asssignment %s'%self.assignment)
        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        self.test_auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))

        logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader,assignment, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        if self.n_class==1:
            n_samples = 0
            c = torch.zeros(net.rep_dim, device=self.device)
    
            net.eval()
            with torch.no_grad():
                for data in train_loader:
                    # get the inputs of the batch
                    side,inputs, _ = data
                    inputs = inputs.to(self.device)
                    outputs = net(inputs)
                    n_samples += outputs[0].shape[0]
                    c += torch.sum(outputs[0], dim=0)
    
            c /= n_samples
    
            # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
            c[(abs(c) < eps) & (c < 0)] = -eps
            c[(abs(c) < eps) & (c > 0)] = eps
            return c
        else:
            dataset = train_loader.dataset
            n_samples = len(dataset)
            c = [torch.zeros(net.rep_dim, device=self.device)]*self.n_class
            n_samples=[0]*self.n_class
            net.eval()
            with torch.no_grad():
                for data in train_loader:
                    # get the inputs of the batch
                    side,inputs, _ = data

                    inputs = inputs.to(self.device)
                    outputs = net(inputs)
                    
                    for k in range(self.n_class):
                        n_samples[k] = outputs[k].shape[0]
                  
                        c[k] = torch.transpose(torch.matmul(torch.transpose(outputs[k],0,1),assignment[k].float().view(outputs[k].shape[0],1)),0,1)[0]

            for k in range(self.n_class):
                c[k] /= assignment[k].sum()
                c[k][(abs(c[k]) < 0.01) & (c[k] < 0)] = -0.01
                c[k][(abs(c[k]) < 0.01) & (c[k] >= 0)] = 0.01
        # logger.info(c)
        return c
    
    def update_clusters(self, centroid_sums,cluster_assignments, embeddings):
        k = centroid_sums.size(0)
        
        for i in range(k):
           
            temp = cluster_assignments[:,i][:, None]*embeddings
            centroid_sums[i] = temp.sum(dim=0)/  cluster_assignments[:,i].sum()
        return centroid_sums.detach()
    
    
    def centroid_init(self,train_loader: DataLoader,encoder: BaseNet):
        def scaler(df):
            x = df #returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            return x_scaled
        centroid_sums = Variable(torch.zeros(self.n_class, encoder.rep_dim))
        for data in train_loader:
            side,inputs, _ = data
            X_var=Variable(side)
            cluster_ass = Variable(torch.LongTensor(side.size(0)).random_(self.n_class))
            cluster_assignments = F.one_hot(cluster_ass, self.n_class)
            # logger.info(cluster_assignments)
            embeddings = encoder(X_var)
            centroid_sums=self.update_clusters(centroid_sums,cluster_assignments, embeddings)
        return centroid_sums.clone()
          
    
    def cluster_responsibilities(self,side_info, centers, beta):
        N, _ = side_info.shape[0], side_info.shape[1]
        dist =  torch.zeros(N, self.n_class)  #np.zeros((N, K))
        side = side_info
        
        # logger.info(side)
        
        for k in range(self.n_class):
            for n in range(N): 
                dist[n,k] = torch.exp(-self.beta*torch.sum((side[n]-centers[k])**2))
                
        assign = dist/dist.sum(1, keepdim=True)
        return assign

def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
