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
import copy
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

    def train(self, clus, labels,dataset: BaseADDataset, net: BaseNet):
        net = net.to(self.device)
        logger = logging.getLogger()
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader,clus,labels, net)
            logger.info('Center c initialized.')

        # Get train data loader
        
        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)


        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()

        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            loss=torch.tensor([0.0], requires_grad=True)
            for data in train_loader:
                side, inputs, _ = data
                inputs_k=inputs[labels==clus]
                inputs_k = inputs_k.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs_k)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1
            if epoch%100==0:
                epoch_train_time = time.time() - epoch_start_time
                logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                            .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

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

    def init_center_c(self, train_loader: DataLoader,clus,assignment, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                side,inputs, _ = data
                # logger.info(inputs)
                inputs_k=torch.tensor(inputs[np.where(assignment==clus)])
                inputs_k = inputs_k.to(self.device)
                outputs = net(inputs_k)
                # logger.info(outputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c
        
    def update_clusters(self, centroid_sums,cluster_assignments, embeddings):
        k = centroid_sums.size(0)
        # print(cluster_assignments.shape, embeddings.shape)
        for i in range(k):
            # print(cluster_assignments[:,i][:, None].shape)
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
            embeddings = encoder(X_var)
            centroid_sums=self.update_clusters(centroid_sums,cluster_assignments, embeddings)
        return centroid_sums.clone()
            
    
    def cluster_responsibilities(self,side_info, centers, beta):
        N, _ = side_info.shape[0], side_info.shape[1]
        dist =  torch.zeros(N, self.n_class)  #np.zeros((N, K))
        side = side_info
        for k in range(self.n_class):
            for n in range(N): 
                dist[n,k] = torch.exp(-self.beta*torch.sum((side[n]-centers[k])**2))

        assign = dist/dist.sum(1, keepdim=True)
        return assign
   

def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
