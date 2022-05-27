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

    # def train(self, dataset: BaseADDataset, encoder: BaseNet, decoder: BaseNet, kmeans: BaseNet,net_main: BaseNet):
    #     logger = logging.getLogger()

    #     # Set device for network
    #     net_cond = nn.Sequential(encoder, decoder) 
    #     net_main = net_main.to(self.device)
        
    #     # Get train data loader
    #     train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
    #     net=nn.Sequential(encoder, decoder,net_main)
    #     # Set optimizer (Adam optimizer for now)
    #     optimizer = optim.Adam(net_cond.parameters(), lr=0.01, weight_decay=self.weight_decay,
    #                             amsgrad=self.optimizer_name == 'amsgrad')

    #     # Set learning rate scheduler
    #     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
    
        
    #     # Training
    #     # step 1: conditional network
    #     # step 2: main network
    #     logger.info('Starting training...')
    #     start_time = time.time()
    #     net_cond.train()
    #     optimizer.zero_grad()
    #     torch.autograd.set_detect_anomaly(True)
    #     max_iter=10

    #     self.centroids = Variable(self.centroid_init(train_loader,encoder).data)

    #     loss_cond_old=torch.tensor([10000000.0], requires_grad=True)
    #     loss1=torch.tensor([0.0], requires_grad=True)
    #     logger.info('centroids:%s'%self.centroids)   

    #     loss_main_old=10000000.0
    #     def scaler(df):
    #         x = df #returns a numpy array
    #         min_max_scaler = preprocessing.MinMaxScaler()
    #         x_scaled = min_max_scaler.fit_transform(x)
          
    #         return x_scaled
        
    #     scheduler.step()
    #     loss_epoch = 0.0
    #     loss_main = 0.0
    #     loss_cond = 0.0
    #     n_batches = 0
    #     epoch_start_time = time.time()
    #     alpha=self.alpha

    #     for ec in range(5000):
    #         optimizer.zero_grad()
    #         loss_epoch_cond=0.0
    #         centroid_sums = torch.zeros_like(self.centroids)
    #         centroid_counts = Variable(torch.zeros(self.n_class))
    #         for data in train_loader:
    #             optimizer.zero_grad()
    #             side,inputs, _= data
    #             # logger.info(side)
    #             side_scaled=torch.tensor(scaler(side.numpy()))
    #             embeddings = encoder(side) #net_cond(side)
    #             X_hat = decoder(embeddings)
    #             recon_loss = F.mse_loss(X_hat, side)
    #             # logger.info(embeddings[:5])
    #             cluster_loss, cluster_assignments = kmeans(embeddings, self.centroids)
    #             cluster_assignments = F.one_hot(cluster_assignments, self.n_class)
    #             cond_loss = recon_loss + cluster_loss

    #             # logger.info(cluster_assignments)
    #             # logger.info(recon_loss)
    #             # logger.info(cluster_loss)
    #             # logger.info('embeddings: %s'%embeddings)

                
    #             self.centroids  = self.update_clusters(centroid_sums,cluster_assignments,embeddings)
    #             loss_epoch_cond += cond_loss.item()
    #             optimizer.zero_grad()
    #             cond_loss.backward(retain_graph=True)
    #             optimizer.step()
    #                   # logger.info(self.centroids)
    #             if (ec%1000==0):
    #                 logger.info('  iter {}\t Loss: {:.8f}' .format(ec + 1, loss_epoch_cond / (ec+1)))


    #         # logger.info('cluster asssignment for 5 pts: %s'%cluster_assignments[:5])
    #         # logger.info('cluster asssignment for 5 pts: %s'%cluster_assignments[-5:])
    #     self.assignment=torch.transpose(cluster_assignments,0,1)
    #     logger.info(cluster_assignments)
    #     sys.exit()
    
    
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
        loss_main_old=0.0
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
                # logger.info(outputs,self.c)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                    # loss= torch.sum(torch.mul(assignment[:,k],dist))/assignment[:,k].sum()
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1
            # if abs(loss_main_old-loss)/loss_main_old<0.000001:
            #     break
            # else:
            #     loss_main_old=loss
            # log epoch statistics
            if epoch%100==0:
                epoch_train_time = time.time() - epoch_start_time
                logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                            .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))
            # logger.info(outputs[:5])
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
        logger = logging.getLogger()

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
        logger = logging.getLogger()
        centroid_sums = Variable(torch.zeros(self.n_class, encoder.rep_dim))
        centroid_counts = Variable(torch.zeros(self.n_class))
        for data in train_loader:
            side,inputs, _ = data
            side_scaled=torch.tensor(scaler(side.numpy()))
            X_var=Variable(side)
            cluster_ass = Variable(torch.LongTensor(side.size(0)).random_(self.n_class))
            cluster_assignments = F.one_hot(cluster_ass, self.n_class)
            # logger.info(cluster_assignments)
            embeddings = encoder(X_var)
            centroid_sums=self.update_clusters(centroid_sums,cluster_assignments, embeddings)
        return centroid_sums.clone()
            
        # dataset = train_loader.dataset
        # n_samples = len(dataset)
        # centers = [torch.zeros(net.rep_dim, device=self.device)]*self.n_class
        # used_idx = []
        # net.eval()
        # def scaler(df):
        #     x = df #returns a numpy array
        #     min_max_scaler = preprocessing.MinMaxScaler()
        #     x_scaled = min_max_scaler.fit_transform(x)
          
        #     return x_scaled
        # with torch.no_grad():
        #     for data in train_loader:
        #         # get the inputs of the batch
        #         side,inputs, _ = data
        #         inputs = inputs.to(self.device)
        #         side_info = torch.tensor(scaler(side.numpy()))
        #         outputs = side_info #net(side_info)
                
        #         for k in range(self.n_class):
        #             idx = np.random.choice(n_samples)
        #             while idx in used_idx:
        #                 idx = np.random.choice(n_samples)
        #             used_idx.append(idx)
        #             centers[k] = outputs[idx]
    
        # return centers
    
    def cluster_responsibilities(self,side_info, centers, beta):
        logger = logging.getLogger()
        N, _ = side_info.shape[0], side_info.shape[1]
        dist =  torch.zeros(N, self.n_class)  #np.zeros((N, K))
        # min_max_scaler = preprocessing.MinMaxScaler()
        side = side_info
        
        # logger.info(side)
        
        for k in range(self.n_class):
            for n in range(N): 
                # logger.info(torch.sum((side_info[n]-centers[k])**2))
                dist[n,k] = torch.exp(-self.beta*torch.sum((side[n]-centers[k])**2))
                # dist[n,k] = torch.sum((side_info[n]-centers[k])**2)
        # logger.info(dist)
        
        assign = dist/dist.sum(1, keepdim=True)
        # logger.info(assign)
        return assign
    
    # def update_clusters(self,centroid_sums, centroid_counts,
    #                 cluster_assignments, embeddings):
    #     logger = logging.getLogger()
    #     k = centroid_sums.size(0)
    #     centroid_sums.index_add_(0, cluster_assignments, embeddings)
    #     np_counts = np.bincount(cluster_assignments.data.numpy(), minlength=k)
    #     centroid_counts.add_(Variable(torch.FloatTensor(np_counts)))
    #     # logger.info(embeddings)
    #     # logger.info(cluster_assignments)
    #     # logger.info(centroid_sums)
    #     # logger.info(centroid_counts)        
    #     return centroid_sums,centroid_counts
        
    # def centroid_init(self,trainloader,net: BaseNet):
    #     logger = logging.getLogger()
    #     k=self.n_class
    #     centroid_sums = Variable(torch.zeros(k, net.rep_dim))
    #     centroid_counts = Variable(torch.zeros(k))
    #     for X,_,_ in trainloader:
    #         side = Variable(X)
    #         cluster_assignments = Variable(torch.LongTensor(X.size(0)).random_(k))
    #         embeddings = net(side)
    #         # logger.info(side)
    #         centroid_sums,centroid_counts = self.update_clusters(centroid_sums, centroid_counts,
    #                         cluster_assignments, embeddings)
        
    #     centroid_means = centroid_sums / centroid_counts[:, None]
    #     return centroid_means.clone()
    
    
    # def update_centers(self, side_info, r):
    #     N, D = side_info.shape[0], side_info.shape[1]
    #     centers = torch.zeros(self.n_class, D) #np.zeros((K, D))
    #     for k in range(self.n_class):
    #         centers[k] =  torch.sum(torch.mul(r[:, k].view(N, 1), side_info),0) / r[:, k].sum()
    #     return centers
    



def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
