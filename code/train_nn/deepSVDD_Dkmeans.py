import json
import torch
import numpy as np

from base.base_dataset import BaseADDataset
from networks.main import build_network, build_autoencoder
# from optim.deepSVDD_trainer_AE import DeepSVDDTrainer
from optim.deepSVDD_trainer_Dkmeans import DeepSVDDTrainer

from optim.ae_trainer import AETrainer
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from sklearn import preprocessing
import logging

class DeepSVDD(object):
    """A class for the Deep SVDD method.

    Attributes:
        objective: A string specifying the Deep SVDD objective (either 'one-class' or 'soft-boundary').
        nu: Deep SVDD hyperparameter nu (must be 0 < nu <= 1).
        R: Hypersphere radius R.
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network \phi.
        ae_net: The autoencoder network corresponding to \phi for network weights pretraining.
        trainer: DeepSVDDTrainer to train a Deep SVDD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SVDD network.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
    """

    def __init__(self, objective: str = 'one-class',n_class: int=1,beta:float=0.1,alpha:float=0.1, nu: float = 0.1):
        """Inits DeepSVDD with one of the two objectives and hyperparameter nu."""

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
        self.nu = nu
        self.n_class=n_class
        self.R = [0.0]*int(n_class)  # hypersphere radius R
        self.c = None  # hypersphere center c
        self.beta=beta
        self.alpha=alpha
        
        self.net_name = None
        self.net_cond = None  # conditional neural network for assignment
        self.net_main = None  # main neural network 
        self.nets=[]
        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

    def set_network(self, net_name,main_dim,side_dim,out_dim,beta,lmbda,n_class):
        """Builds the neural network \phi."""
        self.net_name = net_name
        self.encoder,self.decoder,self.soft_KMeans,self.net_main = build_network(net_name,main_dim,side_dim,out_dim,beta,lmbda,n_class)

    def train(self,clus,labels,dataset: BaseADDataset, n_class: int = 1 ,optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """Trains the Deep SVDD model on the training data."""

        self.optimizer_name = optimizer_name
        self.trainer = DeepSVDDTrainer(self.objective, self.n_class, self.R, self.c, self.beta,self.alpha, self.nu, optimizer_name, lr=lr,
                                       n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size,
                                       weight_decay=weight_decay, device=device, n_jobs_dataloader=n_jobs_dataloader)
        # Get the model
        self.net = self.trainer.train(clus,labels,dataset,self.net_main)
        self.R = self.trainer.R#.cpu().data.numpy()  # get float
        self.c = self.trainer.c#.cpu().data.numpy().tolist()  # get list
        
        self.results['train_time'] = self.trainer.train_time

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SVDD model on the test data."""

        if self.trainer is None:
            self.trainer = DeepSVDDTrainer(self.objective, self.R, self.c, self.nu,
                                           device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test(dataset, self.net)
        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

    def pretrain(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """Pretrains the weights for the Deep SVDD network \phi via autoencoder."""

        self.ae_net = build_autoencoder(self.net_name)
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)
        self.ae_trainer.test(dataset, self.ae_net)
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SVDD network weights from the encoder weights of the pretraining autoencoder."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def save_model(self, export_model, save_ae=True):
        """Save Deep SVDD model to export_model."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save({'R': self.R,
                    'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)

    def load_model(self, model_path, load_ae=False):
        """Load Deep SVDD model from model_path."""

        model_dict = torch.load(model_path)

        self.R = model_dict['R']
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
            
            
         
            
class AE(object):
    def __init__(self, net_name,main_dim,side_dim,out_dim,objective: str = 'one-class',n_class: int=1,beta:float=0.1,lmbda:float=0.1,alpha:float=0.1, nu: float = 0.1):
        k, d = 2, 3

        self.encoder,self.decoder,self.KMeans,self.net_main = build_network(net_name,main_dim,side_dim,out_dim,beta,lmbda,n_class)
        self.autoencoder = nn.Sequential(self.encoder, self.decoder)
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)
        
    
    def centroid_init(self,dataset,n_classes, d):
        logger = logging.getLogger()
        self.n_class=n_classes
        centroid_sums = Variable(torch.zeros(n_classes, d))
        centroid_counts = Variable(torch.zeros(n_classes))
        trainloader, _ = dataset.loaders(batch_size=1000)
        for data in trainloader:
            side,inputs, _= data
            # side_scaled=torch.tensor(scaler(side.numpy()))
            X_var=Variable(side)
            cluster_ass = Variable(torch.LongTensor(side.size(0)).random_(n_classes))

            cluster_assignments = F.one_hot(cluster_ass, n_classes)
            logger.info(cluster_assignments)
            logger.info(torch.sum(cluster_assignments,axis=0))
            embeddings = self.encoder(X_var)
            logger.info(embeddings)
            centroid_means=self.update_clusters(centroid_sums,cluster_assignments, embeddings)
            centroid_counts=torch.sum(cluster_assignments,axis=0)
        # centroid_means = centroid_sums / centroid_counts[:, None]
        # self.centroids=centroid_means
        return centroid_means.clone()

    def update_clusters(self, centroid_sums, cluster_assignments, embeddings):
        k = centroid_sums.size(0)
        logger = logging.getLogger()
        # print(cluster_assignments.shape, embeddings.shape)
        for i in range(k):
            # print(cluster_assignments[:,i][:, None].shape)
            temp = cluster_assignments[:,i][:, None]*embeddings
            centroid_sums[i] = temp.sum(dim=0)/  cluster_assignments[:,i].sum()
            logger.info(temp)
            logger.info(centroid_sums)
        # centroid_counts=torch.sum(cluster_assignments,axis=0)
        return centroid_sums.detach()
        # centroid_counts.add_(Variable(torch.FloatTensor(np_counts)))
    
    def train(self,dataset,centroids, print_every=100, verbose=False):
        logger = logging.getLogger()
        logger.info('started training......')
        k, d = centroids.size()
        def scaler(df):
            x = df #returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
          
            return x_scaled
        centroid_sums = torch.zeros_like(centroids)
        centroid_counts = Variable(torch.zeros(k))
        trainloader, _ = dataset.loaders(batch_size=1000)
        # run one epoch of gradient descent on autoencoders wrt centroids
        for data in trainloader:
            side,inputs, _= data
            # forward pass and compute loss
            # side_scaled=torch.tensor(scaler(side.numpy()))
            embeddings = self.encoder(side) #net_cond(side)
            logger.info(embeddings)
            X_hat = self.decoder(embeddings)
            recon_loss = F.mse_loss(X_hat, side)
            
            cluster_loss, cluster_assignments = self.KMeans(embeddings, centroids)
            logger.info(cluster_loss)
            logger.info(cluster_assignments)
            cluster_assignments = F.one_hot(cluster_assignments, self.n_class)
            loss = recon_loss + cluster_loss

            # run update step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            # store centroid sums and counts in memory for later centering
            centroid_means=self.update_clusters(centroid_sums,
                            cluster_assignments, embeddings)
            
            # if verbose and i % print_every == 0:
            #     # batch_hat = autoencoder(Variable(batch))
            #     # plot_batch(batch_hat.data)
            #     losses = (loss.item(), recon_loss.item(), cluster_loss.item())
            #     print('Trn Loss: %.3f [Recon Loss %.3f, Cluster Loss %.3f]' % losses)
        
        # update centroids based on assignments from autoencoders
        # centroid_means = centroid_sums / (centroid_counts[:, None] + 1)
        return centroid_means.detach(), cluster_assignments.detach()
    
    def evaluate(encoder, decoder, loader):
        for X, y in loader:
            X_var, y_var = Variable(X), Variable(y)
            s = encoder(X_var)
            X_hat = decoder(s)
            # do something
