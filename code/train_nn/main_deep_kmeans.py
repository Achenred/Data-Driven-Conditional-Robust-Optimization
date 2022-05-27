import click
import torch
import logging
import random
import numpy as np
import pandas as pd
import csv
import sys
import os
from sklearn import preprocessing
import math
import torch.nn.functional as F
from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD_Dkmeans import DeepSVDD
from deepSVDD_Dkmeans import AE
from datasets.main import load_dataset
from soft_assign import soft_assign
from torch.autograd import Variable
from sklearn.cluster import KMeans
import os, shutil

################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'cifar10', 'mine','portfolio']))
@click.argument('net_name', type=click.Choice(['deep_kmeans_AE','port_soft_assign_AE1','port_soft_assign_AE2','port_soft_assign_AE3','port_soft_assign_AE','port_soft_assign','soft_assign','soft_assign_AE1','soft_assign_AE2','soft_assign_AE3','mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'mine_net2', 'mine_net3', 'mine_net4', 'mine_net5', 'mine_sp','mine_gen']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('test_path', type=click.Path(exists=True))
# @click.argument('side_info_path', type=click.Path(exists=True),default=None)
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
              
#@click.option('--test_path', type=click.Path(exists=True), default=None,
#              help='Test file path.')              
#@click.option('--train_path', type=click.Path(exists=True), default=None,
#              help='Test file path.')              
              
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--n_clusters', type=int, default=1,
                help='Select the number of clusters')
@click.option('--objective', type=click.Choice(['one-class', 'soft-boundary']), default='one-class',
              help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
@click.option('--year', type=int, default=2017, help='year')
@click.option('--beta', type=float, default=0.1, help='conditional network assignment control variable.')
@click.option('--lmbda', type=float, default=1, help='distance scaling factor for soft kmeans loss')
@click.option('--alpha', type=float, default=0.1, help='weight for the conditional loss')
@click.option('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for Deep SVDD network training.')
@click.option('--lr', type=float, default=0.001,help='Initial learning rate for Deep SVDD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=(9000,), multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
@click.option('--pretrain', type=bool, default=True,
              help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=0, multiple=False,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
def main(dataset_name, net_name, xp_path, data_path, test_path,load_config, load_model,n_clusters, objective, year,beta,lmbda, alpha, nu, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain, ae_optimizer_name, ae_lr,
         ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, normal_class):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    lr=cfg.settings['lr']
    n_classes=cfg.settings['n_clusters']
    alpha=cfg.settings['alpha']
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log_'+dataset_name+'_'+net_name+'_'+str(lr)+'_'+str(n_classes)+'.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print configuration
    logger.info('Deep SVDD objective: %s' % cfg.settings['objective'])
    logger.info('Nu-paramerter: %.2f' % cfg.settings['nu'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)
    
    
    
    # Load data
    #separate side info from the data
    # data=pd.read_csv(data_path, sep=",", header=None)
    # side_info_train = data.iloc[:, 0:2]
    # data.iloc[:, 2:].to_csv(data_path,index=False,header=False)
    dataset = load_dataset(dataset_name, data_path, test_path, normal_class,xp_path)
    side_dim=dataset.train_set.side_train_set.size(1)
    main_dim=dataset.train_set.data_train_set.size(1)
    logger.info('input dim of network is: %s'%dataset.train_set.data_train_set.size(0))
    out_dim=5 #3 for simulated
    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(cfg.settings['objective'],cfg.settings['n_clusters'], cfg.settings['beta'], cfg.settings['alpha'],cfg.settings['nu'])
    k_means_AE = AE(net_name,main_dim,side_dim,out_dim,cfg.settings['objective'],cfg.settings['n_clusters'], cfg.settings['beta'], cfg.settings['alpha'],cfg.settings['nu'])
    deep_SVDD.set_network(net_name,main_dim,side_dim,out_dim,cfg.settings['beta'],cfg.settings['lmbda'],cfg.settings['n_clusters'])


    logger.info('cond_network: %s' % deep_SVDD.encoder)
    logger.info('cond_network: %s' % deep_SVDD.decoder)
    logger.info('cond_network: %s' % deep_SVDD.soft_KMeans)
    logger.info('main_network: %s' % deep_SVDD.net_main)
    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        deep_SVDD.load_model(model_path=load_model, load_ae=True)
        logger.info('Loading model from %s.' % load_model)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
        logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

        # Pretrain model on dataset (via autoencoder)
        deep_SVDD.pretrain(dataset,
                           optimizer_name=cfg.settings['ae_optimizer_name'],
                           lr=cfg.settings['ae_lr'],
                           n_epochs=cfg.settings['ae_n_epochs'],
                           lr_milestones=cfg.settings['ae_lr_milestone'],
                           batch_size=cfg.settings['ae_batch_size'],
                           weight_decay=cfg.settings['ae_weight_decay'],
                           device=device,
                           n_jobs_dataloader=n_jobs_dataloader)

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    

    
    # Train model on dataset
    # logger.info(assignments)
    centroids = Variable(k_means_AE.centroid_init(dataset,n_classes, out_dim).data)
    # logger.info(centroids)
    for _ in range(5):
        centroids, assignments = k_means_AE.train(dataset,centroids)
        # logger.info(centroids)
        # logger.info(torch.sum(assignments,axis=0))
    logger.info(assignments)
    list_1=assignments
    
    result= r'path/deepK/'+dataset_name+'_'+net_name+'_alpha'+str(alpha)+'/'
    os.makedirs(result, exist_ok=True)
    for filename in os.listdir(result):
        file_path = os.path.join(result, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
            
    logger.info(result)
    for k in range(n_classes):
        logger.info('Cluster : %g' %k)
        deep_SVDD.set_network(net_name,main_dim,side_dim,out_dim,cfg.settings['beta'],cfg.settings['lmbda'],cfg.settings['n_clusters'])
        deep_SVDD.train(k,list_1,dataset,
                    optimizer_name=cfg.settings['optimizer_name'],
                    lr=cfg.settings['lr'],
                    n_epochs=cfg.settings['n_epochs'],
                    lr_milestones=cfg.settings['lr_milestone'],
                    batch_size=cfg.settings['batch_size'],
                    weight_decay=cfg.settings['weight_decay'],
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader)
        # logger.info(deep_SVDD.trainer.assignment.numpy())

        # logger.info(itertools.islice(deep_SVDD.net_main.state_dict(), 2))
        
        
        #normalize
        n_samples = 0
        c = torch.zeros(deep_SVDD.net_main.rep_dim,device=device)
        norm_data=pd.DataFrame()
        deep_SVDD.net_main.eval()
        n_samples=0
        cov=[0]
        with torch.no_grad():
            train_loader, _ = dataset.loaders(batch_size=batch_size, num_workers=n_jobs_dataloader)
            for data in train_loader:
                # get the inputs of the batch
                side,inputs, _ = data
                inputs_k=inputs[list_1==k]
                inputs_k = inputs_k.to(device)

                outputs = deep_SVDD.net_main(inputs_k)
                n_samples = inputs_k.shape[0]
                c = torch.sum(outputs, dim=0)
                cov=torch.cov(outputs.T)
        # logger.info(c)
        c /= n_samples
        c[(abs(c) < 0.01) & (c < 0)] = -0.01
        c[(abs(c) < 0.01) & (c >= 0)] = 0.01


        with open(result+'/c_'+str(k)+'.txt', 'w') as f:
           f.write(','.join(str(i) for i in deep_SVDD.c.numpy()))
        
        j=0
        # logger.info(deep_SVDD.net.state_dict())
        for item in deep_SVDD.net.state_dict():
            with open(result+'/W_'+str(int(k))+'_'+str(j)+'.txt', 'w') as f:
                for l in deep_SVDD.net.state_dict()[item]:
                    f.write(','.join(str(i) for i in l.numpy()))
                    f.write('\n')
                j+=1
                    
        np.set_printoptions(threshold=sys.maxsize)
        # print(cov)
        with open(result+'/cov_'+str(k)+'.txt', 'w') as f:
            for item in cov:
                f.write(','.join(str(i) for i in item.numpy()))
                f.write('\n')
                    
        def scaler(df):
            x = df #returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            return x_scaled
        
    with torch.no_grad():
        _, test_loader = dataset.loaders(batch_size=batch_size, num_workers=n_jobs_dataloader)
        for data in test_loader:
            side,_ , _= data
            # logger.info('embeddings: %s'%embeddings)
            kmeans = KMeans(n_clusters=n_classes,max_iter = 1, random_state=0).fit(side.detach().numpy())
            test_label=kmeans.labels_
        # logger.info(test_label)
 
        np.save(result+'/train_assignments', list_1)
        np.save(result+'/test_assignments', test_label)


if __name__ == '__main__':
    main()
