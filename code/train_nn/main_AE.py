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
import os, shutil
from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD_AE import DeepSVDD
from datasets.main import load_dataset
from soft_assign import soft_assign

    

################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['generated', 'mine','portfolio']))
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
@click.option('--eps', type=float, default=1.0, help='weight for the encoder decoder network')
@click.option('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for Deep SVDD network training.')
@click.option('--lr', type=float, default=0.001,help='Initial learning rate for Deep SVDD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=(200,), multiple=True,
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
@click.option('--ae_lr_milestone', type=int, default=200, multiple=False,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
def main(dataset_name, net_name, xp_path, data_path, test_path,load_config, load_model,n_clusters, objective, year,beta,lmbda, alpha, eps, nu, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain, ae_optimizer_name, ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, normal_class):
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
    eps=cfg.settings['eps']
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
    # logger.info('side input dim of network is: %s'%side_dim)
    # logger.info('main input dim of network is: %s'%main_dim)
    if dataset_name=="portfolio":
        out_dim=5 #3 for simulated
    else:
        out_dim=2
    
    from sklearn.datasets import load_iris
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from k_means_constrained import KMeansConstrained
    
    X = dataset.train_set.side_train_set
    # if cfg.settings['n_clusters']==0:
    #     sil_score_max = -1 #this is the minimum possible score
    #     best_n_clusters = 0
    #     for n_clusters in range(2,5):
    #       clf = KMeansConstrained(n_clusters=n_clusters, size_min=100, random_state=0)
    #       clf.fit_predict(np.array(X))
    #       labels = clf.labels_
    #       sil_score = silhouette_score(X, labels)
    #       if sil_score > sil_score_max:
    #         sil_score_max = sil_score
    #         best_n_clusters = n_clusters

    #     n_clusters=best_n_clusters
    # else:
    n_clusters=cfg.settings['n_clusters']

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(cfg.settings['objective'],n_clusters, cfg.settings['beta'], cfg.settings['alpha'],cfg.settings['nu'])
    deep_SVDD.set_network(net_name,main_dim,side_dim,out_dim,cfg.settings['beta'],cfg.settings['lmbda'],n_clusters)
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
    deep_SVDD.train(dataset,
                    optimizer_name=cfg.settings['optimizer_name'],
                    lr=cfg.settings['lr'],
                    n_epochs=cfg.settings['n_epochs'],
                    lr_milestones=cfg.settings['lr_milestone'],
                    batch_size=cfg.settings['batch_size'],
                    weight_decay=cfg.settings['weight_decay'],
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader)
    logger.info(deep_SVDD.trainer.assignment.numpy())
    
    ##print network
    #save center and weights
    year=cfg.settings['year']
    # logger.info(itertools.islice(deep_SVDD.net_main.state_dict(), 2))
    if dataset_name=='mine':
        result= r'path/mine_'+net_name+'_'+str(lr)+'_'+str(n_classes)+'_alpha'+str(alpha)+'/'
    else:
        result= r'path/'+str(year)+dataset_name+'_'+net_name+'_'+str(lr)+'_'+str(n_classes)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_eps'+str(eps)+"/"
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
    #normalize
    n_samples = 0
    c = torch.zeros(deep_SVDD.net_main.rep_dim,device=device)
    
    norm_data=pd.DataFrame()
    deep_SVDD.net_main.eval()
    c = [torch.zeros(deep_SVDD.net_main.rep_dim)]*n_clusters
    sig = [torch.zeros((2, 3), device = device)]*n_clusters
    n_samples=[0]*n_clusters
    with torch.no_grad():
        train_loader, _ = dataset.loaders(batch_size=batch_size, num_workers=n_jobs_dataloader)
        for data in train_loader:
            # get the inputs of the batch
            side,inputs, _ = data
            assignments = deep_SVDD.trainer.assignment
            
            inputs = inputs.to(device)
            outputs = deep_SVDD.net_main(inputs.float())
            
            for k in range(n_clusters):
                n_samples[k] = outputs[k].shape[0]
                c[k] = torch.transpose(torch.matmul(torch.transpose(outputs[k],0,1),assignments[k].float().view(outputs[k].shape[0],1)),0,1)[0]/assignments[k].sum()
                
                cov_mat = torch.cov(outputs[k].T, aweights= assignments[k])
                sig[k] = torch.inverse(cov_mat)
    logger.info(c)

    for k in range(n_clusters):
        c[k][(abs(c[k]) < 0.01) & (c[k] < 0)] = -0.01
        c[k][(abs(c[k]) < 0.01) & (c[k] >= 0)] = 0.01
   
    #save center and weights
    with open(result+'/c.txt', 'w') as f:
        # f.write(','.join(str(i) for i in c))
        f.write(','.join(str(i) for i in deep_SVDD.c))
    for k in range(n_clusters):
        with open(result+'/c_'+str(k)+'.txt', 'w') as f:
            logger.info(c[k])
            # f.write(','.join(str(i) for i in c[k].numpy()))
            f.write(','.join(str(i) for i in deep_SVDD.c[k].detach().numpy()))
            
    for k in range(n_clusters):
        with open(result+'/cov_'+str(k)+'.txt', 'w') as f:
            for item in deep_SVDD.cov_mat[k]:
                f.write(','.join(str(i) for i in item))
                f.write('\n')

    n=len( deep_SVDD.net_main.state_dict())/n_clusters
    for i in range(n_clusters):
        j=0
        for k in dict(list(deep_SVDD.net_main.state_dict().items())[int(i*n):int((i+1)*n)]):
            with open(result+'/W_'+str(int(i))+'_'+str(j)+'.txt', 'w') as f:
                for item in deep_SVDD.net_main.state_dict()[k]:
                    f.write(','.join(str(i) for i in item.numpy()))
                    f.write('\n')
            j+=1
    np.set_printoptions(threshold=sys.maxsize)
    
    def scaler(df):
        x = df #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = 100*min_max_scaler.fit_transform(x)
        return x_scaled
    
    with torch.no_grad():
        _, test_loader = dataset.loaders(batch_size=batch_size, num_workers=n_jobs_dataloader)
        for data in test_loader:
            side,inputs, _= data
            # logger.info('embeddings: %s'%embeddings)
            embeddings = deep_SVDD.encoder(side) #net_cond(side)
            _, test_cluster_assignments = deep_SVDD.soft_KMeans(embeddings, deep_SVDD.trainer.centroids)
            
    test_cluster_assignments=torch.transpose(test_cluster_assignments,0,1)
    logger.info(test_cluster_assignments.shape)
    logger.info(inputs.shape)
    with torch.no_grad():
        _, test_loader = dataset.loaders(batch_size=batch_size, num_workers=n_jobs_dataloader)
        for data in test_loader:
            side,inputs, _= data
            # logger.info(side)
            # logger.info('embeddings: %s'%embeddings)
            embeddings = deep_SVDD.encoder(side) #net_cond(side)
            _, test_cluster_assignments = deep_SVDD.soft_KMeans(embeddings, deep_SVDD.trainer.centroids)
            # test_cluster_assignments = deep_SVDD.trainer.cluster_responsibilities(embeddings,deep_SVDD.trainer.centroids, beta)
    test_cluster_assignments=torch.transpose(test_cluster_assignments,0,1)
    logger.info(test_cluster_assignments)
    
    np.save(result+'/train_assignments', deep_SVDD.trainer.assignment.numpy())
    np.save(result+'/test_assignments', test_cluster_assignments.numpy())
   
if __name__ == '__main__':
    main()
