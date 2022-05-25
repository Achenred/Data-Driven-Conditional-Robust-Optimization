#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 19:44:03 2022

@author: nbandi
"""

import numpy as np
import pandas as pd
#https://stackoverflow.com/questions/38713746/python-numpy-conditional-simulation-from-a-multivatiate-distribution
mean = np.array([1, 2, 0, 4])
cov = np.array(
    [[ 1.0,  0.0,  0.3, -0.1], 
     [ 0.0,  -1.0,  0.1, -0.2], 
     [ 0.3,  0.1,  1.0, 2.0], 
     [-0.1, -0.2, 2.0,  1.0]])  # diagonal covariance

c11 = cov[0:2, 0:2] # Covariance matrix of the dependent variables
c12 = cov[0:2, 2:4] # Custom array only containing covariances, not variances
c21 = cov[2:4, 0:2] # Same as above
c22 = cov[2:4, 2:4] # Covariance matrix of independent variables

m1 = mean[0:2].T # Mu of dependent variables
m2 = mean[2:4].T # Mu of independent variables

conditional_data_1 = np.random.multivariate_normal(m2, c22, 10000)

conditional_mu = m1 + c12.dot(np.linalg.inv(c22)).dot((conditional_data_1 - m2).T).T
conditional_cov = np.linalg.inv(np.linalg.inv(cov)[0:2, 0:2])

dependent_data_1 = np.array([np.random.multivariate_normal(c_mu, conditional_cov, 1)[0] for c_mu in conditional_mu])

mean = np.array([5, 5, 4, 0])
cov = np.array(
    [[ 1.0,  0.0,  0.3, -0.1], 
     [ 0.0,  -1.0,  0.1, -0.2], 
     [ 0.3,  0.1,  1.0, 0.0], 
     [-0.1, -0.2, 0.0,  1.0]])  # diagonal covariance

c11 = cov[0:2, 0:2] # Covariance matrix of the dependent variables
c12 = cov[0:2, 2:4] # Custom array only containing covariances, not variances
c21 = cov[2:4, 0:2] # Same as above
c22 = cov[2:4, 2:4] # Covariance matrix of independent variables

m1 = mean[0:2].T # Mu of dependent variables
m2 = mean[2:4].T # Mu of independent variables

conditional_data_2 = np.random.multivariate_normal(m2, c22, 10000)

conditional_mu = m1 + c12.dot(np.linalg.inv(c22)).dot((conditional_data_2 - m2).T).T
conditional_cov = np.linalg.inv(np.linalg.inv(cov)[0:2, 0:2])

dependent_data_2 = np.array([np.random.multivariate_normal(c_mu, conditional_cov, 1)[0] for c_mu in conditional_mu])

pis = np.array([0.5, 0.5])
acc_pis = [np.sum(pis[:i]) for i in range(1, len(pis)+1)]
assert np.isclose(acc_pis[-1], 1)
df_dep=[]
df_con=[]
clus=[]
for ind in range(10000):
    # sample uniform
    r = np.random.uniform(0, 1)
    # select Gaussian
    k = 0
    for i, threshold in enumerate(acc_pis):
        if r < threshold:
            k = i
            break
    clus.append(k)
    if k==0:
        df_dep.append(dependent_data_1[ind])
        df_con.append(conditional_data_1[ind])
        
    else:
        df_dep.append(dependent_data_2[ind])
        df_con.append(conditional_data_2[ind])
df=pd.concat([pd.DataFrame(df_con),pd.DataFrame(df_dep)],axis=1)

df.iloc[:500,:].to_csv('scripts/selectiondata16/train-10-10-10.txt',header=None,sep=',',index=False)
df.iloc[500:,:].to_csv('scripts/selectiondata16/test-10-10-10.txt',header=None,sep=',',index=False)

'''
import numpy as np
#https://medium.com/analytics-vidhya/sampling-from-gaussian-mixture-models-f1ab9cac2721

def inv_sigmoid(values):
    return np.log(values/(1-values))

mus = [np.array([0, 1]), np.array([7, 1])] #, np.array([-3, 3])
covs = [np.array([[1, 2], [2, 1]]), np.array([[1, 0], [0, 1]])] #, np.array([[10, 1], [1, 0.3]])
pis = np.array([0.5, 0.5])
acc_pis = [np.sum(pis[:i]) for i in range(1, len(pis)+1)]
assert np.isclose(acc_pis[-1], 1)
df=[]
for i in range(10000):
    # sample uniform
    r = np.random.uniform(0, 1)
    # select Gaussian
    k = 0
    for i, threshold in enumerate(acc_pis):
        if r < threshold:
            k = i
            break
    selected_mu = mus[k]
    selected_cov = covs[k]
    
    
    # sample from selected Gaussian
    lambda_, gamma_ = np.linalg.eig(selected_cov)
    dimensions = len(lambda_)
    # sampling from normal distribution
    y_s = np.random.uniform(0, 1, size=(dimensions*1, 2))
    x_normal = np.mean(inv_sigmoid(y_s), axis=1).reshape((-1, dimensions))
    # transforming into multivariate distribution
    x_multi = (x_normal*lambda_) @ gamma_ + selected_mu
    df.append(x_multi[0])
'''





