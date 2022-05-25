import RO_DNN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
import random as r
import gurobipy as gp
from gurobipy import GRB
import csv
import torch
import sys
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import rsome as rso
from rsome import ro
from rsome import msk_solver as my_solver 

from rsome import ro
from rsome import grb_solver as grb
import rsome as rso
import numpy as np
import os
from datetime import date

datadir="scripts/selectiondata16/"
analysisdir = "scripts/results/"

t=int(sys.argv[1])
l=int(sys.argv[2])
s=int(sys.argv[3])
'''
t=10
l=10
s=10
'''

def testPolicyVaR_new(returns,delta):
    tmp = sorted(returns)
    VaR=tmp[int(np.floor(delta*len(returns))) ]
    
    return VaR
    
    # cvar_95 = [ i for i in tmp if i >= VaR]

    # return sum(cvar_95) / float(len(cvar_95))



def backtest_policy(val,VaR):
    x=[1 for i in val if i>VaR]
    return np.sum(x)/len(val)

def getWCdist(m):
    # Returns a uniform distribution over the extreme points of [-1, 1]^m
    # When calling [vals,ps] = getWCdist(m)
    #    m: size of the random vector
    #    vals : m x N matrixarray of values that the distribution takes
    #    ps : array of probability associated to each value

    if (m==1):
        vals=np.array([-1, 1]).reshape(1,2)
        ps = np.array([0.5, 0.5])
    else:
        (vals0,ps0)=getWCdist(m-1)
        n=np.size(vals0,axis=1)
        v=np.concatenate((-1*np.ones(n), np.ones(n)))
        val=np.concatenate((vals0,vals0),axis=1)
        vals=np.concatenate(([v],val),axis=0)   
        ps = np.concatenate((0.5*ps0, 0.5*ps0))
    return (vals, ps)

def testPolicyVaR(x,returns,VaReps):
    tmp = sorted(x@returns)
    VaR=-tmp[int(np.floor(VaReps*returns.shape[1]))]
    return VaR


fileName = datadir+'test-'+str(t)+'-'+str(l)+'-'+str(s)+'.txt'
if '.csv' in fileName:
    X_test_df=pd.read_csv(fileName) #pd.read_csv(fileName)  
    X_test = X_test_df.to_numpy()
else:
    X_test = np.genfromtxt(fileName, delimiter=',')
    X_test_df=pd.DataFrame(X_test)  #scaler(pd.DataFrame(X_test))
# side_cols=[i for i in X_test_df.columns if '_SI' in i]
# main_cols=[i for i in X_test_df.columns if '_SI' not in i]
X_test_side=X_test_df.iloc[:,0:2].to_numpy()
X_test_main=X_test_df.iloc[:,2:4].to_numpy()
fileName = datadir+'train-'+str(t)+'-'+str(l)+'-'+str(s)+'.txt'
if '.csv' in fileName:
    X_train_df=pd.read_csv(fileName) #pd.read_csv(fileName) 
    # X_train_df = X_train_df.sample(n = 20)
    print(X_train_df.shape)
    X_train = X_train_df.to_numpy()
else:
    X_test = np.genfromtxt(fileName, delimiter=',')
    X_train_df=pd.DataFrame(X_test)  #scaler(pd.DataFrame(X_test))
    
# X_train_df=pd.DataFrame(df)
X_train_side=X_train_df.iloc[:,0:2].to_numpy()
X_train_main=X_train_df.iloc[:,2:4].to_numpy()

df=X_train_df.copy()

center = X_train_main.mean(axis=0)

N=X_train_main.shape[1]
Rs=np.transpose(X_train_main)
RsTest=np.transpose(X_test_main)
(nStocks,nMonths)=Rs.shape

mu=np.mean(Rs, axis=1)
tmp = Rs - mu.reshape(-1,1)@np.ones((1,nMonths))
tmp_max = abs(tmp).max(1)
Zs = np.diag(1/tmp_max)@tmp
P = np.diag(tmp_max)
#getWCdist allows us to create a set of worst-case scenarios for returns
# tmp=P@getWCdist(nStocks)[0]
# WCRETURNS = mu.reshape(-1,1)@np.ones((1,tmp.shape[1]))+tmp

# #solve neural network

df_space_min=df.iloc[:,2:4].min()*1.5-df.iloc[:,2:4].max()*0.25
df_space_max=df.iloc[:,2:4].max()*1.5
df_space=[]

# i=df_space_min.values[0]
# j=df_space_min.values[1]
# while True:
#     df_space.append([i,j])
#     i+=0.5
#     if i>=df_space_max.values[0]:
#         break
# while True:
#     df_space.append([i,j])
#     j+=0.5
#     if j>=df_space_max.values[1]:
#         break
    
    
# for i in range(int(df_space_min.values[0]),int(df_space_max.values[0]),5):
#     for j in range(int(df_space_min.values[1]),int(df_space_max.values[1]),5):
#         df_space.append([i,j])

for a in np.linspace(df_space_min.values[0]-5, df_space_max.values[0]+5, num=250):
    for b in np.linspace(df_space_min.values[1]-5,df_space_max.values[1]+5, num=250):
        df_space.append([a,b])
        
df_space_all=pd.DataFrame(df_space)
for quantil in [0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]:
    # Calibrating the ellipsoid set
    tmp=np.sort(np.linalg.norm(Zs,axis=0))
    # gamma=tmp[int(np.ceil((1-quantil)*len(tmp)))-1]
    # gamma =np.quantile(tmp,quantil)
    gamma = testPolicyVaR_new(tmp,quantil)
    
    print('Calibrating the ellipsoid set: gamma={0:0.6f}'.format(gamma))
    
    
    model = ro.Model('solveRobustPortfolio_Ellipsoidal')
    x=model.dvar(nStocks)
    z=model.rvar(nStocks)
    EllipsoidSet=(rso.norm(z,2)<=gamma)
    model.minmax(-(mu+P@z)@x,EllipsoidSet)
    # EllipsoidSet=(rso.norm(z - center,1)<=gamma)
    # model.minmax((z)@x,EllipsoidSet)
    # model.st(x<=1)
    model.st(sum(x)==1)
    model.st(x>=0)
    model.st(x <= 1.0)
    model.solve(grb)
    
    dist=np.linalg.norm(X_train_main-mu,axis=1)
    R=testPolicyVaR_new(dist,quantil)
    df[str(quantil)]=list(dist<=testPolicyVaR_new(dist,quantil))
    
    #in the xy space
    dist_space=np.linalg.norm(df_space_all.iloc[:,0:2]-mu,axis=1)
    df_space_all[str(quantil)]=list(dist_space<=R)
    
    
    obj_Ellipsoid=model.get()
    xx_Ellipsoid=x.get() #get optimal portfolio
    xx_Ellipsoid = np.round(100*xx_Ellipsoid, 2)
    
    print('Estimated VaR is {0:0.4f}'.format(obj_Ellipsoid))
    print(quantil, list(xx_Ellipsoid))
    # print('VaR from 2000 to 2009 is {0:0.4f}'.format(testPolicyVaR(xx_Ellipsoid, Rs,quantil)))
    # print('VaR from 2010 to 2014 is {0:0.4f}'.format(testPolicyVaR(xx_Ellipsoid, RsTest,quantil)))

    avg, maximum, vals = RO_DNN.evaluateSolution(xx_Ellipsoid,X_test_main)
     
    csvFile = open(analysisdir+"port_ellipsoid.csv", 'a')
    out = csv.writer(csvFile,delimiter=",", lineterminator = "\n")
    row = [s,t,l,quantil]
    row.append(gamma)   #Radius
    row.append(obj_Ellipsoid)
    row.append(xx_Ellipsoid)
    row.append(avg)
    row.append(maximum)
    row.append(testPolicyVaR_new(vals,quantil))
    row.append(backtest_policy(vals,-1*obj_Ellipsoid))
    print(quantil, testPolicyVaR_new(vals,quantil))
    for i in range(len(vals)):
        row.append(vals[i])
    out.writerow(row)
    csvFile.close()


df['SSIZE']=s
df.to_csv(analysisdir+"ellipsoid_plot_data.csv",index=False)
df_space_all.to_csv(analysisdir+"ellipsoid_plot_data_new.csv",index=False)

           
