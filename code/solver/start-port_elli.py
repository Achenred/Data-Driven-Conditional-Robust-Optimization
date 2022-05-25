import RO
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

datadir="path/data/"
resultdir="path/portfolio_mark/"
# resultdir="/Users/nbandi/Dropbox/Mac/Desktop/PhD/Research/Robust/RO-DNN-master/path/simulated_mark/"
analysisdir = "scripts/results/"

year=int(sys.argv[1])
seed=int(sys.argv[2])

def scaler(df):
    x=df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = 100*pd.DataFrame(x_scaled)
    return df


def testPolicyVaR_new(returns,delta):
    tmp = sorted(returns)
    VaR=tmp[int(np.floor(delta*len(returns))) ]
    
    return VaR


def backtest_policy(val,VaR):
    x=[1 for i in val if i>VaR]
    return np.sum(x)/len(val)

def path_generator(filename):
    
    today = date.today()
    filename_updated = filename
    if str(today) not in os.listdir(filename):
        os.mkdir(filename+str(today)) 
        
    if "RO" not in os.listdir(filename+str(today)):
        os.mkdir(filename+str(today)+str("/RO/"))
    
    # if str(year) not in os.listdir(filename+str(today)+str("/IDCC/")):
    #     os.mkdir(filename+str(today)+str("/IDCC/")+str(year)+str("/"))
        
    filename_updated = filename+str(today)+str("/RO/") 
    filename_updated = filename_updated + "/" + str(len(os.listdir(filename_updated))) + "/"
    os.mkdir(filename_updated)
    
    return filename_updated
    
# analysisdir = path_generator(analysisdir)

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


fileName = datadir+'test_'+str(year)+'.csv'
if '.csv' in fileName:
    X_test_df=pd.read_csv(fileName) #pd.read_csv(fileName)  
    X_test = X_test_df.to_numpy()
else:
    X_test = np.genfromtxt(fileName, delimiter=',')
    X_test_df=pd.DataFrame(X_test)  #scaler(pd.DataFrame(X_test))
side_cols=[i for i in X_test_df.columns if '_SI' in i]
main_cols=[i for i in X_test_df.columns if '_SI' not in i]
X_test_side=X_test_df[side_cols].to_numpy()
X_test_main=X_test_df[main_cols].to_numpy()
fileName = datadir+'train_'+str(year)+'.csv'
if '.csv' in fileName:
    X_train_df=pd.read_csv(fileName) #pd.read_csv(fileName) 
    # X_train_df = X_train_df.sample(n = 20)
    print(X_train_df.shape)
    X_train = X_train_df.to_numpy()
else:
    X_test = np.genfromtxt(fileName, delimiter=',')
    X_train_df=pd.DataFrame(X_test)  #scaler(pd.DataFrame(X_test))
    

X_train_side=X_train_df[side_cols].to_numpy()
X_train_main=scaler(X_train_df[main_cols]).to_numpy()
# X_train_main=X_train_df[main_cols].to_numpy()
X_test_main= (X_test_df[main_cols]).to_numpy()

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


for quantil in [0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]:
    # Calibrating the ellipsoid set
    tmp=np.sort(np.linalg.norm(Zs,axis=0))
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

    obj_Ellipsoid=model.get()
    xx_Ellipsoid=x.get() #get optimal portfolio
    xx_Ellipsoid = np.round(100*xx_Ellipsoid, 2)
    
    print('Estimated VaR is {0:0.4f}'.format(obj_Ellipsoid))
    print(year, quantil, list(xx_Ellipsoid))
    # print('VaR from 2000 to 2009 is {0:0.4f}'.format(testPolicyVaR(xx_Ellipsoid, Rs,quantil)))
    # print('VaR from 2010 to 2014 is {0:0.4f}'.format(testPolicyVaR(xx_Ellipsoid, RsTest,quantil)))

    avg, maximum, vals = RO.evaluateSolution(xx_Ellipsoid,X_test_main)
     
    csvFile = open(analysisdir+"port_ellipsoid.csv", 'a')
    out = csv.writer(csvFile,delimiter=",", lineterminator = "\n")
    row = [seed,year,quantil]
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
    
    csvFile = open(analysisdir+"consolidated.csv", 'a')
    out = csv.writer(csvFile,delimiter=",", lineterminator = "\n")
    row = [seed,year,0,0,0,quantil]
    row.append(testPolicyVaR_new(vals,quantil))
    out.writerow(row)
    csvFile.close()
    
    
    
           
