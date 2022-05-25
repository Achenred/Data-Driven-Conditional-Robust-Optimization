import RO_DNN as RO
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

datadir="scripts/selectiondata16/"
resultdir="path/simulated_mark/"
analysisdir = "scripts/results/"

t=int(sys.argv[1])
l=int(sys.argv[2])
s=int(sys.argv[3])
'''
t=2
l=1
s=1500
'''
def testPolicyVaR(returns,delta):
    tmp = sorted(returns)
    VaR=tmp[int(np.floor(delta*len(returns)))]
    return VaR


def backtest_policy(val,VaR):
    x=[1 for i in val if i>VaR]
    return np.sum(x)/len(val)

fileName = datadir+'test-'+str(t)+'-'+str(l)+'-'+str(s)+'.txt'
X_test = np.genfromtxt(fileName, delimiter=',')
X_test_df=pd.DataFrame(X_test)  #scaler(pd.DataFrame(X_test))
X_test_side=X_test_df.iloc[:,0:2].to_numpy()
X_test_main=X_test_df.iloc[:,2:4].to_numpy()
fileName = datadir+'train-'+str(t)+'-'+str(l)+'-'+str(s)+'.txt'
X_train = np.genfromtxt(fileName, delimiter=',')
X_train_df=pd.DataFrame(X_train)    #scaler(pd.DataFrame(X_train))
X_train_side=X_train_df.iloc[:,0:2].to_numpy()
X_train_main=X_train_df.iloc[:,2:4].to_numpy()
N=X_train_main.shape[1]
df=X_test_df.copy()
# #solve neural network
E = 2
L = 2
fileName = resultdir+"c.txt"
c0 = np.genfromtxt(fileName, delimiter=',')

listW = []
dimLayers = []

for F in range(0,L,1):
    fileName = resultdir+"W_0_"+str(F)+".txt"
    listW.append(np.genfromtxt(fileName, delimiter=','))
    dimLayers.append(listW[F].shape[0])
    
N=listW[0].shape[1]


maxScenEntry = max(np.amax(X_train),np.amax(X_test))
maxEntry = max(np.amax(X_train),np.amax(X_test))
M=[]
for i in range(0,L,1):
    rowSums = np.sum(np.absolute(listW[i]),axis=1)
    M.append(maxEntry*np.amax(rowSums))
    maxEntry = maxEntry*np.amax(rowSums)
    
for delta in [0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]:
       
    
    
    R_all, R,sigmas,lb,ub = RO.getRadiiDataPoints(L,c0,X_train_main,listW, 0, delta)
    print('Radius: {:.8f}'.format(R))
    R_all_test, _,_,_,_ = RO.getRadiiDataPoints(L,c0,X_test_main,listW, 0, delta)   
    df[str(delta)]=list(R_all_test<=R)
    start = time.time()
    p=1
    obj, x = RO.solveRobustSelection(p,N,L,dimLayers,c0, R, listW, M, lb, ub,0, False, 0,sigmas)
    print('obj: {:.8f}\t Policy: {}'.format(obj,x))
    end = time.time()
    t = end-start
        
    avg, maximum, vals = RO.evaluateSolution(x,X_test_main)

    csvFile = open(analysisdir+"simulation_kmeans_mark.csv", 'a')
    out = csv.writer(csvFile,delimiter=",", lineterminator = "\n")
    row = [delta]
    row.append(R)
    row.append(obj)
    row.append(x)
    row.append(avg)
    row.append(maximum)
    row.append(testPolicyVaR(vals,delta))
    row.append(backtest_policy(vals,obj))
    out.writerow(row)
    csvFile.close()

df['SSIZE']=s
df.to_csv(analysisdir+"DDDRO_plot_data.csv")



