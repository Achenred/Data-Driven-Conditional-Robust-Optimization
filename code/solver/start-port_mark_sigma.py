import RO as RO
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
import os
from datetime import date


year=int(sys.argv[1])
lr=float(sys.argv[2])
alp=int(sys.argv[3])
seed=int(sys.argv[4])

'''
year = 2013
lr= 0.01
'''
datadir="path/data/"
resultdir="path/portfolio_mark/"
analysisdir = "scripts/results/"


def scaler(df):
    x=df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = 100*pd.DataFrame(x_scaled)
    return df


def testPolicyVaR(returns,delta):
    tmp = sorted(returns)
    VaR=tmp[int(np.floor(delta*len(returns)))]
    return VaR

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
        
    if "DDDRO" not in os.listdir(filename+str(today)):
        os.mkdir(filename+str(today)+str("/DDDRO/"))

    filename_updated = filename+str(today)+str("/DDDRO/") 
    filename_updated = filename_updated + "/" + str(len(os.listdir(filename_updated))) + "/"
    os.mkdir(filename_updated)
    
    return filename_updated

def bisection(f,a,b,N):

    if f(a)*f(b) >= 0:
        print("Bisection method fails.")
        return None
    a_n = a
    b_n = b
    for n in range(1,N+1):
        m_n = (a_n + b_n)/2
        f_m_n = f(m_n)
        if f(a_n)*f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n)*f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            # print("Found exact solution.")
            return m_n
        else:
            # print("Bisection method fails.")
            return None
    return (a_n + b_n)/2

def get_alpha(df, mu, c, radius, delta,sig = None):
    alpha_list = []
        
    for i in range(len(df)):
        if sig is not None:
            a=df[i].reshape(1,-1) - mu.reshape(1,-1)*(1-0.1) - c.reshape(1,-1)*(0.1)
            b=df[i].reshape(-1,1) - (1-0.1)*mu.reshape(-1,1) - (0.1)*c.reshape(-1,1)
            f = lambda x:  np.matmul(np.matmul((1/x)*(df[i].reshape(1,-1) - mu.reshape(1,-1)) +mu.reshape(1,-1) - c.reshape(1,-1),sig), ( (1/x)*(df[i].reshape(-1,1) - mu.reshape(-1,1)) - c.reshape(-1,1)))[0][0] - radius**2
            
        else:
            f = lambda x:  np.linalg.norm((1/x)*(df[i] - mu) + mu - c, 2) - radius
        approx_alpha = bisection(f,0.0000001, 1,5000)  #0.0000001
        alpha_list.append(approx_alpha)
    
        
    alpha_list = list(filter(None, alpha_list))
    alpha_sorted = sorted(alpha_list)
    for alp in alpha_sorted:
        distlist = []
        if sig is not None:
            f = lambda x:  np.matmul(np.matmul((1/x)*(df[i].reshape(1,-1) - mu.reshape(1,-1)) +mu.reshape(1,-1) - c.reshape(1,-1),sig), ( (1/x)*(df[i].reshape(-1,1) - mu.reshape(-1,1)) - c.reshape(-1,1)))[0][0] - radius**2
        else:
            f = lambda x:  np.linalg.norm((1/x)*(df[i] - mu) + mu - c, 2) - radius
        
        for i in range(len(df)):
            distlist.append(np.where(f(alp) <= 0, 1, 0).item())
                        
        if (sum(distlist) < int(delta*len(distlist))):
            
            continue
        else:
            break

    return alp
  
    
def get_alpha_for_convex_hull(X_train_main,X_proj, mu, c,sig, radius, delta):
        
    center = X_train_main.mean(axis=0)
        
    N=X_train_main.shape[1]
    Rs=np.transpose(X_train_main)
    (nStocks,nMonths)=Rs.shape

    mu_temp=np.mean(Rs, axis=1)

 
    tmp = Rs - mu_temp.reshape(-1,1)@np.ones((1,nMonths))
    tmp_max = abs(tmp).max(1)

    # Zs = np.diag(1/tmp_max)@tmp
    P = np.diag(tmp_max)
    R_all = np.linalg.norm(tmp,axis=0)
    tmp=np.sort(np.linalg.norm(tmp,axis=0))
    gamma = testPolicyVaR_new(tmp,delta)
    
    outside  = [i for i in range(len(R_all)) if R_all[i] > gamma]
    outside_points = np.take(X_proj, outside, 0)
    
    inside  = [i for i in range(len(R_all)) if R_all[i] <= gamma]
    inside_points = np.take(X_proj, inside, 0)
    
    
    
    alp = get_alpha(inside_points, mu, c, radius, delta,sig)
    
    return alp
    

# analysisdir = path_generator(analysisdir)

for y in [year]:
        
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
            X_train = X_train_df.to_numpy()
        else:
            X_test = np.genfromtxt(fileName, delimiter=',')
            X_train_df=pd.DataFrame(X_test)  #scaler(pd.DataFrame(X_test))

        X_train_side=X_train_df[side_cols].to_numpy()
        X_train_main=scaler(X_train_df[main_cols]).to_numpy()
        
        N=X_train_main.shape[1]
        mu = np.array(X_train_main.mean(axis=0)).reshape((1, N))

        # #solve neural network
        
        for quantil in [0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]:
            E = 2
            L = 2
            fileName = resultdir+"c.txt"
            c0 = np.genfromtxt(fileName, delimiter=',')
            
            fileName = resultdir+"cov.txt"
            sig=np.genfromtxt(fileName, delimiter=',')
            sig_inv = np.linalg.inv(sig)
            sig = sig_inv
            

            listW = []
            dimLayers = []
            
            for F in range(0,L,1):
                fileName = resultdir+"W_0_"+str(F)+".txt"
                listW.append(np.genfromtxt(fileName, delimiter=','))
                dimLayers.append(listW[F].shape[0])
                
                
            N=listW[0].shape[1]
            
            
            maxScenEntry = max(np.amax(X_train_main),np.amax(X_test))
            maxEntry = max(np.amax(X_train_main),np.amax(X_test))
            M=[]
            for i in range(0,L,1):
                rowSums = np.sum(np.absolute(listW[i]),axis=1)
                M.append(maxEntry*np.amax(rowSums))
                maxEntry = maxEntry*np.amax(rowSums)
               
            X_train_hat=[]
            for j in range(0,X_train_main.shape[0],1):
                outLayer = X_train_main[j,:]
                sigma = []
                for i in range(0,L-1,1):
                    sigmal = []
                    for l in range(listW[i].shape[0]):
                        if np.dot(listW[i],outLayer)[l] > 0:
                            sigmal.append(1)
                        else:
                            sigmal.append(0)
                    outLayer = np.maximum(np.zeros(listW[i].shape[0]),np.dot(listW[i],outLayer))
                    # print(i,outLayer)
                    sigma.append(sigmal)
                    
                # sigmas.append(sigma)
                outLayer = np.dot(listW[L-1],outLayer)
                X_train_hat.append(outLayer)
                
            #mu_hat
            outLayer = mu[0]
            for i in range(0,L-1,1):
                sigmal = []
                for l in range(listW[i].shape[0]):
                    if np.dot(listW[i],outLayer)[l] > 0:
                        sigmal.append(1)
                    else:
                        sigmal.append(0)
                outLayer = np.maximum(np.zeros(listW[i].shape[0]),np.dot(listW[i],outLayer))
            mu_hat = np.dot(listW[L-1],outLayer)
            # print(np.linalg.norm(mu_hat-c0))
            # print(mu,mu_hat,c0)
            
            X_train_temp = np.array(X_train_hat)
            X_train_assigned=X_train_main
            R_all, R,sigmas,lb,ub = RO.getRadiiDataPoints(L,c0,sig, X_train_main,listW, 0, quantil)
            print('Radius: {:.8f}'.format(R))
            if alp==0:
                _, Rand_R,_,_,_ = RO.getRadiiDataPoints(L,c0,sig,X_train_main,listW, 0, 0.99)
                alp = get_alpha_for_convex_hull(X_train_assigned,X_train_temp,mu_hat, c0,sig, Rand_R, quantil)
            else:
                mu_hat = None
                alp = -1000
            start = time.time()
            p=1

            obj, x = RO.solveRobustSelection(p,N,L,X_train_main,dimLayers,c0,mu_hat, sig, alp, R, listW, M, lb, ub,0, False, 0,sigmas)
            x = 100*np.round(x, 4)
            print('obj: {:.8f}\t Policy: {}'.format(obj,x))
            end = time.time()
            t = end-start

            avg, maximum, vals = RO.evaluateSolution(x,X_test_main)
            
            csvFile = open(analysisdir+"port_mark.csv", 'a')
            out = csv.writer(csvFile,delimiter=",", lineterminator = "\n")
            row = [seed,lr,year,quantil]
            row.append(R)
            row.append(obj)
            row.append(x)
            row.append(avg)
            row.append(maximum)
            row.append(testPolicyVaR(vals,quantil))
            row.append(backtest_policy(vals,obj))
            for i in range(len(vals)):
                row.append(vals[i])
            out.writerow(row)
            csvFile.close()
            
            csvFile = open(analysisdir+"consolidated.csv", 'a')
            out = csv.writer(csvFile,delimiter=",", lineterminator = "\n")
            row = [seed,year,0,1,0,quantil]
            row.append(testPolicyVaR(vals,quantil))
            out.writerow(row)
            csvFile.close()
           
