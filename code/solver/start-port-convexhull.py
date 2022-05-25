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
from sklearn import preprocessing
import random
import logging
import os
from datetime import date


alpha=float(sys.argv[1])
lr=float(sys.argv[2])
n_class=int(sys.argv[3])
year=int(sys.argv[4])
val=int(sys.argv[5])
beta = float(sys.argv[6])
eps = float(sys.argv[7])
net = int(sys.argv[8])
alp_bisection=int(sys.argv[9])
seed=int(sys.argv[10])
'''
alpha= 0.5
lr= 0.01
n_class= 3
year=2012
beta =  0.1 # 0.11
eps = 0.5
net = 1
alp_bisection=0
'''

datadir="path/data/"
resultdir="path/"+str(year)+"portfolio_port_soft_assign_AE"+str(net)+'_'+str(lr)+'_'+str(n_class)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_eps'+str(eps)+"/"
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
    # data=pd.DataFrame(data)
    x=[1 for i in val if i>VaR]
    return np.mean(x),x

def get_assignnments(test_assignments):
    test_assignments_t = test_assignments.transpose()    
    numberList = range(len(test_assignments))
    test_ass_1 = []
    for i in range(len(test_assignments_t)):
        test_ass_1.append(random.choices(numberList, weights= test_assignments_t[i], k=1)[0])
    return np.array(test_ass_1)


def path_generator(filename):
    
    today = date.today()
    filename_updated = filename
    if str(today) not in os.listdir(filename):
        os.mkdir(filename+str(today)) 
        
    if "IDCC" not in os.listdir(filename+str(today)):
        os.mkdir(filename+str(today)+str("/IDCC/"))
    
    if str(year) not in os.listdir(filename+str(today)+str("/IDCC/")):
        os.mkdir(filename+str(today)+str("/IDCC/")+str(year)+str("/"))
        
    filename_updated = filename+str(today)+str("/IDCC/")+str(year)+str("/") 
    filename_updated = filename_updated + "/" + str(len(os.listdir(filename_updated))) + "/"
    os.mkdir(filename_updated)
    
    return filename_updated

def bisection(f,a,b,N):
    # print(f(a))
    # print(f(b))
    
    # import sys
    # sys.exit()
    
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
        
        approx_alpha = bisection(f,0.000001, 1,1000)  #0.0000001
        alpha_list.append(approx_alpha)
    
        # print(approx_alpha)
    
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
with open(resultdir+'test_assignments.npy', 'rb') as f:
    test_assignments = np.load(f)
with open(resultdir+'train_assignments.npy', 'rb') as f:
    train_assignments = np.load(f)
    
mu = np.array(X_train_main.mean(axis=0)).reshape((1, N))

data_assignment = train_assignments.argmax(0)
test_assignment = test_assignments.argmax(0)

data_assignment_rand = get_assignnments(train_assignments)
test_assignment_rand = get_assignnments(test_assignments)
# test_assignment_rand = np.array(random.choices(population=[0,1],weights=[0.5, 0.5],k=251))
print(sum(test_assignment_rand), len(test_assignment_rand))
print(sum(data_assignment_rand), len(data_assignment_rand))


assignments_df = pd.DataFrame(test_assignments).T
assignments_df['max_assign'] = test_assignment
assignments_df['random_assign'] = test_assignment_rand
assignments_df.to_csv(resultdir+'assignments_df'+str(alp_bisection)+'.csv')

prefixed = [filename for filename in os.listdir(resultdir) if filename.startswith("c_")]
n_class=len(prefixed)

#if pobability distribution is equal, randomly assign clusters
if len(set(data_assignment))!=n_class:
    data_assignment = np.array(random.choices(range(n_class), k=train_assignments.shape[1]))
    test_assignment = np.array(random.choices(range(n_class), k=test_assignments.shape[1]))
# #solve neural network

# analysisdir = path_generator(analysisdir)

analysis_list = []

for alp_bisection in [0,-1000]:
    for delta in [0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]:
        back_test_violations=[]
        R_mark=[]
        R_bootstrap=[]
        obj_mark=[]
        obj_bootstrap=[]
        test_vals_mark=[]
        test_vals_mark_rand=[]
        test_vals_bootstrap=[]
        for k in range(n_class):
            E = 2
            L = 2
        
            fileName = resultdir+"c_"+str(k)+".txt"
            c0 = np.genfromtxt(fileName, delimiter=',')
            
            
            fileName = resultdir+"cov_"+str(k)+".txt"
            sig=np.genfromtxt(fileName, delimiter=',')
            sig_inv = np.linalg.inv(sig)
            sig = sig_inv
                    
            listW = []
            dimLayers = []
            
            for F in range(0,L,1):
                fileName = resultdir+"W_"+str(k)+"_"+str(F)+".txt"
                listW.append(np.genfromtxt(fileName, delimiter=','))
                dimLayers.append(listW[F].shape[0])
            
                
            N=listW[0].shape[1]
            
            
            #@AC made X_train to X_train_main in the next 2 lines#
            maxScenEntry = max(np.amax(X_train_main),np.amax(X_train_main))
            maxEntry = max(np.amax(X_train_main),np.amax(X_train_main))
            M=[]
            for i in range(0,L,1):
                rowSums = np.sum(np.absolute(listW[i]),axis=1)
                M.append(maxEntry*np.amax(rowSums))
                maxEntry = maxEntry*np.amax(rowSums)
                
            
            
            X_train_hat=[]
            for j in range(0,X_train_main[np.where(data_assignment_rand==k)[0],:].shape[0],1):
                outLayer = X_train_main[np.where(data_assignment_rand==k)[0],:][j,:]
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

            #marks radius
            R_all_, R,sigmas,lb,ub = RO.getRadiiDataPoints(L,c0,sig, X_train_main[np.where(data_assignment==k)[0],:],listW, 0, delta)
            R_mark=R
            
            
            Rand_R_all_, Rand_R,Rand_sigmas,Rand_lb,Rand_ub = RO.getRadiiDataPoints(L,c0,sig, X_train_main[np.where(data_assignment_rand==k)[0],:],listW, 0, delta)
            # dist, _,_,_,_ = RO.getRadiiDataPoints(L,c0,X_train_main,listW, 0, delta)
                        
            X_train_temp = np.array(X_train_hat)
            X_train_assigned = X_train_main[np.where(data_assignment_rand==k)[0],:]
    
            outside  = [i for i in range(len(Rand_R_all_)) if Rand_R_all_[i] > Rand_R]
            outside_points = np.take(X_train_assigned, outside, 0)

            if alp_bisection==0:
                _, Rand_R,_,_,_ = RO.getRadiiDataPoints(L,c0,sig,X_train_main[np.where(data_assignment==k)[0],:],listW, 0, 0.99)
                
                alp = get_alpha_for_convex_hull(X_train_assigned,X_train_temp,mu_hat, c0,sig, Rand_R, delta)
            
            else:
                alp = -1000
    
            epsilon=delta/n_class
            # R_all,R_bootstrap = RO.bootstrap_radius(dist,list(train_assignments[k]), epsilon,delta)
            dist, _,_,_,_ = RO.getRadiiDataPoints(L,c0,sig,X_train_main,listW, 0, delta)
            R = RO.radius(Rand_R_all_,train_assignments[k,np.where(data_assignment_rand==k)[0]],delta)
            # print(Rand_R_all_)
            Rand_R=R

            # print(delta,R_bootstrap)
                # bootstrap_R[k]={delta:R}
            print('Class: {}\t Delta: {:.8f}\t Alpha: {:.8f}\t Radius: {:.8f}\t Rand Radius: {:.8f}'.format(k,delta,alp, R, Rand_R))
           
            start = time.time()
            p=1
            try:
                obj, x_mark = RO.solveRobustSelection(p,N,L,X_train_main[np.where(data_assignment_rand==k)[0],:],dimLayers,c0, mu_hat, sig, alp, R, listW, M, lb, ub,0, False, 0,sigmas)
                obj_mark=obj
                x_mark = 100*np.round(x_mark, 4)
            except:
                print('couldnt solve for x')
                continue
            print("x=",x_mark)
            
            Rand_obj, Rand_x_mark = RO.solveRobustSelection(p,N,L,X_train_main[np.where(data_assignment_rand==k)[0],:], dimLayers,c0, mu_hat, sig, alp, Rand_R, listW, M, lb, ub,0, False, 0,sigmas)
            Rand_obj_mark=Rand_obj
            Rand_x_mark = 100*np.round(Rand_x_mark, 4)
            print("Rand x=",Rand_obj_mark)

            end = time.time()
            t = end-start
            
            avg, maximum, vals = RO.evaluateSolution(x_mark,X_test_main[np.where(test_assignment==k)[0],:])
            test_vals_mark.extend(vals) 
    
            Rand_avg, Rand_maximum, Rand_vals = RO.evaluateSolution(Rand_x_mark,X_test_main[np.where(test_assignment_rand==k)[0],:])
            test_vals_mark_rand.extend(Rand_vals)    

            
            _,back_test=backtest_policy(vals,obj)
            back_test_violations.extend(back_test)
            csvFile = open(analysisdir+"IDCC_portfolio-val-cluster_"+str(year)+".csv", 'a')
            out = csv.writer(csvFile,delimiter=",", lineterminator = "\n")
            row = [seed,alp,net,alpha,beta,eps,lr,n_class,delta,k]
            row.append(R_mark)
            # row.append(R_bootstrap)
            row.append(obj_mark)
            row.append(x_mark)
            row.append(avg)
            row.append(maximum)
            row.append(testPolicyVaR(vals,delta))
            
            for i in range(len(vals)):
                row.append(vals[i])
            # row.append(test_vals_bootstrap)
            out.writerow(row)
            csvFile.close()
            
            
            csvFile = open(analysisdir+"IDCC_Rand_portfolio-val-cluster_"+str(year)+".csv", 'a')
            out = csv.writer(csvFile,delimiter=",", lineterminator = "\n")
            row = [seed,alp,net,alpha,beta,eps,lr,n_class,delta,k]
            row.append(Rand_R)
            # row.append(R_bootstrap)
            row.append(Rand_obj_mark)
            row.append(Rand_x_mark)
            row.append(Rand_avg)
            row.append(Rand_maximum)
            row.append(testPolicyVaR(Rand_vals,delta))
            print("VAR "+ str(testPolicyVaR(Rand_vals,delta)))
            for i in range(len(Rand_vals)):
                row.append(Rand_vals[i])
            # row.append(test_vals_bootstrap)
            out.writerow(row)
            csvFile.close()
            
        csvFile = open(analysisdir+"IDCC_portfolio-val-overall_"+str(year)+".csv", 'a')
        out = csv.writer(csvFile,delimiter=",", lineterminator = "\n")
        row = [seed,alp,net,alpha,beta,eps,lr,n_class,delta]
        row.append(testPolicyVaR(test_vals_mark,delta))
        
        for i in range(len(test_vals_mark)):
            row.append(test_vals_mark[i])
        # row.append(test_vals_mark)
        # row.append(testPolicyVaR(test_vals_bootstrap,delta))
        row.append(np.sum(back_test_violations)/len(test_vals_mark))
        out.writerow(row)
        csvFile.close()
        
        csvFile = open(analysisdir+"IDCC_Rand_portfolio-val-overall_"+str(year)+".csv", 'a')
        out = csv.writer(csvFile,delimiter=",", lineterminator = "\n")
        row = [seed,alp,net,alpha,beta,eps,lr,n_class,delta]
        row.append(testPolicyVaR(test_vals_mark_rand,delta))
        print("VAR "+ str(testPolicyVaR(test_vals_mark_rand,delta)))
        for i in range(len(test_vals_mark_rand)):
            row.append(test_vals_mark_rand[i])
        out.writerow(row)
        csvFile.close()
        
        csvFile = open(analysisdir+"consolidated.csv", 'a')
        out = csv.writer(csvFile,delimiter=",", lineterminator = "\n")
        row = [seed,year,beta,n_class,alp_bisection,delta]
        row.append(testPolicyVaR(test_vals_mark_rand,delta))
        out.writerow(row)
        csvFile.close()
        
df_mark=pd.read_csv(analysisdir+"port_mark.csv",header=None).tail(8)
df_mark.index=df_mark.iloc[:,3].tail(8)
df_mark=df_mark.iloc[:,11:]
try:
    df_mark_clus=pd.read_csv(analysisdir+"port_mark_cluster.csv").values.tolist()
except:
    df_mark_clus=[]
for k in set(test_assignment_rand):
    temp=df_mark.iloc[:,test_assignment_rand==k]
    for i in range(0,df_mark.shape[0]):
        df_mark_clus.append([seed,df_mark.index[i],k,temp.shape[1]/df_mark.shape[1],testPolicyVaR(temp.iloc[i,:],df_mark.index[i])])
    
pd.DataFrame(df_mark_clus,columns=['seed','quantile','cluster','ratio','VaR']).to_csv(analysisdir+"port_mark_cluster.csv",index=False)


df_elli=pd.read_csv(analysisdir+"port_ellipsoid.csv",header=None).tail(8)
df_elli.index=df_elli.iloc[:,2]
df_elli=df_elli.iloc[:,10:]
try:
    df_elli_clus=pd.read_csv(analysisdir+"port_eli_cluster.csv").values.tolist()
except:
    df_elli_clus=[]
for k in set(test_assignment_rand):
    temp=df_elli.iloc[:,test_assignment_rand==k]
    for i in range(0,df_elli.shape[0]):
        df_elli_clus.append([seed,df_elli.index[i],k,temp.shape[1]/df_mark.shape[1],testPolicyVaR(temp.iloc[i,:],df_elli.index[i])])
    
pd.DataFrame(df_elli_clus,columns=['seed','quantile','cluster','ratio','VaR']).to_csv(analysisdir+"port_eli_cluster.csv",index=False)
                    

